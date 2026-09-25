import torch
import numpy as np
import os
import matplotlib.pyplot as plt

import glob

from pynput import keyboard
from typing import List

import isaaclab.sim as sim_utils
from isaaclab.envs.direct_rl_env import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from jit_utils import torch_jit_utils as torch_utils
from components.frame_stack import FrameStack

import json

from pxr import UsdPhysics, UsdShade
import omni.usd

# from rl_games.algos_torch.torch_ext import AverageMeter

def get_env_bool(name, default=False):
    value = os.environ.get(name)

    if value is None:
        return default

    return value.lower() in ("1", "true", "yes", "on")

@configclass
class EventCfg:
    """Configuration for per-episode domain randomization."""

    wheel_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=r".*wheel.*",
            ),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 128,
            "make_consistent": True,
        },
    )


def make_robot_cfg(optimized: bool) -> ArticulationCfg:

    if optimized:
        usd_path = (
            os.getcwd()
            + "/assets/robomaster_optimized/robomaster_asset.usda"
        )
        roller_joint_expr = "RevoluteJoint_roller_.*"

    else:
        usd_path = (
            os.getcwd()
            + "/assets/robomaster/robomaster.usd"
        )
        roller_joint_expr = "RollerJoint_.*"

    if optimized:
        actuators = {
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_joint"],
                velocity_limit_sim=200.0,
                effort_limit_sim=1000.0,
                stiffness=0.0,
                damping=1e5,
            ),

            "rollers": ImplicitActuatorCfg(
                joint_names_expr=[roller_joint_expr],
                velocity_limit_sim=100000000.0,
                effort_limit_sim=1000.0,
                stiffness=0.0,
                damping=0.0,
            ),
        }

        actuators["suspension"] = ImplicitActuatorCfg(
            joint_names_expr=["Suspension_.*"],
            stiffness=3000.0,
            damping=100.0,
        )   

    else:
        actuators = {
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_joint"],
                velocity_limit_sim=200.0,
                effort_limit_sim=1000.0,
                stiffness=0.0,
                damping=1e5,
            ),

            "rollers": ImplicitActuatorCfg(
                joint_names_expr=[roller_joint_expr],
                velocity_limit_sim=100000000.0,
                effort_limit_sim=1000.0,
                stiffness=0.0,
                damping=0.0,
            ),
        }


    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",

        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,

            activate_contact_sensors=False,

            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(

            # All joints start at zero.
            #
            # This works for both models and avoids having to maintain
            # separate lists for wheels, rollers and optimized suspension.
            joint_pos={
                ".*": 0.0,
            },

            joint_vel={
                ".*": 0.0,
            },

            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),

        actuators=actuators,
    )

@configclass
class EdgeCasesEnvCfg(DirectRLEnvCfg):
    # if given, will override the device setting in gym.
    
    # Environment settings
    num_envs: int = 1024
    env_spacing: float = 5.00
    decimation: int = 8
    episode_length_s: int = 20
   

    enable_debug_vis: bool = False
    record_data: bool = False
    keyboard_agent: bool = False
    playback_data: bool = False
    record_vel_data: bool = False
    playback_vel_data: bool = False
    domain_rand: bool = get_env_bool("EDGECASE_DOMAIN_RAND", False)

    train_seed = 0

    optimized: bool = get_env_bool(
        "EDGECASE_OPTIMIZED",
        False,
    )

    use_ik: bool = get_env_bool("EDGECASE_USE_IK", False)
    offset: bool = get_env_bool("EDGECASE_OFFSET", False)

    # use_ik: bool = False
    # offset: bool = False

    events: EventCfg | None = EventCfg() if domain_rand else None


    start_position_noise: float = 0.0
    start_rotation_noise: float = 0.0

    frame_stack: int = 1
    history_size: int = 100
    action_space: int = 4
    # single_obs_size: int = 12
    # single_obs_size: int = 15
    single_obs_size: int = 19
    # single_obs_size: int = 24
    
    # observation_space: int = single_obs_size * frame_stack
    observation_space: int = single_obs_size * frame_stack
    num_states = 0
    state_space = 0

    action_scale: float = 0.1
    max_velocity: float = 100.0  # Rad/s
    # max_acceleration: List[float] = [260., 260., 260., 260.]
    max_acceleration: List[float] = [300., 300., 300., 300.]

    # Random command velocity ranges
    command_linear_x_range: List[float] = [-1.0, 1.0]
    command_linear_y_range: List[float] = [-1.0, 1.0]
    command_yaw_range: List[float] = [-3.14, 3.14]

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,
        render_interval=decimation,
        physx=PhysxCfg(
            # gpu_max_rigid_contact_count=4096 * 4096,
            # gpu_max_rigid_patch_count=1000000,
            gpu_max_rigid_contact_count = 2**23,
            gpu_max_rigid_patch_count= 12 * 2**15,
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=1,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)

    robot: ArticulationCfg = make_robot_cfg(optimized=optimized)

    # Terrain settings
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
    )

    # Reward settings
    terminal_reward: float = 0.0
    linear_velocity_x_reward_scale: float = 0.5
    linear_velocity_y_reward_scale: float = 0.5
    linear_velocity_x_y_reward_scale: float = 1.5
    angular_velocity_z_reward_scale: float = 2.0
    wheel_reward_scale: float = 2.5
    orientation_x_reward_scale: float = 0.5
    orientation_y_reward_scale: float = 0.5
    orientation_reward_scale: float = 2.5
    progress_reward_scale: float = 1.0
    off_road_reward: float = -500.0
    stationary_reward: float = -1.0
    action_rate_reward_scale: float = -0.01
    smoothness: float = 0.1
    survival: float = 0.05
    base_velocity_scale: float = 6.0
    wheel_velocity_scale: float = 0.0

    # Normalization
    linear_velocity_x_scale: float = 2.5
    linear_velocity_y_scale: float = 2.5
    angular_velocity_scale: float = 2.5

    # Noise settings
    add_noise: bool = False
    noise_Level: float = 1.0  # scales other values
    linear_velocity_noise: float = 0.001
    angular_velocity_noise: float = 0.03
    gravity_noise: float = 0.0002

class EdgeCasesEnv(DirectRLEnv):

    cfg: EdgeCasesEnvCfg

    def __init__(self, cfg: EdgeCasesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.use_ik = cfg.use_ik
        self.keyboard_agent = cfg.keyboard_agent
        self.offset= cfg.offset
        self.visualization = cfg.enable_debug_vis

        # ----------------------------------------------------------
        # Unexpected reset statistics
        # ----------------------------------------------------------

        self.unexpected_reset_count = 0

        self.unexpected_reset_count_per_env = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        # Optional output file used by run_edgecase_benchmarks.py
        self.reset_stats_file = os.environ.get(
            "EDGECASE_RESET_STATS_FILE"
        )

        self.rm_start_rot = torch.tensor((0, 0, 0, 1), device=self.device, 
                                         dtype=torch.float32).unsqueeze(0).repeat_interleave(self.num_envs, 0)
        self.start_position_noise = cfg.start_position_noise
        self.start_rotation_noise = cfg.start_rotation_noise

        self.action_scale = cfg.action_scale
        self.max_velocities = cfg.max_velocity
        self.max_acceleration = torch.tensor(cfg.max_acceleration, 
                                            device=self.device).repeat(self.num_envs, 1)
        self.frame_stack = 1
        self.history_size = cfg.history_size
        
        self.gravity_vec = self._robot.data.GRAVITY_VEC_W
        self.gravity_vec_rm_proj = self._robot.data.GRAVITY_VEC_W * -1 
        self.forward_vec = self._robot.data.FORWARD_VEC_B

        self.rm_dof_pos = self._robot.data.joint_pos.data 
        self.rm_dof_vel = self._robot.data.joint_vel.data
        if cfg.optimized:
            self.dof_wheel_indices = torch.LongTensor([4, 6, 7, 5])
        else:
            self.dof_wheel_indices = torch.LongTensor([0, 1, 2, 3])

        self.initial_dof_pos = torch.zeros_like(self.rm_dof_pos, device=self.device, dtype=torch.float)
        self.rm_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.commands_scale = torch.tensor([cfg.linear_velocity_x_scale, cfg.linear_velocity_y_scale, 
                                            cfg.angular_velocity_scale],
                                            device=self.device,
                                            requires_grad=False) 
        
        self.command_x_range = cfg.command_linear_x_range
        self.command_y_range = cfg.command_linear_y_range
        self.command_yaw_range = cfg.command_yaw_range


        self.rew_scales = {}
        self.rew_scales["termination"] = cfg.terminal_reward
        self.rew_scales["lin_vel_xy"] = cfg.linear_velocity_x_y_reward_scale
        self.rew_scales["lin_vel_x"] = cfg.linear_velocity_x_reward_scale
        self.rew_scales["lin_vel_y"] = cfg.linear_velocity_y_reward_scale
        self.rew_scales["ang_vel_z"] = cfg.angular_velocity_z_reward_scale
        self.rew_scales["wheel"] = cfg.wheel_reward_scale
        self.rew_scales["orient"] = cfg.orientation_reward_scale
        self.rew_scales["action_rate"] = cfg.action_rate_reward_scale
        self.rew_scales["base_vel"] = cfg.base_velocity_scale / self.step_dt
        self.rew_scales["wheel_vel"] = cfg.wheel_velocity_scale / self.step_dt
        
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.step_dt

        self.initial_base_orientation = self._robot.data.root_quat_w.clone()
        self.initial_base_position = self._robot.data.root_pos_w.clone()

        self.command_velocities = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_command_velocities = torch.zeros_like(self.command_velocities)
        self._previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.clipped_velocities = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_clipped_velocities = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.wheel_seperation_width = 0.1
        self.wheel_seperation_length = 0.1
        self.wheel_radius = 0.05
        self.wheel_velocities_matrix = torch.ones((4, 3), device=self.device)
        # self.wheel_velocities_matrix[0, 1] = self.wheel_velocities_matrix[2, 1] = -1
        # self.wheel_velocities_matrix[2, 2] = self.wheel_velocities_matrix[1, 2] = (
        #             self.wheel_seperation_width + self.wheel_seperation_length)
        # self.wheel_velocities_matrix[3, 2] = self.wheel_velocities_matrix[0, 2] = -(
        #             self.wheel_seperation_width + self.wheel_seperation_length)
        # self.wheel_velocities_matrix = (1 / self.wheel_radius) * self.wheel_velocities_matrix

        self.wheel_velocities_matrix[0, 1] = self.wheel_velocities_matrix[2, 1] = -1
        self.wheel_velocities_matrix[3, 2] = self.wheel_velocities_matrix[2, 2] = (
                    self.wheel_seperation_width + self.wheel_seperation_length)
        self.wheel_velocities_matrix[0, 2] = self.wheel_velocities_matrix[1, 2] = -(
                    self.wheel_seperation_width + self.wheel_seperation_length)
        self.wheel_velocities_matrix = (1 / self.wheel_radius) * self.wheel_velocities_matrix


        if self.keyboard_agent:
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
        
        if self.frame_stack != 1:
            self.observation_stack = FrameStack(self.num_envs, self.frame_stack, cfg.single_obs_size, self.device)

        self.observation_history = torch.zeros(self.num_envs, self.history_size + 1, cfg.single_obs_size + cfg.action_space)
        # print(f'History: {self.observation_history.shape}')

        self.robot_heading = torch.zeros((self.num_envs,))

        
        self.rot_matrix_z = torch.zeros((self.num_envs, 2, 2), dtype=torch.float64, device=self.device)
        
        # folder_path = f'trajectories_new_256{self.trajectory_speed}'
        # folder_path = f'multiple_trajectories_{self.num_envs}'
        folder_path = f'edge_cases'
        # self.paths = [
        #         f'{folder_path}/circle_big_radius.csv', #
        #         f'{folder_path}/circle_small_radius.csv', #
        #         f'{folder_path}/curve_left_90.csv', #
        #         f'{folder_path}/curve_right_90.csv', #
        #         f'{folder_path}/forward_fixed_orientation_-90deg.csv', #
        #         f'{folder_path}/forward_fixed_orientation_0deg.csv', #
        #         f'{folder_path}/forward_fixed_orientation_45deg.csv', #
        #         f'{folder_path}/forward_fixed_orientation_90deg.csv', #
        #         f'{folder_path}/forward_fixed_orientation_180deg.csv', #
        #         f'{folder_path}/forward_then_backward.csv', #
        #         f'{folder_path}/forward_then_stop.csv', #
        #         f'{folder_path}/reverse_while_turning.csv', #
        #         f'{folder_path}/spin_in_place.csv', #
        #         f'{folder_path}/spinning_while_forward.csv', #
        #         f'{folder_path}/zigzag_lateral_motion.csv', #
        #     ]

        self.paths = sorted(glob.glob(f"{folder_path}/*.csv"))

        # traj_idx = 11
        # self.trajectory_path = paths[traj_idx]

        # # self.trajectory = torch.from_numpy(np.genfromtxt('trajectories/mecanum_path.csv', delimiter=',', skip_header=1)).to(self.device)[:, 3:]
        trajectories_length = 100
        # # trajectories_length = 700
        # # self.trajectories = torch.zeros((self.num_envs, trajectories_length, 3))
        # self.init_orientations_xyzw = torch.zeros((self.num_envs, 4), device=self.device)
        
        # trajectory = torch.from_numpy(np.genfromtxt(self.trajectory_path, delimiter=',', skip_header=1)).to(self.device)
        # self.trajectories  = trajectory[:, :3].unsqueeze(0)
        
        # yaw_angle = trajectory[0, 3]
        # quat_wxyz = torch_utils.quat_from_euler_xyz(
        #     torch.tensor([0.0], device=self.device),
        #     torch.tensor([0.0], device=self.device),
        #     yaw_angle.view(1)
        # )  # returns [w, x, y, z]

        # self.init_orientations_xyzw = quat_wxyz[:, [3, 0, 1, 2]].squeeze(0).float().to(self.device)

        num_edge_cases = len(self.paths)
        assert self.num_envs == num_edge_cases, (
            f"Expected num_envs={num_edge_cases}, but got {self.num_envs}. "
            "Set cfg.num_envs to the number of edge-case CSV files."
        )

        self.trajectories = torch.zeros((self.num_envs, trajectories_length, 3))
        self.init_orientations_xyzw = torch.zeros((self.num_envs, 4), device=self.device)
        for i, path in enumerate(self.paths):
            trajectory_np = torch.from_numpy(np.genfromtxt(path, delimiter=",", skip_header=1)).to(self.device)

            # CSV columns: vx, vy, omega, theta
            self.trajectories[i]  = trajectory_np[:trajectories_length, :3]
            yaw_angle = trajectory_np[0, 3]

            quat_wxyz = torch_utils.quat_from_euler_xyz(
                        torch.tensor([0.0], device=self.device),
                        torch.tensor([0.0], device=self.device),
                        yaw_angle.view(1)
                    )  # returns [w, x, y, z]

            self.init_orientations_xyzw[i] = quat_wxyz[:, [3, 0, 1, 2]].squeeze(0).float().to(self.device)

        self.trajectory_idx = 0
        self.positions = torch.zeros((self.num_envs, trajectories_length, 3))
        self.velocity_profiles = torch.zeros_like(self.positions)
        self.reference_vels = torch.zeros_like(self.positions)

        self.record_name = f'trajectory_'
        self.ref_record_name = f'trajectory_reference'
        if self.use_ik:
            self.record_name += 'ik'
        elif self.offset:
            self.record_name += 'offset'
        else:
            self.record_name += 'policy'
    

        self.decimation = cfg.decimation
        self.compute_reference_positions()
        if self.visualization:
            self.current_env_idx = 0
            self.setup_live_plot()

        self._wheel_joint_ids, self._wheel_joint_names = self._robot.find_joints(
            [
                "wheel_fl_joint",
                "wheel_fr_joint",
                "wheel_rl_joint",
                "wheel_rr_joint",
            ],
            preserve_order=True,
        )

        print("Wheel IDs:", self._wheel_joint_ids)
        print("Wheel names:", self._wheel_joint_names)

        print("\nALL JOINTS:")
        for i, name in enumerate(self._robot.joint_names):
            print(i, name)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _get_observations(self):
        """Return state observations for RL."""

        base_linear_velocity = self._robot.data.root_lin_vel_b
        base_angular_velocity = self._robot.data.root_ang_vel_b

        dt = self.step_dt

        base_lin_acc = (base_linear_velocity - self.prev_base_lin_vel)
        base_ang_acc_z = (base_angular_velocity[:, 2] - self.prev_base_ang_vel[:, 2]) 


        projected_gravity = self._robot.data.projected_gravity_b
        

        ### Single Obs Size 12 ###
        base_velocity = torch.cat((base_linear_velocity[:, :2], base_angular_velocity[:, 2].unsqueeze(-1)), dim=1)
        current_cmd = self.command_velocities[:, :3] * self.commands_scale
        tracking_error = current_cmd - base_velocity
        
       

        observations = torch.cat((base_velocity, base_lin_acc[:, :2], base_ang_acc_z.unsqueeze(-1),
                                  projected_gravity, current_cmd, tracking_error, self._actions), dim=-1)
        
        if self.frame_stack != 1:
            self.observation_stack.add_obs_history(observations)
            observations = self.observation_stack.return_obs()

        self.prev_base_lin_vel = base_linear_velocity.clone()
        self.prev_base_ang_vel = base_angular_velocity.clone()


        return {"policy": observations}


    def setup_live_plot(self):
        plt.ion()  # interactive mode on
        self.fig, self.ax = plt.subplots()
        self.ref_line, = self.ax.plot([], [], 'g--', label="Reference Trajectory")
        self.actual_line, = self.ax.plot([], [], 'b-', label="Actual Trajectory")
        self.start_dot, = self.ax.plot([], [], 'bo', label="Start")

        self.heading_arrow = self.ax.arrow(-1, 0, 0, 0, 
                                   head_width=0.1,
                                   head_length=0.2,
                                   fc='r', ec='r')
        
        self.mode = ""
        if self.use_ik:
            self.mode = "IK"
        elif self.offset:
            self.mode = "Offset Policy"
        else:
            self.mode = "Full Policy"

        
        self.ax.set_title(f"Trajectory Comparison (Env 0) {self.mode}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.legend()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def update_live_plot(self, env_idx=0):
        actual = np.array(self.actual_positions[env_idx])

        ref = self.reference_positions[:, env_idx, :2]  # shape (T, 2)

        self.ref_line.set_data(ref[:, 0], ref[:, 1])

        self.heading_arrow.remove()

        if actual.shape[0] > 0:
            x, y = actual[-1]  # robot's current position
            theta = self.robot_heading.detach().cpu().numpy()[env_idx]  # heading angle in radians

            dx = -np.cos(-theta) * 0.5  # arrow length
            dy = -np.sin(-theta) * 0.5

            # Create new arrow
            self.heading_arrow = self.ax.arrow(
                x, y,
                dx, dy,
                head_width=0.1,
                head_length=0.2,
                fc='r', ec='r'
            )

            self.actual_line.set_data(actual[:, 0], actual[:, 1])
            self.start_dot.set_data([actual[0, 0]], [actual[0, 1]])

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



    def compute_reference_positions(self, initial_pos=None):
        """
        Recomputes reference positions from velocity commands.
        Initial x,y come from initial_pos.
        Initial heading comes from self.init_orientations_xyzw.
        """

        if initial_pos is None:
            initial_xy = np.zeros((self.num_envs, 2), dtype=np.float32)
        else:
            initial_xy = initial_pos[:, :2].copy()

        # initial yaw from stored quaternions
        _, _, yaw = torch_utils.get_euler_xyz(self.init_orientations_xyzw[:, [1,2,3,0]])
        init_yaw = yaw.cpu().numpy()

        # pose = [x, y, theta]
        pose = np.zeros((self.num_envs, 3), dtype=np.float32)
        pose[:, :2] = initial_xy
        pose[:, 2] = init_yaw

        positions = [pose.copy()]

        trajs_np = self.trajectories.cpu().numpy().transpose(1, 0, 2)  # (T, num_envs, 3)

        sx = self.commands_scale[0].cpu().numpy()
        sy = self.commands_scale[1].cpu().numpy()
        sw = self.commands_scale[2].cpu().numpy()

        for vxyw in trajs_np:  # vxyw shape: (num_envs, 3)
            vx, vy, omega = vxyw.T

            theta = pose[:, 2]

            dx = (np.cos(theta) * vx - np.sin(theta) * vy) * self.step_dt * sx
            dy = (np.sin(theta) * vx + np.cos(theta) * vy) * self.step_dt * sy
            dtheta = omega * self.step_dt * sw

            pose[:, 0] += dx
            pose[:, 1] += dy
            pose[:, 2] += dtheta

            pose[:, 2] = torch_utils.wrap_to_pi(torch.tensor(pose[:, 2]))

            positions.append(pose.copy())

        self.reference_positions = np.array(positions)


    def _pre_physics_step(self, actions):
        self._prev_actions = self._actions 
        self._actions = actions.clone().clamp(-1.0, 1.0).to(self.device)

        
        self.command_velocities[:, :3] = self.trajectories[:, self.trajectory_idx]
        
        if not hasattr(self, "actual_positions"):
            self.actual_positions = [[] for _ in range(self.num_envs)]

        root_pos = self._robot.data.root_pos_w  # shape: (num_envs, 3)
        for i in range(self.num_envs):
            self.actual_positions[i].append(root_pos[i, :2].cpu().numpy())
        if self.visualization:
            self.update_live_plot(env_idx=self.current_env_idx)

        self.prev_clipped_velocities = self.clipped_velocities


    def _apply_action(self):
        # Convert actions into wheel velocities (this needs adjustment for Mercanum wheels)
        if self.use_ik:
            wheel_actions = torch_utils.get_wheel_velocities(torch.transpose(self.command_velocities[:, :3] * self.commands_scale, 0, -1), 
                                               self.wheel_velocities_matrix).clamp(-self.max_velocities, self.max_velocities) 
            # print(wheel_actions[16])
        else:
            if self.offset:
                offset = self._actions *  (self.max_velocities * self.action_scale)
                wheel_actions = torch_utils.get_wheel_velocities(torch.transpose(self.command_velocities[:, :3] * self.commands_scale, 0, -1), 
                                               self.wheel_velocities_matrix).clamp(-self.max_velocities, self.max_velocities) + offset
            else:
                wheel_actions = self._actions * self.max_velocities
        
        # print(f'Should: {torch_utils.get_wheel_velocities(torch.transpose(self.command_velocities[:, :3] * self.commands_scale, 0, -1), self.wheel_velocities_matrix).clamp(-self.max_velocities, self.max_velocities)[0] }')
        # print(f'Is: {wheel_actions[0]}')
        self.rm_dof_targets[:] = 0.0
        accelerations = wheel_actions - self.rm_dof_vel[:, self.dof_wheel_indices]
        accelerations[accelerations < -self.max_acceleration] = -self.max_acceleration[accelerations < -self.max_acceleration]
        accelerations[accelerations > self.max_acceleration] = self.max_acceleration[accelerations > self.max_acceleration]

        self.clipped_velocities = (self.rm_dof_vel[:, self.dof_wheel_indices] 
                              + accelerations).clamp(-self.max_velocities, self.max_velocities)
        # self.clipped_velocities = elf.rm_dof_vel[:, self.dof_wheel_indices] + accelerations
        # print(clipped_velocities)

        self.rm_dof_targets[:, self.dof_wheel_indices[0]] = self.clipped_velocities[:, 0]  # FL Wheel
        self.rm_dof_targets[:, self.dof_wheel_indices[1]] = self.clipped_velocities[:, 1]  # RL Wheel
        self.rm_dof_targets[:, self.dof_wheel_indices[2]] = self.clipped_velocities[:, 2]  # RR Wheel
        self.rm_dof_targets[:, self.dof_wheel_indices[3]] = self.clipped_velocities[:, 3]  # FR Wheel

        self._robot.set_joint_velocity_target(self.rm_dof_targets[:, self.dof_wheel_indices], joint_ids=self.dof_wheel_indices)

    
    def _get_dones(self):
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        
        base_quat = self._robot.data.root_quat_w[:, [3, 0, 1, 2]]
        forward = torch_utils.quat_apply(base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        _, _, base_rot = torch_utils.get_euler_xyz(base_quat)
        self.robot_heading = base_rot
        
        
            
        root_quat = self._robot.data.root_quat_w[..., [1, 2, 3, 0]]
        actual_head = torch_utils.wrap_to_pi(torch_utils.get_euler_xyz(root_quat)[2])
        self.positions[:, self.trajectory_idx] = torch.cat((self._robot.data.root_pos_w[:, :2], actual_head.unsqueeze(-1)), dim=1) 
        self.velocity_profiles[:, self.trajectory_idx] = torch.cat(
            (self._robot.data.root_lin_vel_b[:, :2], self._robot.data.root_ang_vel_b[:, 2].unsqueeze(-1)), dim=-1)
        self.reference_vels[:, self.trajectory_idx] = self.command_velocities[:, :3]
        if self.trajectory_idx == self.trajectories.shape[1]-1:
            self.save_reset_statistics()

            for i, saving_dir in enumerate(self.paths):
                abort = i == len(self.paths)-1
                if self.visualization:
                    self.update_live_plot(env_idx=i)
                    self.ax.set_title(f"Trajectory Comparison (Env {i}) {self.mode}")
                # directory = f'records_new_256{self.trajectory_speed}/{saving_dir}'
                if self.cfg.domain_rand:
                    dom_rand = 'rand'
                else:
                    dom_rand = 'no_rand'
            
                directory = f'record/{saving_dir[:-4]}'
                save_as_csv(self.positions[i], directory, self.record_name, save_fig=True)
                save_as_csv(self.velocity_profiles[i], directory, self.record_name+"_vel", save_fig=False)
                save_as_csv(self.reference_vels[i], directory, self.ref_record_name+"_vel", save_fig=False)
                save_as_csv(self.reference_positions[1:, i], directory, self.ref_record_name, abort=abort)
        self.trajectory_idx = (self.trajectory_idx + 1) % (self.trajectories.shape[1])
        # print(self._actions[:3])
        # print(self.trajectory_idx)
            
        

        reset = torch.zeros(self.num_envs, device=self.device)
        projected_gravity = self._robot.data.projected_gravity_b
        err = (projected_gravity * self.gravity_vec_rm_proj).sum(dim=1)
        death_mask = (err + 1.0).abs() > 0.2
        # ----------------------------------------------------------
        # Count UNEXPECTED resets
        # ----------------------------------------------------------

        unexpected_reset_mask = (
            death_mask
            & (~truncated.bool())
        )

        num_unexpected_resets = int(
            unexpected_reset_mask.sum().item()
        )

        if num_unexpected_resets > 0:

            self.unexpected_reset_count += (
                num_unexpected_resets
            )

            self.unexpected_reset_count_per_env += (
                unexpected_reset_mask.long()
            )

            reset_env_ids = torch.nonzero(
                unexpected_reset_mask,
                as_tuple=False,
            ).squeeze(-1)

            reset_names = [
                os.path.basename(
                    self.paths[int(env_id)]
                ).replace(".csv", "")
                for env_id in reset_env_ids
            ]

            print()
            print(
                "[UNEXPECTED RESET] "
                f"+{num_unexpected_resets} | "
                f"Total: {self.unexpected_reset_count}"
            )

            print(
                "Affected edge cases:",
                ", ".join(reset_names),
            )

        reset = torch.where(death_mask, torch.ones_like(reset), reset)


        self._previous_actions = self._actions.clone()

        return truncated, reset
    
    
    def _get_rewards(self):
        target_vel = self.command_velocities[:, :3] * self.commands_scale
        
        root_lin_vels = self._robot.data.root_lin_vel_b
        root_ang_vels = self._robot.data.root_ang_vel_b
        x_vel_error = torch.square(target_vel[:, 0] - root_lin_vels[:, 0])
        y_vel_error = torch.square(target_vel[:, 1] - root_lin_vels[:, 1])
        ang_vel_error = torch.square((target_vel[:, 2] - root_ang_vels[:, 2]).clip(-0.5, 0.5))
        rew_lin_vel_x = torch.exp(-x_vel_error / 0.25) * self.rew_scales['lin_vel_x']
        rew_lin_vel_y = torch.exp(-y_vel_error / 0.25) * self.rew_scales['lin_vel_y']
        rew_lin_vel_xy = (rew_lin_vel_x + rew_lin_vel_y) / (
                    (self.rew_scales['lin_vel_x'] + self.rew_scales['lin_vel_y']) / self.rew_scales['lin_vel_xy'])
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales['ang_vel_z']

        vel_reward = rew_lin_vel_xy + rew_ang_vel_z

        target_wheel_speed = torch_utils.get_wheel_velocities(
            torch.transpose(target_vel, 0, -1), self.wheel_velocities_matrix).clamp(-self.max_velocities, self.max_velocities) / self.max_velocities
        
        current_wheel_speed = self._actions

        rew_wheels = torch.sum(torch.square(target_wheel_speed - current_wheel_speed), dim=1)
        wheels_reward = torch.exp(-rew_wheels / 0.25) * self.rew_scales['wheel']
        reward = self.rew_scales['base_vel'] * vel_reward + self.rew_scales['wheel_vel'] * wheels_reward

        # death_mask = (err + 1.0).abs() > 0.2
        # reward[death_mask] -= 500.0
        # self.rewards[self.current_steps_after_reset, :] = reward
        
        return reward
    

    def _reset_idx(self, env_ids):
        """Reset specific environments when necessary."""
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self.prev_base_lin_vel = self._robot.data.root_lin_vel_b.clone()
        self.prev_base_ang_vel = self._robot.data.root_ang_vel_b.clone()
        rnd_floats = -self.start_rotation_noise + (2.0 * self.start_rotation_noise * torch.rand(len(env_ids),
                                                                                                    device=self.device,
                                                                                                    dtype=torch.float32))
        random_poses = torch_utils.randomize_rotation(rnd_floats, self.gravity_vec_rm_proj[env_ids])
        
        positions = self.initial_base_position.clone()
        orientations = self.initial_base_orientation.clone()


        positions[env_ids, :2] += -self.start_position_noise + (
                2 * self.start_position_noise * torch.rand((len(env_ids), 2), device=self.device, dtype=torch.float32))

        orientations[env_ids, :] = torch_utils.quat_mul(self.rm_start_rot[env_ids], random_poses)
        orientations = orientations[:, [3, 0, 1, 2]] # change to wxyz -> write_root_pose_to_sim changes to xyzw

        # orientations[env_ids, :] = self.init_orientations_xyzw[:]
        orientations[env_ids, :] = self.init_orientations_xyzw[env_ids]
        positions[env_ids, :2] = torch.zeros_like(positions[env_ids, :2])

        _, _, base_rot = torch_utils.get_euler_xyz(orientations)
        self.robot_heading = base_rot
        
        pose = torch.cat((positions, orientations), dim=1)
        velocities = torch.cat((torch.zeros_like(self._robot.data.root_lin_vel_b), 
                                  torch.zeros_like(self._robot.data.root_ang_vel_b)), dim=1)
        
        self.rm_dof_pos[env_ids] = self.initial_dof_pos[env_ids]
        self.rm_dof_vel[env_ids] = 0

        self._robot.set_joint_position_target(self.rm_dof_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(self.rm_dof_pos.clone()[env_ids], self.rm_dof_vel.clone()[env_ids], env_ids=env_ids)
        self._robot.write_root_pose_to_sim(pose[env_ids], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(velocities[env_ids], env_ids=env_ids)

        self.actual_positions = [[] for _ in range(self.num_envs)]
        
    def on_key_press(self, event):
        if event.key == 'd':
            self.current_env_idx = (self.current_env_idx + 1) % self.num_envs
        elif event.key == 'a':
            self.current_env_idx = (self.current_env_idx - 1) % self.num_envs

        self.ax.set_title(f"Trajectory Comparison (Env {self.current_env_idx}) {self.mode}")
        if self.visualization:
            self.update_live_plot(env_idx=self.current_env_idx)

    def save_reset_statistics(self):

        total = int(self.unexpected_reset_count)

        per_edgecase = {}

        for env_id in range(self.num_envs):

            count = int(
                self.unexpected_reset_count_per_env[env_id].item()
            )

            edgecase_name = os.path.basename(
                self.paths[env_id]
            ).replace(".csv", "")

            per_edgecase[edgecase_name] = count

        print()
        print("=" * 80)
        print("UNEXPECTED RESET STATISTICS")
        print("=" * 80)
        print(f"Total unexpected resets: {total}")

        for edgecase_name, count in per_edgecase.items():
            if count > 0:
                print(
                    f"  {edgecase_name}: {count}"
                )

        if total == 0:
            print("No unexpected resets occurred.")

        print("=" * 80)
        print()

        # Used by the outer benchmark runner
        if self.reset_stats_file is not None:

            directory = os.path.dirname(
                self.reset_stats_file
            )

            if directory:
                os.makedirs(
                    directory,
                    exist_ok=True,
                )

            with open(
                self.reset_stats_file,
                "w",
            ) as f:

                json.dump(
                    {
                        "total": total,
                        "per_edgecase": per_edgecase,
                    },
                    f,
                    indent=4,
                )

def save_as_csv(tensor, dir, name, abort=False, save_fig=False):
    try:
        array = tensor.cpu().numpy()
    except:
        array = tensor
    os.makedirs(dir, exist_ok=True)
    np.savetxt(f'{dir}/{name}.csv', array, delimiter=',')
    if abort:
        print('Saved ... ')
        # raise not NotImplementedError('')
        raise SystemExit(0)
    else:
        if save_fig:
            plt.savefig(f'{dir}/{name}.png')




        