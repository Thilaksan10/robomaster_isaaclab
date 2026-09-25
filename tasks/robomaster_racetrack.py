import torch
import numpy as np
import os
import matplotlib.pyplot as plt

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
from components.trajectory_generator import create_original_racetrack, get_track_mask, extract_centerline_skeleton, ordered_centerline_points, shift_path_randomly, compute_optimal_line, compute_optimal_line_mecanum
from components.racetracks import create_racetrack
from components.training_generator import generate_dataset

from rl_games.algos_torch.torch_ext import AverageMeter


@configclass
class EventCfg:
    """Configuration for per-episode domain randomization."""


    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    r".*wheel.*",
                ],
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
                damping=0.01,
            ),

            "rollers": ImplicitActuatorCfg(
                joint_names_expr=[roller_joint_expr],
                velocity_limit_sim=100000000.0,
                effort_limit_sim=1000.0,
                stiffness=0.0,
                damping=0.5,
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
class RacetrackEnvCfg(DirectRLEnvCfg):
    # if given, will override the device setting in gym.
    
    # Environment settings
    num_envs: int = 20
    env_spacing: float = 5.00
    decimation: int = 8
    episode_length_s: int = 6
    track_length: int = 700
    omni_ratio: float = 0.0
   

    enable_debug_vis: bool = False
    record_data: bool = False
    playback_data: bool = False
    optimized: bool = True
    record_vel_data: bool = False
    playback_vel_data: bool = False
    new_data = True
    use_ik: bool = False
    omni: bool = True
    curriculum: bool = False


    offset: bool = False
    domain_rand: bool = False
    events: EventCfg | None = None

    map_size: int = 6
    boundary_size: int = 3
    cell_size: float = 0.4

    start_position_noise: float = 0.0
    start_rotation_noise: float = 0.0

    action_space: int = 4
    # single_obs_size: int = 12
    # single_obs_size: int = 15
    # single_obs_size: int = 16
    single_obs_size: int = 19
    
    
    observation_space: int = single_obs_size
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
    linear_velocity_x_reward_scale: float = 0.5
    linear_velocity_y_reward_scale: float = 0.5
    linear_velocity_x_y_reward_scale: float = 2.0
    angular_velocity_z_reward_scale: float = 1.0
    wheel_reward_scale: float = 0.0
    orientation_x_reward_scale: float = 0.5
    orientation_y_reward_scale: float = 0.5
    orientation_reward_scale: float = 1.0
    position_reward_scale: float = 0.0
    energy: float = 0.0
    smoothness: float = 0.0
    symmetry: float = 0.0
    base_velocity_scale: float = 1.0
    wheel_velocity_scale: float = 0.0
    linear_position_x_y_reward_scale: float = 2.0
    linear_position_x_reward_scale: float = 0.5
    linear_position_y_reward_scale: float = 0.5
    angular_heading_z_reward_scale: float = 1.0
    position_scale: float = 0.0

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


class RacetrackEnv(DirectRLEnv):

    cfg: RacetrackEnvCfg

    def __init__(self, cfg: RacetrackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.use_ik = cfg.use_ik
        self.omni = cfg.omni
        self.omni_ratio = cfg.omni_ratio
        self.curriculum = cfg.curriculum
        self.offset= cfg.offset
        self.visualization = cfg.enable_debug_vis

        self.track_length = cfg.track_length

        # self.rm_start_rot = torch.tensor((0, 0, 0, 1), device=self.device, 
        #                                  dtype=torch.float32).unsqueeze(0).repeat_interleave(self.num_envs, 0)
        self.rm_start_rot = torch.tensor((0,0,0,1), device=self.device).expand(self.num_envs,4)

        self.start_position_noise = cfg.start_position_noise
        self.start_rotation_noise = cfg.start_rotation_noise

        self.action_scale = cfg.action_scale
        self.max_velocities = cfg.max_velocity
        self.max_acceleration = torch.tensor(cfg.max_acceleration, 
                                            device=self.device).repeat(self.num_envs, 1)
        
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
        self.rew_scales["lin_vel_xy"] = cfg.linear_velocity_x_y_reward_scale
        self.rew_scales["lin_vel_x"] = cfg.linear_velocity_x_reward_scale
        self.rew_scales["lin_vel_y"] = cfg.linear_velocity_y_reward_scale
        self.rew_scales["ang_vel_z"] = cfg.angular_velocity_z_reward_scale
        self.rew_scales["wheel"] = cfg.wheel_reward_scale
        self.rew_scales["orient"] = cfg.orientation_reward_scale
        self.rew_scales["wheel_energy"] = cfg.energy
        self.rew_scales["action_smooth"] = cfg.smoothness
        self.rew_scales["wheel_sym"] = cfg.symmetry
        self.rew_scales["position_xy"] = cfg.position_reward_scale
        self.rew_scales["base_vel"] = cfg.base_velocity_scale / self.step_dt
        self.rew_scales["wheel_vel"] = cfg.wheel_velocity_scale / self.step_dt
        self.rew_scales["lin_pos_xy"] = cfg.linear_position_x_y_reward_scale
        self.rew_scales["lin_pos_x"] = cfg.linear_position_x_reward_scale
        self.rew_scales["lin_pos_y"] = cfg.linear_position_y_reward_scale
        self.rew_scales["ang_head_z"] = cfg.angular_heading_z_reward_scale
        self.rew_scales["position"] = cfg.position_scale / self.step_dt

        
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.step_dt

        self.initial_base_orientation = self._robot.data.root_quat_w.clone()
        self.initial_base_position = self._robot.data.root_pos_w.clone()

        self.command_velocities = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_command_velocities = torch.zeros_like(self.command_velocities)
        self._previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.clipped_velocities = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_clipped_velocities = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.wheel_seperation_width = 0.1
        self.wheel_seperation_length = 0.1
        self.wheel_radius = 0.05
        self.wheel_velocities_matrix = torch.ones((4, 3), device=self.device)

        self.wheel_velocities_matrix[0, 1] = self.wheel_velocities_matrix[2, 1] = -1
        self.wheel_velocities_matrix[3, 2] = self.wheel_velocities_matrix[2, 2] = (
                    self.wheel_seperation_width + self.wheel_seperation_length)
        self.wheel_velocities_matrix[0, 2] = self.wheel_velocities_matrix[1, 2] = -(
                    self.wheel_seperation_width + self.wheel_seperation_length)
        self.wheel_velocities_matrix = (1 / self.wheel_radius) * self.wheel_velocities_matrix


        self.obs_buffer = torch.zeros((self.num_envs, cfg.single_obs_size), device=self.device)

        self.rot_matrix_z = torch.zeros((self.num_envs, 2, 2), dtype=torch.float32, device=self.device)

        map_x, map_y, vel_map, ori_map = create_original_racetrack(cfg.map_size, cfg.cell_size, cfg.boundary_size)
        # self.max_speed = max(cfg.linear_velocity_x_scale, cfg.linear_velocity_y_scale, cfg.angular_velocity_scale)

        track_mask = get_track_mask(vel_map)
        skeleton = extract_centerline_skeleton(track_mask)
        centerline_px = ordered_centerline_points(skeleton)

        # Convert from pixel coords [row, col] to world coords [x, y]
        self.centerline_m_orig = np.stack([
            centerline_px[:, 1] * cfg.cell_size,
            centerline_px[:, 0] * cfg.cell_size
        ], axis=-1)

        self.trajectories = torch.zeros((self.num_envs, self.max_episode_length, 3), device=self.device)
        self.init_orientations_xyzw = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_heading = torch.zeros((self.num_envs,))
        # self.generate_trajectories()
            
        self.actual_positions = torch.zeros(
            (self.num_envs, self.max_episode_length, 2),
            device=self.device,
            dtype=torch.float32,
        )

        self.actual_pos_idx = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
                
        self.drive_type = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.prev_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)


        self.trajectory_idx = 0
        # self.trajectory_idx = torch.zeros(1, dtype=torch.long, device=self.device)

        #### Racetrack Pool ####
        self.trajectory_pool_size = self.num_envs * 8

        self.trajectory_pool = torch.zeros(
            (self.trajectory_pool_size, self.max_episode_length, 3),
            device=self.device
        )

        self.orientation_pool = torch.zeros(
            (self.trajectory_pool_size, 4),
            device=self.device
        )

        # if not self.cfg.new_data:
        #     for k in range(self.trajectory_pool_size):

        #         centerline_m = shift_path_randomly(self.centerline_m_orig.copy())
        #         if np.random.rand() < 0.5:
        #             centerline_m = centerline_m[::-1]

        #         # _, opt_vel, _, opt_cmds, theta0 = compute_optimal_line(
        #         #     centerline_m, num_points=self.track_length
        #         # )

        #         _, opt_vel, _, opt_cmds, theta0 = compute_optimal_line_mecanum(
        #             2.5, centerline_m, num_points=self.track_length
        #         )

        #         traj = torch.from_numpy(opt_cmds[:self.max_episode_length]).to(self.device)
        #         self.trajectory_pool[k] = traj

        #         if not self.omni:
        #             vx, vy = opt_vel[0, 0], opt_vel[0, 1]
        #             heading_vec = torch.tensor([vx, vy], device=self.device)[:self.max_episode_length]

        #             # Normalize and compute yaw angle
        #             heading_vec = heading_vec / torch.norm(heading_vec)
        #             yaw_angle = torch.atan2(heading_vec[1], heading_vec[0])
        #         else:
        #             yaw_angle = torch.tensor(theta0, device=self.device)

        #         quat_wxyz = torch_utils.quat_from_euler_xyz(
        #             torch.tensor([0.0], device=self.device),
        #             torch.tensor([0.0], device=self.device),
        #             yaw_angle.view(1),
        #         )

        #         self.orientation_pool[k] = quat_wxyz[:, [3,0,1,2]].squeeze(0)
        # else:
        #     generated_trajectories = generate_dataset(self.trajectory_pool_size, sample=False, save_plots=False)
        #     for k, generated_trajectory in enumerate(generated_trajectories):
        #         linear_velocties = generated_trajectory['vel']
        #         angular_velocities = generated_trajectory['omega']
        #         traj = torch.from_numpy(np.column_stack((linear_velocties, angular_velocities))).to(self.device)
        #         self.trajectory_pool[k] = traj
        #         theta0 = generated_trajectory['theta0']
        #         yaw_angle = torch.tensor(theta0, device=self.device)
        #         quat_wxyz = torch_utils.quat_from_euler_xyz(
        #             torch.tensor([0.0], device=self.device),
        #             torch.tensor([0.0], device=self.device),
        #             yaw_angle.view(1),
        #         )

        #         self.orientation_pool[k] = quat_wxyz[:, [3,0,1,2]].squeeze(0)

        num_racetrack = self.trajectory_pool_size // 2
        num_generated = self.trajectory_pool_size - num_racetrack

        # ============================================================
        # 50% Racetrack trajectories
        # ============================================================

        for k in range(num_racetrack):

            centerline_m = shift_path_randomly(
                self.centerline_m_orig.copy()
            )

            if np.random.rand() < 0.5:
                centerline_m = centerline_m[::-1]

            _, opt_vel, _, opt_cmds, theta0 = compute_optimal_line_mecanum(
                2.5,
                centerline_m,
                num_points=self.track_length
            )

            traj = torch.from_numpy(
                opt_cmds[:self.max_episode_length]
            ).float().to(self.device)

            self.trajectory_pool[k] = traj

            if not self.omni:
                vx = opt_vel[0, 0]
                vy = opt_vel[0, 1]

                heading_vec = torch.tensor(
                    [vx, vy],
                    device=self.device,
                    dtype=torch.float32
                )

                heading_vec = heading_vec / torch.norm(heading_vec)

                yaw_angle = torch.atan2(
                    heading_vec[1],
                    heading_vec[0]
                )

            else:
                yaw_angle = torch.tensor(
                    theta0,
                    device=self.device,
                    dtype=torch.float32
                )

            quat_wxyz = torch_utils.quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                yaw_angle.view(1),
            )

            self.orientation_pool[k] = (
                quat_wxyz[:, [3, 0, 1, 2]]
                .squeeze(0)
            )


        # ============================================================
        # 50% Generated edge-case/open trajectories
        # ============================================================
        training_trajectory_types = [
            "forward_then_backward",
            # "curve",
            # "circle_like",
            # "forward_fixed_orientation",
            "spin_while_world_x",
            "spin_in_place",
            # "zigzag",
            "forward_then_stop",
            # "lateral_motion",
            # "diagonal_motion",
            # "s_curve",
            "mixed_segments",
        ]

        generated_trajectories = generate_dataset(
            num_generated,
            sample=False,
            save_plots=False,
            enabled_trajectories=training_trajectory_types
        )

        for i, generated_trajectory in enumerate(
            generated_trajectories
        ):
            # Continue filling the pool after racetrack trajectories
            k = num_racetrack + i

            linear_velocities = generated_trajectory["vel"]
            angular_velocities = generated_trajectory["omega"]

            traj_np = np.column_stack((
                linear_velocities,
                angular_velocities
            ))

            traj = torch.from_numpy(
                traj_np[:self.max_episode_length]
            ).float().to(self.device)

            self.trajectory_pool[k] = traj

            theta0 = generated_trajectory["theta0"]

            yaw_angle = torch.tensor(
                theta0,
                device=self.device,
                dtype=torch.float32
            )

            quat_wxyz = torch_utils.quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                yaw_angle.view(1),
            )

            self.orientation_pool[k] = (
                quat_wxyz[:, [3, 0, 1, 2]]
                .squeeze(0)
            )
        
        self.decimation = cfg.decimation
        if self.visualization:
            self.current_env_idx = 0

        if self.curriculum:
            self.rewards = torch.zeros(self.max_episode_length)

        if self.visualization:
            self.setup_live_plot()

        self.start_k = 0.5
        self.end_k = 0.005
        self.curriculum_steps = 35000

        progress = min(float(self.common_step_counter) / self.curriculum_steps, 1.0)
        self.k = self.start_k * ((self.end_k / self.start_k) ** progress)

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
        projected_gravity = self._robot.data.projected_gravity_b

        observations = self.compute_observations(
            base_linear_velocity, base_angular_velocity, projected_gravity
        )
        return {"policy": observations}


    def compute_observations(self, base_linear_velocity, base_angular_velocity, projected_gravity):
        lin_vel_xy = base_linear_velocity[..., :2]
        ang_vel_z = base_angular_velocity[..., 2:3]

        base_velocity = torch.cat((lin_vel_xy, ang_vel_z), dim=1)
        current_cmd = self.command_velocities[..., :3] * self.commands_scale
        tracking_error = current_cmd - base_velocity
        base_lin_acc = (base_linear_velocity - self.prev_base_lin_vel)[..., :2]
        base_ang_acc_z = (base_angular_velocity - self.prev_base_ang_vel)[..., 2:3]

        self.obs_buffer[..., 0:3] = base_velocity
        self.obs_buffer[..., 3:5] = base_lin_acc
        self.obs_buffer[..., 5:6] = base_ang_acc_z
        self.obs_buffer[..., 6:9] = projected_gravity
        self.obs_buffer[..., 9:12] = current_cmd
        self.obs_buffer[..., 12:15] = tracking_error
        self.obs_buffer[..., 15:] = self._actions

        self.prev_base_lin_vel.copy_(base_linear_velocity)
        self.prev_base_ang_vel.copy_(base_angular_velocity)

        return self.obs_buffer

    def plot_reference(self):
        ref = self.reference_positions
        T, num_envs, _ = ref.shape

        arrow_scale = 0.15

        for env_id in range(num_envs):
            x = ref[:, env_id, 0]
            y = ref[:, env_id, 1]
            theta = ref[:, env_id, 2]

            plt.figure()
            # plt.plot(x, y, "-o", linewidth=1.5, label=f"Env {env_id} trajectory")

            # draw heading arrows from pose[:, 2]
            for i in range(0, T):
                dx = np.cos(theta[i]) * arrow_scale
                dy = np.sin(theta[i]) * arrow_scale
                plt.arrow(
                    x[i], y[i],
                    dx, dy,
                    head_width=0.03,
                    length_includes_head=True,
                    alpha=0.7
                )

            plt.axis("equal")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(
                f"Environment {env_id}\n"
                "Reference trajectory + heading (pose[:, 2])"
            )
            plt.grid(True)
            plt.legend()

        # plt.show()
        pass

    def setup_live_plot(self):
        mode = 'omni' if self.drive_type[self.current_env_idx] == 0 else 'forward'
        plt.ion()  # interactive mode on
        self.fig, self.ax = plt.subplots()
        self.ref_line, = self.ax.plot([], [], 'g--', label="Reference Trajectory")
        self.actual_line, = self.ax.plot([], [], 'b-', label="Actual Trajectory")
        self.start_dot, = self.ax.plot([], [], 'bo', label="Start")

        self.heading_arrow = self.ax.arrow(-1, 0, 0, 0, 
                                   head_width=0.1,
                                   head_length=0.2,
                                   fc='r', ec='r')


        self.ax.set_title(f"Trajectory Comparison (Env {self.current_env_idx} | Mode: {mode})")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.legend()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def update_live_plot(self, env_idx=0):
        idx = self.actual_pos_idx[env_idx].item()
        actual = self.actual_positions[env_idx, :idx+1].detach().cpu().numpy()

        ref = self.reference_positions[:, env_idx, :2].detach().cpu().numpy()  # shape (T, 2)

        self.ref_line.set_data(ref[:, 0], ref[:, 1])

        # Remove old arrow
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


        if actual.shape[0] > 0:
            self.actual_line.set_data(actual[:, 0], actual[:, 1])
            self.start_dot.set_data([actual[0, 0]], [actual[0, 1]])

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def compute_reference_positions(self, mask, initial_pos=None):
        """
        Recomputes reference positions from velocity commands.

        Initial x,y come from initial_pos if provided, otherwise zeros.
        Initial heading comes from self.init_orientations_xyzw.
        Uses mask only if self.omni is False.

        Output:
            self.reference_positions: (T+1, num_envs, 3)
        """
        device = self.trajectories.device
        dtype = self.trajectories.dtype

        num_envs = self.num_envs
        traj = self.trajectories  # expected shape: (num_envs, T, 3)
        T = traj.shape[1]

        # Initial XY
        if initial_pos is None:
            initial_xy = torch.zeros((num_envs, 2), device=device, dtype=dtype)
        else:
            initial_xy = initial_pos[:, :2].to(device=device, dtype=dtype)

        # Initial yaw from stored quaternions
        _, _, yaw = torch_utils.get_euler_xyz(
            self.init_orientations_xyzw[:, [1, 2, 3, 0]]
        )
        init_yaw = yaw.to(device=device, dtype=dtype)  # (num_envs,)

        # Scales
        sx = torch.as_tensor(self.commands_scale[0], device=device, dtype=dtype)
        sy = torch.as_tensor(self.commands_scale[1], device=device, dtype=dtype)
        sw = torch.as_tensor(self.commands_scale[2], device=device, dtype=dtype)
        dt = self.step_dt

        # Split commands: (num_envs, T)
        vx = traj[:, :, 0]
        vy = traj[:, :, 1]
        omega = traj[:, :, 2]

        # Angular increments
        dtheta = omega * dt * sw  # (num_envs, T)

        # Heading BEFORE each step:
        # theta_before[t] = init_yaw + sum(dtheta[:t])
        theta_before = init_yaw[:, None] + torch.cumsum(dtheta, dim=1) - dtheta

        cos_theta = torch.cos(theta_before)
        sin_theta = torch.sin(theta_before)

        # World-frame position increments
        dx = (cos_theta * vx - sin_theta * vy) * dt * sx
        dy = (sin_theta * vx + cos_theta * vy) * dt * sy

        # Integrate positions
        x = initial_xy[:, 0:1] + torch.cumsum(dx, dim=1)   # (num_envs, T)
        y = initial_xy[:, 1:2] + torch.cumsum(dy, dim=1)   # (num_envs, T)

        # Heading AFTER each step
        theta = init_yaw[:, None] + torch.cumsum(dtheta, dim=1)  # (num_envs, T)

        # Build output directly: (T+1, num_envs, 3)
        ref = torch.empty((T + 1, num_envs, 3), device=device, dtype=dtype)

        # Initial pose
        ref[0, :, 0:2] = initial_xy
        ref[0, :, 2] = init_yaw

        # Remaining poses
        ref[1:, :, 0] = x.transpose(0, 1)
        ref[1:, :, 1] = y.transpose(0, 1)
        ref[1:, :, 2] = theta.transpose(0, 1)

        if not self.omni:
            dx_step = ref[1:, :, 0] - ref[:-1, :, 0]
            dy_step = ref[1:, :, 1] - ref[:-1, :, 1]

            heading = torch.atan2(dy_step, dx_step)
            motion_mask = (dx_step * dx_step + dy_step * dy_step) > 1e-10

            fw_heading = torch.where(
                motion_mask,
                heading,
                ref[:-1, :, 2],
            )

            mask_t = mask.to(device=device, dtype=torch.bool)

            ref[1:, mask_t, 2] = fw_heading[:, mask_t]
            ref[0, mask_t, 2] = fw_heading[0, mask_t]

        self.reference_positions = ref


    def _pre_physics_step(self, actions):
        root_pos = self._robot.data.root_pos_w  # shape: (num_envs, 3)
        root_quat = self._robot.data.root_quat_w[..., [3,0,1,2]]

        # self._previous_actions = self._actions.clone()
        # self._actions = actions.clone().clamp(-1.0, 1.0)
        self.compute_pre_physics(actions, root_pos, root_quat)
            
        if self.visualization:
            self.update_live_plot(env_idx=self.current_env_idx)

    def compute_pre_physics(self, actions, root_pos, root_quat):
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions).clamp_(-1.0, 1.0)

        
        # self.command_velocities = self.trajectories[:, self.trajectory_idx].squeeze(1)
        self.command_velocities[..., :3] = self.trajectories[:, self.trajectory_idx].squeeze(1)
        # self.transform_velocities(root_quat)
        self.transform_velocities()

        if self.visualization:
            self.actual_positions[
                self.env_ids,
                self.actual_pos_idx
            ] = root_pos[..., :2]
        

    def _apply_action(self):
        # Convert actions into wheel velocities (this needs adjustment for Mercanum wheels)
        self.prev_clipped_velocities = self.clipped_velocities
        self.clipped_velocities = self.get_clipped_velocities()
        # print(clipped_velocities)

        self.rm_dof_targets[..., self.dof_wheel_indices] = self.clipped_velocities


        self._robot.set_joint_velocity_target(self.rm_dof_targets[..., self.dof_wheel_indices], joint_ids=self.dof_wheel_indices)

    def get_clipped_velocities(self):
        cmd = (self.command_velocities * self.commands_scale)
        if self.use_ik:
            wheel_actions = torch_utils.get_wheel_velocities(cmd.T, self.wheel_velocities_matrix).clamp(-self.max_velocities, self.max_velocities)
        else:
            if self.offset:
                offset = self._actions *  (self.max_velocities * self.action_scale)
                wheel_actions = torch_utils.get_wheel_velocities(cmd.T, self.wheel_velocities_matrix).clamp(-self.max_velocities, self.max_velocities) + offset
            else:
                wheel_actions = self._actions * self.max_velocities
        
        # print(f'Should: {torch_utils.get_wheel_velocities(torch.transpose(self.command_velocities[:, :3] * self.commands_scale, 0, -1), self.wheel_velocities_matrix).clamp(-self.max_velocities, self.max_velocities) }')
        # print(f'Is: {wheel_actions}')
        self.rm_dof_targets[...] = 0.0
        accelerations = wheel_actions - self.rm_dof_vel[..., self.dof_wheel_indices]
        accelerations = accelerations.clamp(
            -self.max_acceleration,
            self.max_acceleration
        )

        clipped_velocities = (self.rm_dof_vel[:, self.dof_wheel_indices] + accelerations).clamp(-self.max_velocities, self.max_velocities)
        
        return clipped_velocities


    def transform_velocities(self):
        root_quat = self._robot.data.root_quat_w[..., [3,0,1,2]] 
        dt = self.step_dt

        _, _, root_rot = torch_utils.get_euler_xyz(root_quat)

        if self.trajectory_idx >= self.reference_positions.shape[0] - 1:
            self.command_velocities.zero_()
            return

        pos_now = self.reference_positions[self.trajectory_idx].squeeze(0)
        pos_next = self.reference_positions[self.trajectory_idx + 1].squeeze(0)

        dx = pos_next[...,0] - pos_now[...,0]
        dy = pos_next[...,1] - pos_now[...,1]

        distance = torch.sqrt(dx*dx + dy*dy)
        target_theta = torch.atan2(-dy, dx)

        heading_error = (target_theta - root_rot + np.pi) % (2*np.pi) - np.pi

        forward_speed = distance / (dt * self.commands_scale[0])

        mask = (self.drive_type == 1)

        self.command_velocities[...,0] = torch.where(mask, forward_speed, self.command_velocities[...,0])
        self.command_velocities[...,1] = torch.where(mask, torch.zeros_like(forward_speed), self.command_velocities[...,1])
        self.command_velocities[...,2] = torch.where(mask, heading_error, self.command_velocities[...,2])

    
    def _get_dones(self):
        base_quat = self._robot.data.root_quat_w[..., [3, 0, 1, 2]]

        projected_gravity = self._robot.data.projected_gravity_b

        truncated, reset = self.post_physics_step(base_quat, projected_gravity)

        return truncated, reset
    
    def post_physics_step(self, base_quat, projected_gravity):
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        _, _, base_rot = torch_utils.get_euler_xyz(base_quat)
        self.robot_heading = base_rot    

        self.trajectory_idx = (self.trajectory_idx + 1) % self.trajectories.shape[1]
        self.actual_pos_idx = (self.actual_pos_idx + 1) % self.max_episode_length

        # reset conditions
        reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        err = (projected_gravity * self.gravity_vec_rm_proj).sum(dim=1)

        death_mask = (err + 1.0).abs() > 0.2
        reset |= death_mask

        # curriculum
        if self.curriculum:
            if torch.sum(self.rewards) >= (150 * self.action_scale):
                if self.action_scale < 1.0:
                    print(
                        f"Action Scale: {self.action_scale} --> {min(self.action_scale + 0.1, 1.0)}"
                    )
                    self.action_scale = min(self.action_scale + 0.1, 1.0)

        return truncated, reset


    def _get_rewards(self):

        root_lin_vels = self._robot.data.root_lin_vel_b
        root_ang_vels = self._robot.data.root_ang_vel_b

        root_quat = self._robot.data.root_quat_w[..., [1, 2, 3, 0]]

        err = (self._robot.data.projected_gravity_b * self.gravity_vec_rm_proj).sum(dim=1)

        reward = self.compute_rewards(root_lin_vels, root_ang_vels, root_quat, err)

        return reward
    
    
    def compute_rewards(self, root_lin_vels, root_ang_vels, root_quat, err):

        target_vel = self.command_velocities * self.commands_scale
        
        # -------- Velocity tracking --------
        # # # vel_error_xy = torch.square(target_vel[..., :2] - root_lin_vels[..., :2]).sum(dim=1)
        # # # ang_vel_error = torch.square(target_vel[..., 2] - root_ang_vels[..., 2])

        # vel_error_xy = torch.abs(target_vel[..., :2] - root_lin_vels[..., :2]).sum(dim=1)
        # ang_vel_error = torch.abs(target_vel[..., 2] - root_ang_vels[..., 2])

        # rew_lin_vel_xy = torch.exp(-vel_error_xy / 0.25) * self.rew_scales["lin_vel_xy"]
        # rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        
        # progress = min(float(self.common_step_counter) / self.curriculum_steps, 1.0)
        # self.k = self.start_k * ((self.end_k / self.start_k) ** progress)

        # rew_lin_vel_xy = torch.exp(-vel_error_xy / self.k) * self.rew_scales["lin_vel_xy"]
        # rew_ang_vel_z = torch.exp(-ang_vel_error / self.k) * self.rew_scales["ang_vel_z"]


        # # # rew_lin_vel_xy = torch.exp(-vel_error_xy / 5.5) * self.rew_scales["lin_vel_xy"]
        # # # rew_ang_vel_z = torch.exp(-ang_vel_error / 5.5) * self.rew_scales["ang_vel_z"]

        # # # vel_reward = rew_lin_vel_xy + rew_ang_vel_z
        # # # # vel_reward = rew_ang_vel_z

        # # # vel_reward = (
        # # #     self.rew_scales["base_vel"] * vel_reward
        # # # )

        err_x = torch.abs(target_vel[..., 0] - root_lin_vels[..., 0])
        err_y = torch.abs(target_vel[..., 1] - root_lin_vels[..., 1])
        err_z = torch.abs(target_vel[..., 2] - root_ang_vels[..., 2])

        # rew_x = torch.exp(-err_x / 3.5) * self.rew_scales["ang_vel_z"] * 1.0
        # rew_y = torch.exp(-err_y / 3.5) * self.rew_scales["ang_vel_z"] * 1.0
        # rew_z = torch.exp(-err_z / 0.5) * self.rew_scales["ang_vel_z"] * 1.0
        if self.offset:
            k = 1.0
            l = 0.75
        else: 
            k = 1.5
            l = 1.5
        rew_x = torch.exp(-err_x / k) * self.rew_scales["ang_vel_z"] * 1.0
        rew_y = torch.exp(-err_y / k) * self.rew_scales["ang_vel_z"] * 1.0
        rew_z = torch.exp(-err_z / l) * self.rew_scales["ang_vel_z"] * 1.0


        bal_err = torch.abs(err_x - err_y)
        rew_bal = 0.0 * torch.exp(-bal_err / 0.5) * self.rew_scales["ang_vel_z"]

        reward = rew_x + rew_y + rew_z + rew_bal

        # -------- Position tracking --------
        ####################
        # ref_pos = self.reference_positions[self.trajectory_idx].squeeze(0)
        # ref_pos = self.reference_positions[self.trajectory_idx]
        # step_ids = self.trajectory_idx.unsqueeze(-1).unsqueeze(-1)
        # ref_pos = torch.gather(
        #     self.reference_positions,
        #     0,
        #     step_ids.expand(-1, 1, 3)
        # ).squeeze(1)

        # actual_pos = self.actual_positions[self.env_ids, self.actual_pos_idx - 1]

        # pos_error_xy = torch.square(ref_pos[:, :2] - actual_pos).sum(dim=1)

        
        # _, _, actual_head = torch_utils.get_euler_xyz(root_quat)

        # ang_pos_error = torch.square(
        #     ref_pos[..., 2] - torch_utils.wrap_to_pi(actual_head)
        # )

        # rew_pos_xy = torch.exp(-pos_error_xy / 5.) * self.rew_scales["lin_pos_xy"]
        # rew_head_z = torch.exp(-ang_pos_error / 1.) * self.rew_scales["ang_head_z"]

        # pos_reward = rew_pos_xy + rew_head_z

        # reward = vel_reward + pos_reward * self.rew_scales["position"]

        # -------- Action penalties --------
        # wheel_energy = torch.square(self._actions).sum(dim=1)

        # delta_actions = self._actions - self._previous_actions
        # smoothness = torch.square(delta_actions).sum(dim=1)

        # # symmetry (vectorized)
        # symmetry = torch.square(self._actions[:, [0, 1]] - self._actions[:, [2, 3]]).sum(dim=1)

        # # # reward = (
        # # #     vel_reward
        # # #     # - self.rew_scales["wheel_energy"] * wheel_energy
        # # #     # - self.rew_scales["action_smooth"] * smoothness
        # # #     # - self.rew_scales["wheel_sym"] * symmetry
        # # #     # + self.rew_scales["orient"] * rew_head_z
        # # # )

        # -------- Termination penalty --------
        death_mask = (err + 1.0).abs() > 0.2

        reward = torch.where(death_mask, reward-1.0, reward)

        # -------- Curriculum --------
        if self.curriculum:
            reward = reward * self.action_scale
            step = self.common_step_counter % self.max_episode_length
            self.rewards[step - 1] = reward.mean()

        return reward

    def _reset_idx(self, env_ids):
        """Reset specific environments when necessary."""
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_fwd = self.num_envs - int(self.num_envs * self.omni_ratio)

        # drive_type should be a torch tensor on self.device
        self.drive_type.zero_()
        if not self.omni:
            fwd_indices = torch.randperm(self.num_envs, device=self.device)[:num_fwd]
            self.drive_type[fwd_indices] = 1
        
        mask = self.drive_type == 1

        self.generate_trajectories(env_ids)
        self.compute_reference_positions(mask)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # self.prev_base_lin_vel = self._robot.data.root_lin_vel_b.clone()
        # self.prev_base_ang_vel = self._robot.data.root_ang_vel_b.clone()
        self.prev_base_lin_vel.copy_(self._robot.data.root_lin_vel_b)
        self.prev_base_ang_vel.copy_(self._robot.data.root_ang_vel_b)


        positions = self.initial_base_position.clone()
        orientations = self.initial_base_orientation.clone()


        if not self.omni:
            fwd_env_ids = env_ids[mask[env_ids]]
            orientations[fwd_env_ids] = self.init_orientations_xyzw[fwd_env_ids]
        else:
            orientations[env_ids] = self.init_orientations_xyzw[env_ids]

        positions[env_ids, :2] = 0.0

        _, _, base_rot = torch_utils.get_euler_xyz(orientations)
        self.robot_heading = base_rot

        pose = torch.cat((positions, orientations), dim=1)
        velocities = torch.cat(
            (
                torch.zeros_like(self._robot.data.root_lin_vel_b),
                torch.zeros_like(self._robot.data.root_ang_vel_b),
            ),
            dim=1,
        )

        self.rm_dof_pos[env_ids] = self.initial_dof_pos[env_ids]
        self.rm_dof_vel[env_ids] = 0

        self._robot.set_joint_position_target(self.rm_dof_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(
            self.rm_dof_pos[env_ids], self.rm_dof_vel[env_ids], env_ids=env_ids
        )
        self._robot.write_root_pose_to_sim(pose[env_ids], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(velocities[env_ids], env_ids=env_ids)

        self.actual_positions[env_ids] = 0
        self.actual_pos_idx[env_ids] = 0


    def generate_trajectories(self, env_ids):

        ids = torch.randint(
            0,
            self.trajectory_pool_size,
            (len(env_ids),),
            device=self.device
        )

        self.trajectories[env_ids] = self.trajectory_pool[ids]
        self.init_orientations_xyzw[env_ids] = self.orientation_pool[ids]
     

    def on_key_press(self, event):
        if event.key == 'd':
            self.current_env_idx = (self.current_env_idx + 1) % self.num_envs
        elif event.key == 'a':
            self.current_env_idx = (self.current_env_idx - 1) % self.num_envs

        mode = 'omni' if self.drive_type[self.current_env_idx] == 0 else 'forward'
        self.ax.set_title(f"Trajectory Comparison (Env {self.current_env_idx} | Mode: {mode})")#
        self.update_live_plot(env_idx=self.current_env_idx)


def save_as_csv(tensor, dir, name, abort=False, save_fig=False):
    try:
        array = tensor.cpu().numpy()
    except:
        array = tensor
    os.makedirs(dir, exist_ok=True)
    np.savetxt(f'{dir}/{name}.csv', array, delimiter=',')
    if abort:
        print('Saved ... ')
        raise not NotImplementedError('')
    else:
        if save_fig:
            plt.savefig(f'{dir}/{name}.png')

        