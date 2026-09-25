import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from tqdm import tqdm
import random
import os

try:
    from .racetracks import create_racetrack, get_available_tracks
except ImportError:
    from racetracks import create_racetrack, get_available_tracks


def create_original_racetrack(map_size, cell_size, boundary_size):
        start_map = -(map_size / 2.0) 
        end_map = map_size / 2.0 + cell_size + (boundary_size * cell_size)
        x_pos = np.arange(start_map, end_map, cell_size)
        y_pos = np.arange(start_map, end_map, cell_size)


        map_y, map_x = np.meshgrid(y_pos, x_pos)  # Remove indexing argument
        map_y = map_y.transpose(0, 1)  # Swap axes for 'ij' indexing
        map_x = map_x.transpose(0, 1)
        map_xy = np.stack((map_x, map_y), axis=-1)

        xy_vels_grid = np.zeros_like(map_xy)
        orientation_map = np.zeros_like(xy_vels_grid) 

        velocity = 1.0
        xy_vels_grid[:,:,1] = 0.0
        xy_vels_grid[:,:,0] = 0.0

        # Straight
        xy_vels_grid[1:4, 4:18, 0] = -velocity
        xy_vels_grid[1:5, 1:4, 1] = velocity
        xy_vels_grid[5:8, 1:10, 0] = velocity
        xy_vels_grid[5:9, 10:13, 1] = velocity
        xy_vels_grid[9:12, 4:13, 0] = -velocity
        xy_vels_grid[9:15, 1:4, 1] = velocity
        xy_vels_grid[15:18, 1:15, 0] = velocity
        xy_vels_grid[4:18, 15:18, 1] = -velocity
        

        # orientation
        orientation_map[1:4, 2:17, 0] = -velocity
        orientation_map[2:7, 1:4, 1] = velocity
        orientation_map[5:8, 2:12, 0] = velocity
        orientation_map[6:11, 10:13, 1] = velocity
        orientation_map[9:12, 2:12, 0] = -velocity
        orientation_map[10:17, 1:4, 1] = velocity
        orientation_map[15:18, 2:17, 0] = velocity
        orientation_map[2:17, 15:18, 1] = -velocity

        orientation_map[1:3, 2:5, 1] = velocity
        orientation_map[4:7, 1:3, 0] = velocity
        orientation_map[5:7, 9:12, 1] = velocity
        orientation_map[8:11, 11:13, 0] = -velocity
        orientation_map[8:11, 11:13, 0] = -velocity
        orientation_map[9:11, 2:5, 1] = velocity
        orientation_map[14:17, 1:3, ] = velocity
        orientation_map[16:18, 14:17, 1] = -velocity
        orientation_map[2:5, 16:18, 0] = -velocity

        orientation_map[15:18, 4:14, 0] = 0.
        orientation_map[15:18, 4:14, 1] = -velocity

        return map_x, map_y, xy_vels_grid, orientation_map

def fit_spline_and_extract_velocities(waypoints, num_points=3500, dt=0.02, v_max=1., omega_max=1.0, return_global_vel=False):
    """
    Fits a spline and returns velocity commands where the robot always drives forward (vx > 0 in local frame).
    
    Parameters:
    - waypoints: (N, 2) array of [x, y] points
    - num_points: number of samples along the spline
    - dt: timestep between points
    - v_max: max forward speed
    - omega_max: max angular velocity
    - return_global_vel: if True, returns [vx, vy] in global frame; else in robot local frame (vx > 0, vy = 0)

    Returns:
    - positions: (num_points, 2)
    - velocities: (num_points, 2)
    - omega: (num_points,)
    - commands: (num_points, 3) as [vx, vy, omega]
    """
    waypoints = np.array(waypoints)
    tck, _ = splprep([waypoints[:, 0], waypoints[:, 1]], s=0)
    u = np.linspace(0, 1, num_points)

    # Sample positions
    x, y = splev(u, tck)
    positions = np.stack([x, y], axis=1)

    # Compute heading from spline derivative
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    theta = np.arctan2(dy, dx)

    # Forward speed (magnitude of motion)
    v = np.sqrt(dx**2 + dy**2)
    v = np.clip(v, 0, v_max)

    # Set local velocity (only forward, no lateral)
    vx_local = v
    vy_local = np.zeros_like(v)

    # Optional: convert local [vx, vy] to global [vx, vy]
    if return_global_vel:
        vx_global = vx_local * np.cos(theta) - vy_local * np.sin(theta)
        vy_global = vx_local * np.sin(theta) + vy_local * np.cos(theta)
        velocities = np.stack([vx_global, vy_global], axis=1)
    else:
        velocities = np.stack([vx_local, vy_local], axis=1)

    # Angular velocity
    dtheta = np.gradient(theta, dt)
    omega = np.clip(dtheta, -omega_max, omega_max)

    # Combine
    commands = np.concatenate([velocities, omega[:, None]], axis=1)
    return positions, velocities, omega, commands

def compute_optimal_line(centerline, offset_strength=0.05, num_points=3500):
    """
    Compute a racing line offset from the centerline based on curvature.
    
    Args:
        centerline: (N, 2) points from ordered skeleton centerline
        offset_strength: multiplier for how aggressive the racing line cuts corners
        num_points: number of points to sample along the spline

    Returns:
        optimal_positions: (num_points, 2) array of [x, y] optimal path
    """
    # Fit spline to centerline
    tck, _ = splprep([centerline[:, 1], centerline[:, 0]], s=5.0)
    u = np.linspace(0, 1, num_points)
    
    # Spline evaluation
    y, x = splev(u, tck)
    dy, dx = splev(u, tck, der=1)
    ddy, ddx = splev(u, tck, der=2)

    # Curvature
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature = np.nan_to_num(curvature)

    # Normals (perpendicular to tangent)
    norm = np.stack([-dy, dx], axis=1)
    norm /= np.linalg.norm(norm, axis=1, keepdims=True)

    # Offset line
    current_offset_strength = np.random.uniform(-offset_strength, offset_strength)
    offset = -current_offset_strength * curvature[:, None] * norm
    optimal_positions = np.stack([x, y], axis=1) + offset

    # Velocities
    vel = np.gradient(optimal_positions, axis=0)
    
    # Heading and omega
    theta = np.arctan2(vel[:, 1], vel[:, 0])
    omega = np.gradient(theta)
    
    vel_mag = np.linalg.norm(vel, axis=1)
    max_v = np.max(vel_mag)
    max_omega = np.max(np.abs(omega)) + 1e-6
    vel /= max_v
    omega /= max_omega

    commands = np.concatenate([vel, omega[:, None]], axis=1)

    return optimal_positions, vel, omega, commands, 0

def compute_optimal_line_mecanum(
    max_speed,
    centerline,
    offset_strength=0.05,
    num_points=3500,
    dt=0.02,
    heading_update_interval=20,
    heading_offset_max=np.pi / 3,
    kp_heading=3.0,
    omega_limit=1.0,
    max_target_step=np.pi / 4,
):
    """
    Generate a racetrack-following mecanum trajectory by:
    - computing the spline racing line as before
    - keeping world velocity tangent to the path
    - assigning an independent smooth random target heading every N steps
    - smoothly steering robot heading toward that target
    - converting world velocity into robot-frame vx, vy
    """

    def wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # same path generation as compute_optimal_line
    tck, _ = splprep([centerline[:, 1], centerline[:, 0]], s=5.0)
    u = np.linspace(0, 1, num_points)

    y, x = splev(u, tck)
    dy, dx = splev(u, tck, der=1)
    ddy, ddx = splev(u, tck, der=2)
    # x, y = splev(u, tck)
    # dx, dy = splev(u, tck, der=1)
    # ddx, ddy = splev(u, tck, der=2)

    optimal_positions = np.stack([x, y], axis=1)

    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8) ** 1.5
    curvature = np.nan_to_num(curvature)

    norm = np.stack([-dy, dx], axis=1)
    norm /= (np.linalg.norm(norm, axis=1, keepdims=True) + 1e-8)

    current_offset_strength = np.random.uniform(-offset_strength, offset_strength)
    offset = -current_offset_strength * curvature[:, None] * norm
    optimal_positions = np.stack([x, y], axis=1) + offset

    # world velocity tangent to path
    # vx_world = np.gradient(optimal_positions[:, 0], dt)
    # vy_world = np.gradient(optimal_positions[:, 1], dt)
    # vel_world = np.stack([vx_world, vy_world], axis=1)

    # speed_world = np.linalg.norm(vel_world, axis=1)
    # max_world_speed = np.max(speed_world) + 1e-8
    # vel_world = vel_world / max_world_speed * max_speed

    # vx_world = vel_world[:, 0]
    # vy_world = vel_world[:, 1]

    vel_world = np.zeros_like(optimal_positions)

    vel_world[:-1] = optimal_positions[1:] - optimal_positions[:-1]
    vel_world[-1] = optimal_positions[0] - optimal_positions[-1]

    vel_world /= np.linalg.norm(vel_world, axis=1, keepdims=True) + 1e-8
    vel_world *= max_speed

    vx_world = vel_world[:, 0]
    vy_world = vel_world[:, 1]

    # ---- heading dynamics ----
    N = num_points
    theta = np.zeros(N)
    omega = np.zeros(N)

    theta[0] = np.random.uniform(-np.pi, np.pi)
    target_heading = theta[0]

    for i in range(N - 1):
        if i % heading_update_interval == 0:
            # choose next target relative to CURRENT target, not initial heading
            target_heading = wrap_to_pi(
                target_heading + np.random.uniform(-max_target_step, max_target_step)
            )
            # keep target inside a bounded range around zero if desired
            # target_heading = np.clip(target_heading, -heading_offset_max, heading_offset_max)

        heading_error = wrap_to_pi(target_heading - theta[i])

        omega_cmd = kp_heading * heading_error
        omega_cmd = np.clip(omega_cmd, -omega_limit, omega_limit)

        omega[i] = omega_cmd
        theta[i + 1] = wrap_to_pi(theta[i] + omega[i] * dt)

    omega[-1] = omega[-2] if N > 1 else 0.0

    # optional smoothing of omega to remove small jerks
    kernel_size = 9
    kernel = np.ones(kernel_size) / kernel_size
    omega = np.convolve(omega, kernel, mode="same")

    # re-integrate theta from smoothed omega
    theta[0] = theta[0]
    for i in range(N - 1):
        theta[i + 1] = wrap_to_pi(theta[i] + omega[i] * dt)

    # convert world velocity to robot frame using current heading 
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vx_robot = cos_t * vx_world + sin_t * vy_world
    vy_robot = -sin_t * vx_world + cos_t * vy_world
    vel_robot = np.stack([vx_robot, vy_robot], axis=1)

    # scaling
    max_vx = np.max(np.abs(vel_robot[:, 0])) + 1e-8
    max_vy = np.max(np.abs(vel_robot[:, 1])) + 1e-8
    max_omega = np.max(np.abs(omega)) + 1e-8

    scale_vx = 1.0 / max_vx
    scale_vy = 1.0 / max_vy
    scale_w = omega_limit / max_omega

    scale = min(scale_vx, scale_vy, scale_w, 1.0)

    vel_robot = vel_robot * scale
    omega = omega * scale

    commands = np.stack([vel_robot[:, 0], vel_robot[:, 1], omega], axis=1)

    # print("max |vx|:", np.max(np.abs(vel_robot[:, 0])))
    # print("max |vy|:", np.max(np.abs(vel_robot[:, 1])))
    # print("max |vx| + |vy|:", np.max(np.abs(vel_robot[:, 0]) + np.abs(vel_robot[:, 1])))

    return optimal_positions, commands[:, :2], commands[:, 2], commands, theta[0]

def get_track_mask(vel_map, threshold=0.1):
    speed = np.linalg.norm(vel_map, axis=-1)
    return (speed > threshold).astype(np.uint8)


def extract_centerline_skeleton(track_mask):
    skeleton = skeletonize(track_mask)  # from skimage
    return skeleton.astype(np.uint8)

def ordered_centerline_points(skeleton):
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return np.empty((0, 2))

    # Use regionprops to find connected components
    labeled = label(skeleton)
    regions = regionprops(labeled)

    # Take the largest component (main track)
    main = max(regions, key=lambda r: r.area)
    coords = main.coords

    from scipy.spatial.distance import cdist
    ordered = [coords[0]]
    used = set([0])
    for _ in range(len(coords) - 1):
        dists = cdist([ordered[-1]], coords)
        dists[0][list(used)] = np.inf
        next_idx = np.argmin(dists)
        ordered.append(coords[next_idx])
        used.add(next_idx)
    return np.array(ordered)

def shift_path_randomly(centerline):
    """
    Shifts the centerline so that it starts from a random point along it.

    Args:
        centerline: (N, 2) numpy array of ordered points

    Returns:
        shifted_centerline: (N, 2) array starting at a random index
    """
    n_points = len(centerline)
    start_idx = np.random.randint(0, n_points)
    shifted = np.concatenate([centerline[start_idx:], centerline[:start_idx]], axis=0)
    return shifted

def integrate_orientation(theta_start, omega, dt=0.02):
    """
    Reconstruct robot orientation using velocities and omega.
    Works for both trajectory generators.
    """

    theta = np.zeros(len(omega))

    # Initial heading from first velocity
    theta[0] = theta_start

    for i in range(1, len(omega)):
        theta[i] = theta[i-1] + omega[i-1] * dt

    return theta

def compute_reference_positions_numpy(trajectories, step_dt, commands_scale=(1.0, 1.0, 1.0), initial_pos=None):
    """
    NumPy equivalent of compute_reference_positions for a single trajectory.

    Args:
        trajectories: (T, 3) array with columns [vx, vy, omega]
        step_dt: integration timestep
        commands_scale: tuple/list (sx, sy, somega)
        initial_pos: (3,) array [x0, y0, theta0]

    Returns:
        reference_positions: (T+1, 3) array [x, y, theta]
    """
    trajectories = np.asarray(trajectories)

    if initial_pos is None:
        initial_pos = np.zeros(3, dtype=np.float64)
    else:
        initial_pos = np.asarray(initial_pos, dtype=np.float64)

    vx = trajectories[:, 0]
    vy = trajectories[:, 1]
    omega = trajectories[:, 2]

    # integrate heading
    dtheta = omega * step_dt * commands_scale[2]
    theta = np.cumsum(dtheta)
    theta = np.concatenate([[initial_pos[2]], theta + initial_pos[2]])

    theta_mid = theta[:-1]

    dx = (
        np.cos(theta_mid) * vx
        - np.sin(theta_mid) * vy
    ) * step_dt * commands_scale[0]

    dy = (
        np.sin(theta_mid) * vx
        + np.cos(theta_mid) * vy
    ) * step_dt * commands_scale[1]

    x = np.cumsum(dx)
    y = np.cumsum(dy)

    x = np.concatenate([[initial_pos[0]], x + initial_pos[0]])
    y = np.concatenate([[initial_pos[1]], y + initial_pos[1]])

    reference_positions = np.stack([x, y, theta], axis=-1)
    return reference_positions

def plot_original_racetrack_generation(
    map_size,
    cell_size,
    boundary_size,
    num_trajectory,
    max_speed,
    num_points,
    output_dir="original_racetrack_generation",
):
    """
    Generate the original racetrack and visualize the complete trajectory
    generation pipeline.

    Saved stages:
        01_track_mask.png
        02_skeleton.png
        03_ordered_centerline.png
        04_generated_trajectory.png
        05_trajectory_orientation_velocity.png
        06_reconstructed_from_commands.png
    """

    os.makedirs(output_dir, exist_ok=True)

    map_x, map_y, vel_map, ori_map = create_original_racetrack(
        map_size,
        cell_size,
        boundary_size,
    )

    track_mask = get_track_mask(vel_map)

    plt.figure(figsize=(8, 8))
    plt.imshow(track_mask, cmap="gray", origin="lower")
    plt.title("Original Racetrack - Track Mask")
    plt.xlabel("Grid x")
    plt.ylabel("Grid y")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"01_track_mask_{num_trajectory}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    skeleton = extract_centerline_skeleton(track_mask)

    plt.figure(figsize=(8, 8))
    plt.imshow(track_mask, cmap="gray", origin="lower")

    skeleton_y, skeleton_x = np.where(skeleton > 0)

    plt.scatter(
        skeleton_x,
        skeleton_y,
        s=8,
        label="Skeleton",
    )

    plt.title("Original Racetrack - Extracted Skeleton")
    plt.xlabel("Grid x")
    plt.ylabel("Grid y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"02_skeleton_{num_trajectory}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    centerline_px = ordered_centerline_points(skeleton)

    if len(centerline_px) == 0:
        raise RuntimeError(
            "No centerline could be extracted from the original racetrack."
        )

    plt.figure(figsize=(8, 8))
    plt.imshow(track_mask, cmap="gray", origin="lower")

    plt.plot(
        centerline_px[:, 1],
        centerline_px[:, 0],
        linewidth=2,
        label="Ordered Centerline",
    )

    # Mark the starting point
    plt.scatter(
        centerline_px[0, 1],
        centerline_px[0, 0],
        s=70,
        marker="o",
        label="Start",
        zorder=5,
    )

    plt.title("Original Racetrack - Ordered Centerline")
    plt.xlabel("Grid x")
    plt.ylabel("Grid y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"03_ordered_centerline_{num_trajectory}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    centerline_m = np.stack(
        [
            centerline_px[:, 1] * cell_size,
            centerline_px[:, 0] * cell_size,
        ],
        axis=-1,
    )

    # compute_optimal_line_mecanum contains random heading generation
    random_state = np.random.get_state()
    np.random.seed(num_trajectory)

    opt_pos, opt_vel, opt_omega, opt_cmds, theta0 = (
        compute_optimal_line_mecanum(
            max_speed,
            centerline_m,
            num_points=num_points,
        )
    )

    np.random.set_state(random_state)

    plt.figure(figsize=(8, 8))

    plt.imshow(
        track_mask,
        cmap="gray",
        origin="lower",
    )

    plt.plot(
        centerline_px[:, 1],
        centerline_px[:, 0],
        linestyle="--",
        linewidth=1.5,
        label="Centerline",
    )

    plt.plot(
        opt_pos[:, 0] / cell_size,
        opt_pos[:, 1] / cell_size,
        linewidth=2,
        label="Generated Trajectory",
    )

    plt.title("Original Racetrack - Generated Trajectory")
    plt.xlabel("Grid x")
    plt.ylabel("Grid y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"04_generated_trajectory_{num_trajectory}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Reconstruct reference trajectory from generated commands
    commands_np = np.concatenate(
        [
            opt_vel,
            opt_omega[:, None],
        ],
        axis=1,
    )

    ref_positions = compute_reference_positions_numpy(
        commands_np,
        step_dt=0.02,
        commands_scale=(2.5, 2.5, 2.5),
        initial_pos=np.array(
            [
                opt_pos[0, 0],
                opt_pos[0, 1],
                theta0,
            ]
        ),
    )

    reconstructed_pos = ref_positions[:, :2]
    theta_robot = ref_positions[:, 2]

    # Transform robot-frame velocity into world frame
    vx_world = (
        opt_vel[:, 0] * np.cos(theta_robot[:-1])
        - opt_vel[:, 1] * np.sin(theta_robot[:-1])
    )

    vy_world = (
        opt_vel[:, 0] * np.sin(theta_robot[:-1])
        + opt_vel[:, 1] * np.cos(theta_robot[:-1])
    )


    step = 20

    plt.figure(figsize=(8, 8))

    plt.imshow(
        track_mask,
        cmap="gray",
        origin="lower",
    )

    plt.plot(
        opt_pos[:, 0] / cell_size,
        opt_pos[:, 1] / cell_size,
        linewidth=2,
        label="Generated Trajectory",
    )

    # Robot orientation
    plt.quiver(
        opt_pos[::step, 0] / cell_size,
        opt_pos[::step, 1] / cell_size,
        np.cos(theta_robot[:-1:step]),
        np.sin(theta_robot[:-1:step]),
        color="red",
        scale=25,
        width=0.003,
        label="Robot Orientation",
    )

    # World velocity vector
    plt.quiver(
        opt_pos[::step, 0] / cell_size,
        opt_pos[::step, 1] / cell_size,
        vx_world[::step],
        vy_world[::step],
        color="blue",
        scale=20,
        width=0.003,
        label="Velocity Vector",
    )

    plt.title(
        "Original Racetrack - Robot Orientation and Velocity"
    )
    plt.xlabel("Grid x")
    plt.ylabel("Grid y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"05_trajectory_orientation_velocity_{num_trajectory}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 8))

    plt.plot(
        reconstructed_pos[:, 0] / cell_size,
        reconstructed_pos[:, 1] / cell_size,
        linewidth=2,
        label="Reconstructed Path",
    )

    plt.title(
        "Original Racetrack - Reconstructed from Commands"
    )
    plt.xlabel("Grid x")
    plt.ylabel("Grid y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"06_reconstructed_from_commands_{num_trajectory}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Original racetrack generation figures saved to: "
        f"{output_dir}"
    )

    return {
        "track_mask": track_mask,
        "skeleton": skeleton,
        "centerline_px": centerline_px,
        "centerline_m": centerline_m,
        "opt_pos": opt_pos,
        "opt_vel": opt_vel,
        "opt_omega": opt_omega,
        "opt_cmds": opt_cmds,
        "theta0": theta0,
        "reconstructed_pos": reconstructed_pos,
    }


if __name__ == '__main__':
    num_points = 700
    num_trajectories = 100
    cell_size = 0.4
    map_size = 6
    boundary_size = 3
    max_speed = 2.5
    # output_base_dir = f'trajectories_{max_speed}_train'
    # output_base_dir = f'trajectories_new_256__'
    output_base_dir = f'multiple_trajectories_{num_trajectories}_thesis'

    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # for i in range(num_trajectories):
    #     plot_original_racetrack_generation(
    #         map_size=map_size,
    #         cell_size=cell_size,
    #         boundary_size=boundary_size,
    #         num_trajectory=i,
    #         max_speed=max_speed,
    #         num_points=num_points,
    #         output_dir="original_racetrack_generation",
    #     )

    #### Trajectory Generation for Original track ####
    # map_x, map_y, vel_map, ori_map = create_original_racetrack(map_size, cell_size, boundary_size)
    # track_mask = get_track_mask(vel_map)
    # skeleton = extract_centerline_skeleton(track_mask)
    # centerline_px = ordered_centerline_points(skeleton)

    # # Convert from pixel coords [row, col] to world coords [x, y]
    # centerline_m_orig = np.stack([
    #     centerline_px[:, 1] * cell_size,
    #     centerline_px[:, 0] * cell_size
    # ], axis=-1)

    # # Generate multiple trajectories
    # for i in tqdm(range(num_trajectories)):
    #     # Randomize centerline path
    #     centerline_m = shift_path_randomly(centerline_m_orig.copy())
    #     if np.random.rand() < 0.2:
    #         centerline_m = centerline_m[::-1]

    #     # Compute spline and optimal trajectory
    #     positions, velocities, omega, commands = fit_spline_and_extract_velocities(centerline_m)
    #     # opt_pos, opt_vel, opt_omega, opt_cmds, theta0 = compute_optimal_line(centerline_m)
    #     opt_pos, opt_vel, opt_omega, opt_cmds, theta0 = compute_optimal_line_mecanum(max_speed, centerline_m)

    #     # Create output directory for this trajectory
    #     traj_dir = os.path.join(output_base_dir, f'trajectory_{i:03d}')
    #     os.makedirs(traj_dir, exist_ok=True)

    #     # Save trajectory files
    #     # with open(os.path.join(traj_dir, 'spline.csv'), 'w', newline='') as csvfile:
    #     #     writer = csv.writer(csvfile)
    #     #     writer.writerow(['vx', 'vy', 'omega'])
    #     #     for vxi, vyi, wi in zip(velocities[:, 0], velocities[:, 1], omega):
    #     #         writer.writerow([vxi, vyi, wi])

    #     with open(os.path.join(traj_dir, 'optimal_spline.csv'), 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(['vx', 'vy', 'omega', 'theta'])
    #         for vxi, vyi, wi in zip(opt_vel[:, 0], opt_vel[:, 1], opt_omega):
    #             writer.writerow([vxi, vyi, wi, theta0])

    #     # Plot and save figure
    #     # plt.figure(figsize=(10, 10))
    #     # # plt.plot(positions[:, 0]/cell_size, positions[:, 1]/cell_size, 'r-', label="Center Spline Path")
    #     # plt.plot(opt_pos[:, 0]/cell_size, opt_pos[:, 1]/cell_size, 'y-', label="Optimal Spline Path")
    #     # plt.imshow(track_mask, cmap='gray', origin='lower')
    #     # # plt.plot(centerline_px[:, 1], centerline_px[:, 0], 'g-', label='Original Centerline')
    #     # # Compute robot orientation
    #     # # theta_robot = np.arctan2(opt_vel[:, 1], opt_vel[:, 0])
    #     # theta_robot = integrate_orientation(theta_robot, opt_omega)
    #     # # vel_direction = np.arctan2(opt_vel[:, 1], opt_vel[:, 0])
    #     # vx_world = opt_vel[:, 0] * np.cos(theta_robot) - opt_vel[:, 1] * np.sin(theta_robot)
    #     # vy_world = opt_vel[:, 0] * np.sin(theta_robot) + opt_vel[:, 1] * np.cos(theta_robot)

    #     # # Plot orientation arrows
    #     # step = 20  # reduce arrow density/ dt
    #     # plt.quiver(
    #     #     opt_pos[::step, 0]/cell_size,
    #     #     opt_pos[::step, 1]/cell_size,
    #     #     np.cos(theta_robot[::step]),
    #     #     np.sin(theta_robot[::step]),
    #     #     color='red',
    #     #     scale=25,
    #     #     width=0.003,
    #     #     label="Robot Orientation"
    #     # )
    #     # plt.quiver(
    #     #     opt_pos[::step, 0] / cell_size,
    #     #     opt_pos[::step, 1] / cell_size,
    #     #     # np.cos(vel_direction[::step]),
    #     #     # np.sin(vel_direction[::step]),
    #     #     vx_world[::step],
    #     #     vy_world[::step],
    #     #     color='blue',
    #     #     scale=20,
    #     #     width=0.003,
    #     #     label="Velocity Vector"
    #     # )

    #     # plt.legend()
    #     # plt.title(f"Spline Path and Optimal Path - Trajectory {i:03d}")
    #     # plt.axis('equal')
    #     # plt.grid(True)
    #     # plt.savefig(os.path.join(traj_dir, 'plot.png'))
    #     # plt.close()

    #     plt.figure(figsize=(20, 10))

    #     commands_np = np.concatenate([opt_vel, opt_omega[:, None]], axis=1)

    #     ref_positions = compute_reference_positions_numpy(
    #         commands_np,
    #         step_dt=0.02,
    #         commands_scale=(2.5, 2.5, 2.5),
    #         initial_pos=np.array([opt_pos[0, 0], opt_pos[0, 1], theta0])
    #     )

    #     reconstructed_pos = ref_positions[:, :2]
    #     theta_robot = ref_positions[:, 2]

    #     vx_world = opt_vel[:, 0] * np.cos(theta_robot[:-1]) - opt_vel[:, 1] * np.sin(theta_robot[:-1])
    #     vy_world = opt_vel[:, 0] * np.sin(theta_robot[:-1]) + opt_vel[:, 1] * np.cos(theta_robot[:-1])

    #     step = 20

    #     # Left: original spline
    #     plt.subplot(1, 2, 1)
    #     plt.plot(opt_pos[:, 0] / cell_size, opt_pos[:, 1] / cell_size, 'y-', label="Optimal Spline Path")
    #     plt.imshow(track_mask, cmap='gray', origin='lower')

    #     plt.quiver(
    #         opt_pos[::step, 0] / cell_size,
    #         opt_pos[::step, 1] / cell_size,
    #         np.cos(theta_robot[:-1:step]),
    #         np.sin(theta_robot[:-1:step]),
    #         color='red',
    #         scale=25,
    #         width=0.003,
    #         label="Robot Orientation"
    #     )

    #     plt.quiver(
    #         opt_pos[::step, 0] / cell_size,
    #         opt_pos[::step, 1] / cell_size,
    #         vx_world[::step],
    #         vy_world[::step],
    #         color='blue',
    #         scale=20,
    #         width=0.003,
    #         label="Velocity Vector"
    #     )

    #     plt.legend()
    #     plt.title(f"Original Spline Path - Trajectory {i:03d}")
    #     plt.axis('equal')
    #     plt.grid(True)

    #     # Right: reconstructed from commands
    #     plt.subplot(1, 2, 2)
    #     plt.plot(reconstructed_pos[:, 0] / cell_size, reconstructed_pos[:, 1] / cell_size, 'c-', label="Reconstructed Path")
    #     # plt.imshow(track_mask, cmap='gray', origin='lower')

    #     # plt.quiver(
    #     #     reconstructed_pos[:-1:step, 0] / cell_size,
    #     #     reconstructed_pos[:-1:step, 1] / cell_size,
    #     #     np.cos(theta_robot[:-1:step]),
    #     #     np.sin(theta_robot[:-1:step]),
    #     #     color='red',
    #     #     scale=25,
    #     #     width=0.003,
    #     #     label="Robot Orientation"
    #     # )

    #     # plt.quiver(
    #     #     reconstructed_pos[:-1:step, 0] / cell_size,
    #     #     reconstructed_pos[:-1:step, 1] / cell_size,
    #     #     vx_world[::step],
    #     #     vy_world[::step],
    #     #     color='blue',
    #     #     scale=20,
    #     #     width=0.003,
    #     #     label="Velocity Vector"
    #     # )

    #     plt.legend()
    #     plt.title(f"Reconstructed from Commands - Trajectory {i:03d}")
    #     plt.axis('equal')
    #     plt.grid(True)

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(traj_dir, 'plot.png'))
    #     plt.close()

    # # centerline_m = shift_path_randomly(centerline_m)
    # # if np.random.rand() < 0.5:
    # #     centerline_m = centerline_m[::-1]

    # # positions, velocities, omega, commands = fit_spline_and_extract_velocities(centerline_m)
    # # opt_pos, opt_vel, opt_omega, opt_cmds = compute_optimal_line(centerline_m)

    # # with open(f'trajectories/race_track_spline.csv', 'w', newline='') as csvfile:
    # #         writer = csv.writer(csvfile)
    # #         writer.writerow(['vx', 'vy', 'omega'])
    # #         for vxi, vyi, wi in zip(velocities[:, 0], velocities[:, 1], omega):
    # #             writer.writerow([vxi, vyi, wi])

    # # with open(f'trajectories/race_track_spline_optimal.csv', 'w', newline='') as csvfile:
    # #         writer = csv.writer(csvfile)
    # #         writer.writerow(['vx', 'vy', 'omega'])
    # #         for vxi, vyi, wi in zip(opt_vel[:, 0], opt_vel[:, 1], opt_omega):
    # #             writer.writerow([vxi, vyi, wi])

    # # # Plot
    # # plt.figure(figsize=(10, 10))
    # # plt.plot(positions[:, 0]/cell_size, positions[:, 1]/cell_size, 'r-', label="Center Spline Path")
    # # plt.plot(opt_pos[:, 0]/cell_size, opt_pos[:, 1]/cell_size, 'y-', label="Optimal Spline Path")
    # # plt.imshow(track_mask, cmap='gray', origin='lower')
    # # plt.plot(centerline_px[:, 1], centerline_px[:, 0], 'g-', label='Centerline')
    # # plt.legend()
    # # plt.title("Spline-Based Centerline and Velocities")
    # # plt.axis('equal')
    # # plt.grid(True)
    # # plt.show()

    ##### Trajectory Generation for all Tracks #####
    track_names = get_available_tracks()
    num_tracks = len(track_names)

    base_count = num_trajectories // num_tracks
    remainder = num_trajectories % num_tracks

    trajectories_per_track = {}
    for i, track_name in enumerate(track_names):
        trajectories_per_track[track_name] = base_count + (1 if i < remainder else 0)

    print("Distribution:")
    for track_name, count in trajectories_per_track.items():
        print(f"{track_name}: {count}")
    print("Total:", sum(trajectories_per_track.values()))
    
    traj_id = 0
    for track_name in track_names:
        print(f"Generating trajectories for {track_name} ...")

        map_x, map_y, vel_map, ori_map = create_racetrack(
            track_name, map_size, cell_size, boundary_size
        )

        track_mask = get_track_mask(vel_map)
        skeleton = extract_centerline_skeleton(track_mask)
        centerline_px = ordered_centerline_points(skeleton)

        if len(centerline_px) == 0:
            print(f"[WARNING] No centerline found for {track_name}, skipping.")
            continue

        centerline_m_orig = np.stack([
            centerline_px[:, 1] * cell_size,
            centerline_px[:, 0] * cell_size
        ], axis=-1)
        
        for i in tqdm(range(trajectories_per_track[track_name]), desc=track_name):
            centerline_m = shift_path_randomly(centerline_m_orig.copy())
            if np.random.rand() < 0.5:
                centerline_m = centerline_m[::-1]

            opt_pos, opt_vel, opt_omega, opt_cmds, theta0 = compute_optimal_line_mecanum(
                max_speed, centerline_m, num_points=num_points
            )

            traj_dir = os.path.join(output_base_dir, f'trajectory_{(traj_id + i):04d}')
            os.makedirs(traj_dir, exist_ok=True)

            with open(os.path.join(traj_dir, 'optimal_spline.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['vx', 'vy', 'omega', 'theta'])
                for vxi, vyi, wi in zip(opt_vel[:, 0], opt_vel[:, 1], opt_omega):
                    writer.writerow([vxi, vyi, wi, theta0])

            plt.figure(figsize=(20, 10))

            commands_np = np.concatenate([opt_vel, opt_omega[:, None]], axis=1)

            ref_positions = compute_reference_positions_numpy(
                commands_np,
                step_dt=0.02,
                commands_scale=(2.5, 2.5, 2.5),
                initial_pos=np.array([opt_pos[0, 0], opt_pos[0, 1], theta0])
            )

            reconstructed_pos = ref_positions[:, :2]
            theta_robot = ref_positions[:, 2]

            vx_world = (
                opt_vel[:, 0] * np.cos(theta_robot[:-1])
                - opt_vel[:, 1] * np.sin(theta_robot[:-1])
            )
            vy_world = (
                opt_vel[:, 0] * np.sin(theta_robot[:-1])
                + opt_vel[:, 1] * np.cos(theta_robot[:-1])
            )

            step = 4


            plt.figure(figsize=(10, 10))

            plt.plot(
                opt_pos[:, 0] / cell_size,
                opt_pos[:, 1] / cell_size,
                'y-',
                label="Optimal Spline Path"
            )

            plt.imshow(
                track_mask,
                cmap='gray',
                origin='lower'
            )

            plt.quiver(
                opt_pos[::step, 0] / cell_size,
                opt_pos[::step, 1] / cell_size,
                np.cos(theta_robot[:-1:step]),
                np.sin(theta_robot[:-1:step]),
                color='red',
                scale=25,
                width=0.003,
                label="Robot Orientation"
            )

            plt.quiver(
                opt_pos[::step, 0] / cell_size,
                opt_pos[::step, 1] / cell_size,
                vx_world[::step],
                vy_world[::step],
                color='blue',
                scale=20,
                width=0.003,
                label="Velocity Vector"
            )

            plt.legend()

            plt.title(
                f"{track_name} - Original Spline Path - "
                f"Trajectory {(traj_id + i):04d}"
            )

            plt.axis('equal')
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(
                os.path.join(traj_dir, 'original_spline.png'),
                dpi=300,
                bbox_inches='tight'
            )

            plt.close()


            plt.figure(figsize=(10, 10))

            plt.plot(
                reconstructed_pos[:, 0] / cell_size,
                reconstructed_pos[:, 1] / cell_size,
                'c-',
                label="Reconstructed Path"
            )

            plt.legend()

            plt.title(
                f"{track_name} - Reconstructed from Commands - "
                f"Trajectory {(traj_id + i):04d}"
            )

            plt.axis('equal')
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(
                os.path.join(traj_dir, 'reconstructed_path.png'),
                dpi=300,
                bbox_inches='tight'
            )

            plt.close()
        
        traj_id += i + 1
