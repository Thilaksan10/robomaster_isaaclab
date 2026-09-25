import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops


def _create_empty_maps(map_size, cell_size, boundary_size):
    start_map = -(map_size / 2.0)
    end_map = map_size / 2.0 + cell_size + (boundary_size * cell_size)

    x_pos = np.arange(start_map, end_map, cell_size)
    y_pos = np.arange(start_map, end_map, cell_size)

    map_y, map_x = np.meshgrid(y_pos, x_pos)
    map_y = map_y.transpose(0, 1)
    map_x = map_x.transpose(0, 1)

    map_xy = np.stack((map_x, map_y), axis=-1)
    xy_vels_grid = np.zeros_like(map_xy)
    orientation_map = np.zeros_like(map_xy)

    return map_x, map_y, xy_vels_grid, orientation_map

def _draw_closed_track(
    xy_vels_grid,
    waypoints,
    half_width=1.15,
    velocity=1.0,
):
    """
    Draw a closed racetrack through a list of grid waypoints.

    Parameters
    ----------
    xy_vels_grid : np.ndarray
        Velocity grid with shape (H, W, 2).

    waypoints : list of tuple
        Track centerline waypoints given as (row, column).

    half_width : float
        Half width of the track in grid cells.
        1.15 gives approximately the same ~3-cell width
        as the existing racetracks.

    velocity : float
        Magnitude of the velocity vectors written into the grid.
    """

    height, width = xy_vels_grid.shape[:2]

    rows, cols = np.mgrid[0:height, 0:width]

    # Used so overlapping parts around corners get the velocity
    # of the closest centerline segment.
    best_distance = np.full((height, width), np.inf)

    waypoints = np.asarray(waypoints, dtype=float)

    for i in range(len(waypoints)):
        start = waypoints[i]
        end = waypoints[(i + 1) % len(waypoints)]

        r0, c0 = start
        r1, c1 = end

        dr = r1 - r0
        dc = c1 - c0

        segment_length_sq = dr**2 + dc**2

        if segment_length_sq < 1e-8:
            continue

        # Projection of every grid point onto this segment
        t = (
            (rows - r0) * dr +
            (cols - c0) * dc
        ) / segment_length_sq

        t = np.clip(t, 0.0, 1.0)

        closest_r = r0 + t * dr
        closest_c = c0 + t * dc

        distance = np.sqrt(
            (rows - closest_r) ** 2 +
            (cols - closest_c) ** 2
        )

        segment_mask = distance <= half_width

        update_mask = (
            segment_mask &
            (distance < best_distance)
        )

        best_distance[update_mask] = distance[update_mask]

        segment_length = np.sqrt(segment_length_sq)

        # Column direction corresponds to x
        vx = velocity * dc / segment_length

        # Row direction corresponds to y
        vy = velocity * dr / segment_length

        xy_vels_grid[update_mask, 0] = vx
        xy_vels_grid[update_mask, 1] = vy


# def create_racetrack_1(map_size, cell_size, boundary_size):
#     map_x, map_y, xy_vels_grid, orientation_map = _create_empty_maps(
#         map_size, cell_size, boundary_size
#     )

#     velocity = 1.0
#     xy_vels_grid[1:4, 4:18, 0] = -velocity
#     xy_vels_grid[1:5, 1:4, 1] = velocity
#     xy_vels_grid[5:8, 1:10, 0] = velocity
#     xy_vels_grid[5:9, 10:13, 1] = velocity
#     xy_vels_grid[9:12, 4:13, 0] = -velocity
#     xy_vels_grid[9:15, 1:4, 1] = velocity
#     xy_vels_grid[15:18, 1:15, 0] = velocity
#     xy_vels_grid[4:18, 15:18, 1] = -velocity

#     return map_x, map_y, xy_vels_grid, orientation_map

def create_racetrack_1(map_size, cell_size, boundary_size):
    """
    Evaluation racetrack 1 - H-shaped closed racetrack.

    - Two large vertical stems
    - Clearly visible horizontal middle bar
    - Track half-width = 1.0
    - Single closed loop
    """

    map_x, map_y, xy_vels_grid, orientation_map = _create_empty_maps(
        map_size,
        cell_size,
        boundary_size,
    )

    waypoints = [
        (2, 2),
        (16, 2),
        (16, 6),
        (11, 6),
        (11, 12),
        (16, 12),
        (16, 16),
        (2, 16),
        (2, 12),
        (7, 12),
        (7, 6),
        (2, 6),
    ]

    _draw_closed_track(
        xy_vels_grid,
        waypoints,
        half_width=1.0,
        velocity=1.0,
    )

    return map_x, map_y, xy_vels_grid, orientation_map

def create_racetrack_2(map_size, cell_size, boundary_size):
    map_x, map_y, xy_vels_grid, orientation_map = _create_empty_maps(
        map_size, cell_size, boundary_size
    )

    velocity = 1.0
    xy_vels_grid[1:4, 1:15, 0] = velocity
    xy_vels_grid[1:15, 15:18, 1] = velocity
    xy_vels_grid[15:18, 4:18, 0] = -velocity
    xy_vels_grid[4:18, 1:4, 1] = -velocity

    return map_x, map_y, xy_vels_grid, orientation_map


def create_racetrack_3(map_size, cell_size, boundary_size):
    map_x, map_y, xy_vels_grid, orientation_map = _create_empty_maps(
        map_size, cell_size, boundary_size
    )

    velocity = 1.0

    xy_vels_grid[1:3, 7:10, 0] = velocity
    xy_vels_grid[1:7, 10:12, 1] = velocity
    xy_vels_grid[7:9, 10:16, 0] = velocity
    xy_vels_grid[7:10, 16:18, 1] = velocity
    xy_vels_grid[10:12, 12:18, 0] = -velocity
    xy_vels_grid[10:16, 10:12, 1] = velocity
    xy_vels_grid[16:18, 9:12, 0] = -velocity
    xy_vels_grid[12:18, 7:9, 1] = -velocity
    xy_vels_grid[10:12, 3:9, 0] = -velocity
    xy_vels_grid[9:12, 1:3, 1] = -velocity
    xy_vels_grid[7:9, 1:7, 0] = velocity
    xy_vels_grid[3:9, 7:9, 1] = -velocity

    return map_x, map_y, xy_vels_grid, orientation_map


# def create_racetrack_4(map_size, cell_size, boundary_size):
#     """
#     New racetrack 4:
#     Chicane + hairpin layout.

#     This is intentionally different from racetrack 2.
#     It contains straights, sharp turns, and left-right-left direction changes.
#     """
#     map_x, map_y, xy_vels_grid, orientation_map = _create_empty_maps(
#         map_size, cell_size, boundary_size
#     )

#     velocity = 1.0


#     xy_vels_grid[1:4, 1:8, 0] = velocity
    
#     xy_vels_grid[1:5, 8:11, 1] = velocity
#     xy_vels_grid[5:8, 8:15, 0] = velocity
#     xy_vels_grid[5:15, 15:18, 1] = velocity
#     xy_vels_grid[15:18, 8:18, 0] = -velocity

#     xy_vels_grid[12:18, 5:8, 1] = -velocity

#     xy_vels_grid[9:12, 4:8, 0] = -velocity

#     xy_vels_grid[4:12, 1:4, 1] = -velocity

#     return map_x, map_y, xy_vels_grid, orientation_map

def create_racetrack_4(map_size, cell_size, boundary_size):
    """
    Evaluation racetrack 4 - hard technical circuit.

    Properties:
    - many small and medium curves
    - several left-right transitions
    - asymmetric layout
    - no crossings
    - no narrow parallel sections
    - uses almost the complete grid
    - designed for the original skeleton-based centerline extraction
    """

    map_x, map_y, xy_vels_grid, orientation_map = _create_empty_maps(
        map_size,
        cell_size,
        boundary_size,
    )

    waypoints = [
        (2, 2),
        (2, 7),
        (4, 9),
        (2, 12),
        (2, 16),
        (7, 16),
        (9, 14),
        (11, 16),
        (16, 16),
        (16, 11),
        (14, 9),
        (16, 6),
        (16, 2),
        (11, 2),
        (9, 4),
        (7, 2),
        (5, 5),
    ]

    _draw_closed_track(
        xy_vels_grid,
        waypoints,
        half_width=1.0,
        velocity=1.0,
    )

    return map_x, map_y, xy_vels_grid, orientation_map


def create_racetrack_5(map_size, cell_size, boundary_size):
    """
    Validation racetrack 5 
    """

    map_x, map_y, xy_vels_grid, orientation_map = _create_empty_maps(
        map_size,
        cell_size,
        boundary_size,
    )

    waypoints = [
        (2, 2),
        (2, 6),
        (7, 6),
        (7, 10),
        (2, 10),
        (2, 16),
        (16, 16),
        (16, 13),
        (11, 13),
        (11, 9),
        (16, 9),
        (16, 2),
    ]

    _draw_closed_track(
        xy_vels_grid,
        waypoints,
        half_width=1.0,
        velocity=1.0,
    )

    return map_x, map_y, xy_vels_grid, orientation_map


TRACK_BUILDERS = {
    "track_1": create_racetrack_1,
    "track_2": create_racetrack_2,
    "track_3": create_racetrack_3,
    "track_4": create_racetrack_4,
    # 'track_5': create_racetrack_5
}


def create_racetrack(track_name, map_size, cell_size, boundary_size):
    if track_name not in TRACK_BUILDERS:
        raise ValueError(
            f"Unknown track_name: {track_name}. Available: {list(TRACK_BUILDERS.keys())}"
        )

    return TRACK_BUILDERS[track_name](map_size, cell_size, boundary_size)


def get_available_tracks():
    return list(TRACK_BUILDERS.keys())


def plot_racetrack(track_name, map_size=20, cell_size=1.0, boundary_size=0, max_speed=1.5):
    map_x, map_y, xy_vels_grid, _ = create_racetrack(
        track_name, map_size, cell_size, boundary_size
    )


    track_mask = get_track_mask(xy_vels_grid)
    skeleton = extract_centerline_skeleton(track_mask)
    centerline_px = ordered_centerline_points(skeleton)

    centerline_m_orig = np.stack([
        centerline_px[:, 1] * cell_size,
        centerline_px[:, 0] * cell_size
    ], axis=-1)

    centerline_m = shift_path_randomly(centerline_m_orig.copy())
    if np.random.rand() < 0.2:
        centerline_m = centerline_m[::-1]

    # Compute spline and optimal trajectory
    positions, velocities, omega, commands = fit_spline_and_extract_velocities(centerline_m)
    # opt_pos, opt_vel, opt_omega, opt_cmds, theta0 = compute_optimal_line(centerline_m)
    opt_pos, opt_vel, opt_omega, opt_cmds, theta0 = compute_optimal_line_mecanum(max_speed, centerline_m)

    plt.plot(opt_pos[:, 0]/cell_size, opt_pos[:, 1]/cell_size, 'y-', label="Optimal Spline Path")
    plt.imshow(track_mask, cmap='gray', origin='lower')
    plt.plot(centerline_px[:, 1], centerline_px[:, 0], 'g-', label='Original Centerline')
    # Compute robot orientation
    theta_robot = np.arctan2(opt_vel[:, 1], opt_vel[:, 0])
    theta_robot = integrate_orientation(theta_robot, opt_omega)
    # vel_direction = np.arctan2(opt_vel[:, 1], opt_vel[:, 0])
    vx_world = opt_vel[:, 0] * np.cos(theta_robot) - opt_vel[:, 1] * np.sin(theta_robot)
    vy_world = opt_vel[:, 0] * np.sin(theta_robot) + opt_vel[:, 1] * np.cos(theta_robot)


    vx = xy_vels_grid[:, :, 0]
    vy = xy_vels_grid[:, :, 1]

    mask = (vx != 0) | (vy != 0)
    rows, cols = np.where(mask)

    plt.quiver(
        cols,
        rows,
        vx[mask],
        vy[mask],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="red",
        width=0.004,
        label="Track Velocity Field"
    )

    # Plot orientation arrows
    step = 20  # reduce arrow density/ dt
    plt.quiver(
        opt_pos[::step, 0]/cell_size,
        opt_pos[::step, 1]/cell_size,
        np.cos(theta_robot[::step]),
        np.sin(theta_robot[::step]),
        color='red',
        scale=25,
        width=0.003,
        label="Robot Orientation"
    )
    plt.quiver(
        opt_pos[::step, 0] / cell_size,
        opt_pos[::step, 1] / cell_size,
        # np.cos(vel_direction[::step]),
        # np.sin(vel_direction[::step]),
        vx_world[::step],
        vy_world[::step],
        color='blue',
        scale=20,
        width=0.003,
        label="Velocity Vector"
    )

    plt.legend()
    plt.title(f"Spline Path and Optimal Path")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_all_racetracks(map_size=20, cell_size=1.0, boundary_size=0):
    for track_name in get_available_tracks():
        plot_racetrack(track_name, map_size, cell_size, boundary_size)

def get_track_mask(vel_map, threshold=0.1):
    speed = np.linalg.norm(vel_map, axis=-1)
    return (speed > threshold).astype(np.uint8)

def extract_centerline_skeleton(track_mask):
    skeleton = skeletonize(track_mask)  # from skimage
    return skeleton.astype(np.uint8)

# def ordered_centerline_points(skeleton):
#     coords = np.argwhere(skeleton)
#     if len(coords) == 0:
#         return np.empty((0, 2))

#     # Use regionprops to find connected components
#     labeled = label(skeleton)
#     regions = regionprops(labeled)

#     # Take the largest component (main track)
#     main = max(regions, key=lambda r: r.area)
#     coords = main.coords

#     # Optional: sort points based on proximity (nearest neighbor path)
#     from scipy.spatial.distance import cdist
#     ordered = [coords[0]]
#     used = set([0])
#     for _ in range(len(coords) - 1):
#         dists = cdist([ordered[-1]], coords)
#         dists[0][list(used)] = np.inf
#         next_idx = np.argmin(dists)
#         ordered.append(coords[next_idx])
#         used.add(next_idx)
#     return np.array(ordered)

def ordered_centerline_points(skeleton):
    """
    Order skeleton pixels by following the actual connected skeleton.

    Unlike the old nearest-neighbor implementation, this function
    only moves to directly connected skeleton pixels and therefore
    cannot jump between nearby sections of the racetrack.
    """

    # Keep only the largest connected component
    labeled = label(skeleton, connectivity=2)
    regions = regionprops(labeled)

    if len(regions) == 0:
        return np.empty((0, 2))

    main = max(regions, key=lambda r: r.area)

    component = np.zeros_like(skeleton, dtype=np.uint8)

    for r, c in main.coords:
        component[r, c] = 1

    coords = np.argwhere(component)

    if len(coords) == 0:
        return np.empty((0, 2))

    # Convert coordinates to a set for fast lookup
    pixel_set = set(map(tuple, coords))

    # 8-connected neighborhood
    neighbor_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    def get_neighbors(point):
        r, c = point

        neighbors = []

        for dr, dc in neighbor_offsets:
            candidate = (r + dr, c + dc)

            if candidate in pixel_set:
                neighbors.append(candidate)

        return neighbors

    start = tuple(coords[np.lexsort((coords[:, 1], coords[:, 0]))][0])

    ordered = [start]
    visited = {start}

    previous = None
    current = start


    while True:

        neighbors = get_neighbors(current)

        # Remove already visited pixels
        candidates = [
            n for n in neighbors
            if n not in visited
        ]

        if len(candidates) == 0:
            break

        if previous is None:
            next_point = candidates[0]


        elif len(candidates) == 1:
            next_point = candidates[0]

        else:
            prev_direction = np.array(current) - np.array(previous)

            prev_norm = np.linalg.norm(prev_direction)

            if prev_norm > 0:
                prev_direction = prev_direction / prev_norm

            best_score = -np.inf
            next_point = candidates[0]

            for candidate in candidates:

                direction = (
                    np.array(candidate) -
                    np.array(current)
                )

                direction_norm = np.linalg.norm(direction)

                if direction_norm > 0:
                    direction = direction / direction_norm

                # Higher dot product = smaller heading change
                score = np.dot(
                    prev_direction,
                    direction
                )

                if score > best_score:
                    best_score = score
                    next_point = candidate

        previous = current
        current = next_point

        ordered.append(current)
        visited.add(current)

    ordered = np.asarray(ordered)

    print(
        f"Skeleton pixels: {len(coords)}, "
        f"ordered pixels: {len(ordered)}"
    )

    return ordered

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
    theta = theta_start

    for i in range(1, len(omega)):
        theta[i] = theta[i-1] + omega[i-1] * dt

    return theta

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
    max_target_step=np.pi / 2,
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

    # ---- same path generation as compute_optimal_line ----
    tck, _ = splprep([centerline[:, 1], centerline[:, 0]], s=1.0)
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

        heading_error = wrap_to_pi(target_heading - theta[i])

        omega_cmd = kp_heading * heading_error
        omega_cmd = np.clip(omega_cmd, -omega_limit, omega_limit)

        omega[i] = omega_cmd
        theta[i + 1] = wrap_to_pi(theta[i] + omega[i] * dt)

    omega[-1] = omega[-2] if N > 1 else 0.0

    # ---- optional smoothing of omega to remove small jerks ----
    kernel_size = 9
    kernel = np.ones(kernel_size) / kernel_size
    omega = np.convolve(omega, kernel, mode="same")

    # re-integrate theta from smoothed omega
    theta[0] = theta[0]
    for i in range(N - 1):
        theta[i + 1] = wrap_to_pi(theta[i] + omega[i] * dt)

    # ---- convert world velocity to robot frame using current heading ----
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vx_robot = cos_t * vx_world + sin_t * vy_world
    vy_robot = -sin_t * vx_world + cos_t * vy_world
    vel_robot = np.stack([vx_robot, vy_robot], axis=1)

    # ---- scaling ----
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

    return optimal_positions, commands[:, :2], commands[:, 2], commands, theta[0]


if __name__ == "__main__":
    plot_all_racetracks(
        map_size=6,
        cell_size=0.4,
        boundary_size=3
    )

    # Plot and save figure
    # plt.figure(figsize=(10, 10))

    # # plt.plot(positions[:, 0]/cell_size, positions[:, 1]/cell_size, 'r-', label="Center Spline Path")
    # # plt.plot(opt_pos[:, 0]/cell_size, opt_pos[:, 1]/cell_size, 'y-', label="Optimal Spline Path")
    # plt.imshow(track_mask, cmap='gray', origin='lower')
    # # plt.plot(centerline_px[:, 1], centerline_px[:, 0], 'g-', label='Original Centerline')
    # # Compute robot orientation
    # # theta_robot = np.arctan2(opt_vel[:, 1], opt_vel[:, 0])
    # theta_robot = integrate_orientation(theta_robot, opt_omega)
    # # vel_direction = np.arctan2(opt_vel[:, 1], opt_vel[:, 0])
    # vx_world = opt_vel[:, 0] * np.cos(theta_robot) - opt_vel[:, 1] * np.sin(theta_robot)
    # vy_world = opt_vel[:, 0] * np.sin(theta_robot) + opt_vel[:, 1] * np.cos(theta_robot)

    # plt.legend()
    # plt.title(f"Spline Path and Optimal Path - Trajectory {i:03d}")
    # plt.axis('equal')
    # plt.grid(True)
    # plt.show()
