import numpy as np
import matplotlib.pyplot as plt
import os


N_STEPS = 300
DT = 0.02

MAX_LINEAR_SPEED = 2.5
MAX_ANGULAR_SPEED = 2.5

OUTPUT_DIR = "edge_cases"


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def clamp_linear_velocity(vx, vy, max_speed=MAX_LINEAR_SPEED):
    speed = np.sqrt(vx**2 + vy**2)
    scale = np.ones_like(speed)

    mask = speed > max_speed
    scale[mask] = max_speed / speed[mask]

    return vx * scale, vy * scale


def clamp_angular_velocity(omega, max_omega=MAX_ANGULAR_SPEED):
    return np.clip(omega, -max_omega, max_omega)


def compute_positions_like_sim(vx, vy, omega, theta0=0.0, dt=DT):
    pose = np.zeros(3, dtype=np.float32)
    pose[2] = theta0

    positions = [pose[:2].copy()]
    thetas = [pose[2].copy()]

    for i in range(len(vx)):
        theta = pose[2]

        dx = (np.cos(theta) * vx[i] - np.sin(theta) * vy[i]) * dt
        dy = (np.sin(theta) * vx[i] + np.cos(theta) * vy[i]) * dt
        dtheta = omega[i] * dt

        pose[0] += dx
        pose[1] += dy
        pose[2] = wrap_to_pi(pose[2] + dtheta)

        positions.append(pose[:2].copy())
        thetas.append(pose[2].copy())

    return np.asarray(positions), np.asarray(thetas)


def make_trajectory(name, vx, vy, omega=None, theta0=0.0):
    if omega is None:
        omega = np.zeros_like(vx)

    vx, vy = clamp_linear_velocity(vx, vy)
    omega = clamp_angular_velocity(omega)

    assert len(vx) == N_STEPS
    assert len(vy) == N_STEPS
    assert len(omega) == N_STEPS

    positions, theta_ref = compute_positions_like_sim(
        vx=vx,
        vy=vy,
        omega=omega,
        theta0=theta0,
        dt=DT,
    )

    return {
        "name": name,
        "pos": positions,                    # 301 x 2
        "vel": np.stack([vx, vy], axis=-1),  # 300 x 2
        "omega": omega,                      # 300
        "theta": theta_ref,                  # 301
        "theta0": theta0,
    }



def traj_curve_left_90(n=N_STEPS, radius=5.0):
    vx = np.ones(n) * 2.5 / 1.9
    vy = np.zeros(n)
    omega = np.ones(n) * min(2.5 / radius, MAX_ANGULAR_SPEED) / 1.9

    return make_trajectory("Left Curve 90°", vx, vy, omega)


def traj_curve_right_90(n=N_STEPS, radius=5.0):
    vx = np.ones(n) * 2.5 / 1.9
    vy = np.zeros(n)
    omega = np.ones(n) * -min(2.5 / radius, MAX_ANGULAR_SPEED) / 1.9

    return make_trajectory("Right Curve 90°", vx, vy, omega)


def traj_circle_small(n=N_STEPS, radius=1.0):
    vx = np.ones(n) * 2.5 / 2.4
    vy = np.zeros(n)
    omega = np.ones(n) * min(2.5 / radius, MAX_ANGULAR_SPEED) / 2.4 

    return make_trajectory("Circle Small Radius", vx, vy, omega)


def traj_circle_big(n=N_STEPS, radius=2.4):
    vx = np.ones(n) * 2.5
    vy = np.zeros(n)
    omega = np.ones(n) * min(2.5 / radius, MAX_ANGULAR_SPEED)

    return make_trajectory("Circle Big Radius", vx, vy, omega)


def traj_forward_fixed_orientation(theta0, n=N_STEPS):
    vx_world = np.ones(n) * 2.5
    vy_world = np.zeros(n)

    vx_body = vx_world * np.cos(theta0) + vy_world * np.sin(theta0)
    vy_body = -vx_world * np.sin(theta0) + vy_world * np.cos(theta0)

    omega = np.zeros(n)

    name = f"Fixed Orientation {np.rad2deg(theta0):.0f}°"

    return make_trajectory(name, vx_body, vy_body, omega, theta0=theta0)


def traj_zigzag(n=N_STEPS):
    t = np.arange(n) * DT

    vx = np.ones(n) * 2.5
    vy = 2.0 * np.sin(2.0 * np.pi * 0.8 * t)

    vx, vy = clamp_linear_velocity(vx, vy)

    omega = np.zeros(n)

    return make_trajectory("Zigzag Lateral Motion", vx, vy, omega)


def traj_sharp_s_curves(n=N_STEPS, num_u_turns=8):
    """
    Accelerate to max speed and perform up to 8 very sharp,
    directly connected 180-degree U-turns.

    NOTE:
    This trajectory requires angular velocities above the current
    MAX_ANGULAR_SPEED = 2.5 rad/s for large num_u_turns.
    """

    if not 1 <= num_u_turns <= 9:
        raise ValueError("num_u_turns must be between 1 and 9")

    vx = np.zeros(n)
    vy = np.zeros(n)
    omega = np.zeros(n)

    accel_steps = 30  # 0.6 s

    vx[:accel_steps] = np.linspace(
        0.0,
        MAX_LINEAR_SPEED,
        accel_steps,
    )

    vx[accel_steps:] = MAX_LINEAR_SPEED

    # Short straight before the first U-turn
    straight_before = 10

    turn_start = accel_steps + straight_before

    straight_after = 20

    available_turn_steps = (
        n
        - turn_start
        - straight_after
    )

    turn_steps = available_turn_steps // num_u_turns

    turn_duration = turn_steps * DT

    # Angular velocity required for exactly 180 degrees
    omega_value = np.pi / turn_duration

    print(f"Number of U-turns: {num_u_turns}")
    print(f"Steps per U-turn: {turn_steps}")
    print(f"Turn duration: {turn_duration:.3f} s")
    print(f"Required omega: {omega_value:.3f} rad/s")
    print(
        f"Turn radius: "
        f"{MAX_LINEAR_SPEED / omega_value:.3f} m"
    )

    current = turn_start

    for i in range(num_u_turns):
        end = current + turn_steps

        if i % 2 == 0:
            omega[current:end] = omega_value
        else:
            omega[current:end] = -omega_value

        current = end


    return make_sharp_trajectory(
        f"Connected U-Curves ({num_u_turns})",
        vx,
        vy,
        omega,
    )

def make_sharp_trajectory(
    name,
    vx,
    vy,
    omega=None,
    theta0=0.0,
):
    if omega is None:
        omega = np.zeros_like(vx)


    vx, vy = clamp_linear_velocity(vx, vy)

    assert len(vx) == N_STEPS
    assert len(vy) == N_STEPS
    assert len(omega) == N_STEPS

    positions, theta_ref = compute_positions_like_sim(
        vx=vx,
        vy=vy,
        omega=omega,
        theta0=theta0,
        dt=DT,
    )

    return {
        "name": name,
        "pos": positions,
        "vel": np.stack([vx, vy], axis=-1),
        "omega": omega,
        "theta": theta_ref,
        "theta0": theta0,
    }


def traj_curve_left_180(n=N_STEPS):
    """
    Accelerate to maximum speed, drive straight,
    perform a wide 180-degree U-turn at maximum speed,
    then continue straight.
    """

    vx = np.zeros(n)
    vy = np.zeros(n)
    omega = np.zeros(n)


    accel_steps = 50      
    straight_steps = 50   
    turn_steps = 150      

    turn_start = accel_steps + straight_steps
    turn_end = turn_start + turn_steps


    vx[:accel_steps] = np.linspace(
        0.0,
        MAX_LINEAR_SPEED,
        accel_steps
    )

    vx[accel_steps:] = MAX_LINEAR_SPEED


    turn_duration = turn_steps * DT


    omega_value = np.pi / turn_duration

    omega[turn_start:turn_end] = omega_value

    return make_trajectory(
        "Left Curve 180°",
        vx,
        vy,
        omega,
    )


def traj_curve_right_180(n=N_STEPS):
    """
    Accelerate to maximum speed, drive straight,
    perform a wide 180-degree right U-turn at maximum speed,
    then continue straight.
    """

    vx = np.zeros(n)
    vy = np.zeros(n)
    omega = np.zeros(n)

    accel_steps = 50
    straight_steps = 50
    turn_steps = 150

    turn_start = accel_steps + straight_steps
    turn_end = turn_start + turn_steps

    # Accelerate
    vx[:accel_steps] = np.linspace(
        0.0,
        MAX_LINEAR_SPEED,
        accel_steps
    )

    # Maximum speed
    vx[accel_steps:] = MAX_LINEAR_SPEED

    # 180-degree turn
    turn_duration = turn_steps * DT
    omega_value = -np.pi / turn_duration

    omega[turn_start:turn_end] = omega_value

    return make_trajectory(
        "Right Curve 180°",
        vx,
        vy,
        omega,
    )

def traj_changing_velocity_profiles(
    n=N_STEPS,
    profile_steps=1,
    seed=24,
):
    """
    Rapidly changing but smooth velocity profiles.

    A new [vx, vy, omega] command is generated every
    `profile_steps` simulation steps.

    The new command is based on the previous command with only
    a small random change, creating a smooth random-walk trajectory.
    """

    rng = np.random.default_rng(seed)

    vx = np.zeros(n)
    vy = np.zeros(n)
    omega = np.zeros(n)


    current_vx = 0.0
    current_vy = 0.0
    current_omega = 0.0


    max_delta_vx = 0.10
    max_delta_vy = 0.10
    max_delta_omega = 0.10

    # Keep the random trajectory in sensible ranges
    vx_min, vx_max = 0.5, 2.2
    vy_min, vy_max = -1.2, 1.2
    omega_min, omega_max = -1.2, 1.2

    for start in range(0, n, profile_steps):
        end = min(start + profile_steps, n)

        # Small change relative to previous profile
        current_vx += rng.uniform(
            -max_delta_vx,
            max_delta_vx,
        )

        current_vy += rng.uniform(
            -max_delta_vy,
            max_delta_vy,
        )

        current_omega += rng.uniform(
            -max_delta_omega,
            max_delta_omega,
        )

        current_vx = np.clip(
            current_vx,
            vx_min,
            vx_max,
        )

        current_vy = np.clip(
            current_vy,
            vy_min,
            vy_max,
        )

        current_omega = np.clip(
            current_omega,
            omega_min,
            omega_max,
        )

        speed = np.sqrt(
            current_vx**2
            + current_vy**2
        )

        if speed > MAX_LINEAR_SPEED:
            scale = MAX_LINEAR_SPEED / speed

            current_vx *= scale
            current_vy *= scale


        vx[start:end] = current_vx
        vy[start:end] = current_vy
        omega[start:end] = current_omega

    return make_trajectory(
        "Changing Velocities",
        vx,
        vy,
        omega,
    )

def get_edge_case_trajectories():
    trajectories = [
        traj_curve_left_90(),
        traj_curve_right_90(),
        traj_circle_small(),
        traj_circle_big(),
        traj_zigzag(),

        traj_curve_left_180(),
        traj_curve_right_180(),

        traj_changing_velocity_profiles()

    ]

    fixed_orientations = [
        0.0,
        np.pi / 4.0,
        np.pi / 2.0,
        np.pi,
        -np.pi / 2.0,
    ]

    for theta0 in fixed_orientations:
        trajectories.append(traj_forward_fixed_orientation(theta0))

    for n in range(3,10):
        trajectories.append(traj_sharp_s_curves(num_u_turns=n))

    return trajectories


def save_trajectory_csv(traj, output_dir):
    vel = traj["vel"]
    omega = traj["omega"]
    theta = traj["theta"]

    theta_cmd = theta[:-1]

    vx_norm = vel[:, 0] / MAX_LINEAR_SPEED
    vy_norm = vel[:, 1] / MAX_LINEAR_SPEED
    omega_norm = omega / MAX_ANGULAR_SPEED

    data = np.column_stack([
        vx_norm,
        vy_norm,
        omega_norm,
        theta_cmd,
    ])

    csv_filename = os.path.join(output_dir, f"{traj['name']}.csv")

    np.savetxt(
        csv_filename,
        data,
        delimiter=",",
        header="vx,vy,omega,theta",
        comments=""
    )


def plot_trajectory(traj, step=10):
    pos = traj["pos"]
    vel = traj["vel"]
    theta = traj["theta"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    arrow_pos = pos[:-1]
    arrow_theta = theta[:-1]

    vx_body = vel[:, 0]
    vy_body = vel[:, 1]

    vx_world = np.cos(arrow_theta) * vx_body - np.sin(arrow_theta) * vy_body
    vy_world = np.sin(arrow_theta) * vx_body + np.cos(arrow_theta) * vy_body

    plt.figure(figsize=(6, 6))

    plt.plot(pos[:, 0], pos[:, 1], "k-", label="sim-integrated trajectory")
    plt.scatter(pos[0, 0], pos[0, 1], label="start")
    plt.scatter(pos[-1, 0], pos[-1, 1], label="end")

    plt.quiver(
        arrow_pos[::step, 0],
        arrow_pos[::step, 1],
        vx_world[::step],
        vy_world[::step],
        angles="xy",
        scale_units="xy",
        scale=2.5,
        width=0.004,
        color="dodgerblue",
        label="world velocity",
    )

    plt.quiver(
        arrow_pos[::step, 0],
        arrow_pos[::step, 1],
        np.cos(arrow_theta[::step]),
        np.sin(arrow_theta[::step]),
        angles="xy",
        scale_units="xy",
        scale=2.0,
        width=0.003,
        color="crimson",
        label="robot orientation",
    )

    plt.title(traj["name"])
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    image_filename = os.path.join(OUTPUT_DIR, f"{traj['name']}.png")

    plt.savefig(image_filename, dpi=300, bbox_inches="tight")
    plt.close()

    save_trajectory_csv(traj, OUTPUT_DIR)


def plot_all_edge_cases():
    for traj in get_edge_case_trajectories():
        plot_trajectory(traj)


if __name__ == "__main__":
    plot_all_edge_cases()