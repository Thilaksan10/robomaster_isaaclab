import numpy as np
import matplotlib.pyplot as plt
import os


N_STEPS = 300
DT = 0.02

MAX_LINEAR_SPEED = 2.5
MAX_ANGULAR_SPEED = 2.5

OUTPUT_DIR = "training_trajectories"


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


def smooth_step_profile(start, end, n):
    s = np.linspace(0.0, 1.0, n)
    s = 3.0 * s**2 - 2.0 * s**3
    return start + (end - start) * s


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
        "vel": np.stack([vx, vy], axis=-1)/MAX_LINEAR_SPEED,
        "omega": omega/MAX_ANGULAR_SPEED,
        "theta": theta_ref,
        "theta0": theta0,
    }


def rand_speed(min_abs=0.4, max_abs=MAX_LINEAR_SPEED, sign=None):
    value = np.random.uniform(min_abs, max_abs)

    if sign is None:
        sign = np.random.choice([-1.0, 1.0])

    return sign * value


def rand_omega(min_abs=0.2, max_abs=MAX_ANGULAR_SPEED, sign=None):
    value = np.random.uniform(min_abs, max_abs)

    if sign is None:
        sign = np.random.choice([-1.0, 1.0])

    return sign * value


def add_small_noise(vx, vy, omega, noise_scale=0.05):
    vx = vx + np.random.normal(0.0, noise_scale, size=vx.shape)
    vy = vy + np.random.normal(0.0, noise_scale, size=vy.shape)
    omega = omega + np.random.normal(0.0, noise_scale, size=omega.shape)

    vx, vy = clamp_linear_velocity(vx, vy)
    omega = clamp_angular_velocity(omega)

    return vx, vy, omega


def gen_forward_then_backward(n=N_STEPS):
    vx = np.zeros(n)
    vy = np.zeros(n)
    omega = np.zeros(n)

    forward_speed = np.random.uniform(0.6, MAX_LINEAR_SPEED)
    backward_speed = -np.random.uniform(0.6, MAX_LINEAR_SPEED)

    forward_end = np.random.randint(int(0.20 * n), int(0.40 * n))
    slowdown_end = np.random.randint(forward_end + 20, int(0.55 * n))
    backward_end = np.random.randint(int(0.70 * n), int(0.95 * n))

    vx[:forward_end] = forward_speed
    vx[forward_end:slowdown_end] = smooth_step_profile(
        forward_speed, 0.0, slowdown_end - forward_end
    )
    vx[slowdown_end:backward_end] = backward_speed
    vx[backward_end:] = smooth_step_profile(
        backward_speed, 0.0, n - backward_end
    )

    vx, vy, omega = add_small_noise(vx, vy, omega, noise_scale=0.02)

    return make_trajectory("train_forward_then_backward", vx, vy, omega)


def gen_curve(n=N_STEPS):
    vx = np.ones(n) * np.random.uniform(0.6, MAX_LINEAR_SPEED)
    vy = np.zeros(n)

    radius = np.random.uniform(1.0, 8.0)
    direction = np.random.choice([-1.0, 1.0])

    omega_value = direction * min(abs(vx[0]) / radius, MAX_ANGULAR_SPEED)
    omega = np.ones(n) * omega_value

    vx, vy, omega = add_small_noise(vx, vy, omega, noise_scale=0.02)

    name = "train_curve_left" if direction > 0 else "train_curve_right"
    return make_trajectory(name, vx, vy, omega)


def gen_circle_like(n=N_STEPS):
    vx = np.ones(n) * np.random.uniform(0.6, MAX_LINEAR_SPEED)
    vy = np.zeros(n)

    radius = np.random.uniform(0.8, 4.0)
    direction = np.random.choice([-1.0, 1.0])

    omega_value = direction * min(abs(vx[0]) / radius, MAX_ANGULAR_SPEED)
    omega = np.ones(n) * omega_value

    vx, vy, omega = add_small_noise(vx, vy, omega, noise_scale=0.015)

    return make_trajectory("train_circle_like", vx, vy, omega)


def gen_forward_fixed_orientation(n=N_STEPS):
    theta0 = np.random.uniform(-np.pi, np.pi)

    speed_world = np.random.uniform(0.5, MAX_LINEAR_SPEED)

    vx_world = np.ones(n) * speed_world
    vy_world = np.zeros(n)

    vx_body = vx_world * np.cos(theta0) + vy_world * np.sin(theta0)
    vy_body = -vx_world * np.sin(theta0) + vy_world * np.cos(theta0)

    omega = np.zeros(n)

    vx_body, vy_body, omega = add_small_noise(
        vx_body, vy_body, omega, noise_scale=0.02
    )

    return make_trajectory(
        "train_forward_fixed_orientation",
        vx_body,
        vy_body,
        omega,
        theta0=theta0,
    )


def world_x_motion_body_commands(n=N_STEPS, speed_world=2.5, omega_value=2.5, theta0=0.0):
    t = np.arange(n) * DT
    theta = wrap_to_pi(theta0 + omega_value * t)

    vx_world = np.ones(n) * speed_world
    vy_world = np.zeros(n)

    vx_body = vx_world * np.cos(theta) + vy_world * np.sin(theta)
    vy_body = -vx_world * np.sin(theta) + vy_world * np.cos(theta)

    omega = np.ones(n) * omega_value

    return vx_body, vy_body, omega


def gen_spin_while_world_x(n=N_STEPS):
    speed_world = rand_speed(min_abs=0.5, max_abs=MAX_LINEAR_SPEED)
    omega_value = rand_omega(min_abs=0.5, max_abs=MAX_ANGULAR_SPEED)
    theta0 = np.random.uniform(-np.pi, np.pi)

    vx, vy, omega = world_x_motion_body_commands(
        n=n,
        speed_world=speed_world,
        omega_value=omega_value,
        theta0=theta0,
    )

    vx, vy, omega = add_small_noise(vx, vy, omega, noise_scale=0.015)

    return make_trajectory(
        "train_spin_while_world_x",
        vx,
        vy,
        omega,
        theta0=theta0,
    )


def gen_spin_in_place(n=N_STEPS):
    vx = np.zeros(n)
    vy = np.zeros(n)

    omega_value = rand_omega(min_abs=0.5, max_abs=MAX_ANGULAR_SPEED)
    omega = np.ones(n) * omega_value

    return make_trajectory("train_spin_in_place", vx, vy, omega)


def gen_zigzag(n=N_STEPS):
    t = np.arange(n) * DT

    vx_base = np.random.uniform(0.5, MAX_LINEAR_SPEED)
    amp = np.random.uniform(0.3, 2.0)
    freq = np.random.uniform(0.3, 1.2)

    vx = np.ones(n) * vx_base
    vy = amp * np.sin(2.0 * np.pi * freq * t)
    omega = np.zeros(n)

    vx, vy = clamp_linear_velocity(vx, vy)

    return make_trajectory("train_zigzag", vx, vy, omega)


def gen_forward_then_stop(n=N_STEPS):
    vx = np.ones(n) * np.random.uniform(0.6, MAX_LINEAR_SPEED)
    vy = np.zeros(n)
    omega = np.zeros(n)

    stop_start = np.random.randint(int(0.25 * n), int(0.60 * n))
    stop_end = np.random.randint(stop_start + 20, int(0.85 * n))

    vx[stop_start:stop_end] = smooth_step_profile(
        vx[stop_start - 1], 0.0, stop_end - stop_start
    )
    vx[stop_end:] = 0.0

    return make_trajectory("train_forward_then_stop", vx, vy, omega)


def gen_lateral_motion(n=N_STEPS):
    vx = np.zeros(n)
    vy = np.ones(n) * rand_speed(min_abs=0.5, max_abs=MAX_LINEAR_SPEED)
    omega = np.zeros(n)

    return make_trajectory("train_lateral_motion", vx, vy, omega)


def gen_diagonal_motion(n=N_STEPS):
    angle = np.random.uniform(-np.pi, np.pi)
    speed = np.random.uniform(0.5, MAX_LINEAR_SPEED)

    vx = np.ones(n) * speed * np.cos(angle)
    vy = np.ones(n) * speed * np.sin(angle)
    omega = np.zeros(n)

    vx, vy = clamp_linear_velocity(vx, vy)

    return make_trajectory("train_diagonal_motion", vx, vy, omega)


def gen_s_curve(n=N_STEPS):
    vx = np.ones(n) * np.random.uniform(0.6, MAX_LINEAR_SPEED)
    vy = np.zeros(n)

    omega = np.zeros(n)
    half = n // 2

    omega_peak = np.random.uniform(0.4, MAX_ANGULAR_SPEED)
    direction = np.random.choice([-1.0, 1.0])

    omega[:half] = smooth_step_profile(0.0, direction * omega_peak, half)
    omega[half:] = smooth_step_profile(
        direction * omega_peak,
        -direction * omega_peak,
        n - half,
    )

    return make_trajectory("train_s_curve", vx, vy, omega)


def gen_mixed_segments(n=N_STEPS):
    vx = np.zeros(n)
    vy = np.zeros(n)
    omega = np.zeros(n)

    num_segments = np.random.randint(3, 6)
    split_points = np.linspace(0, n, num_segments + 1, dtype=int)

    for i in range(num_segments):
        a = split_points[i]
        b = split_points[i + 1]
        length = b - a

        primitive = np.random.choice([
            "straight",
            "lateral",
            "spin",
            "curve",
            "diagonal",
        ])

        if primitive == "straight":
            vx[a:b] = rand_speed(0.4, MAX_LINEAR_SPEED)
            vy[a:b] = 0.0
            omega[a:b] = 0.0

        elif primitive == "lateral":
            vx[a:b] = 0.0
            vy[a:b] = rand_speed(0.4, MAX_LINEAR_SPEED)
            omega[a:b] = 0.0

        elif primitive == "spin":
            vx[a:b] = 0.0
            vy[a:b] = 0.0
            omega[a:b] = rand_omega(0.4, MAX_ANGULAR_SPEED)

        elif primitive == "curve":
            vx_val = np.random.uniform(0.5, MAX_LINEAR_SPEED)
            radius = np.random.uniform(1.0, 6.0)
            direction = np.random.choice([-1.0, 1.0])

            vx[a:b] = vx_val
            vy[a:b] = 0.0
            omega[a:b] = direction * min(vx_val / radius, MAX_ANGULAR_SPEED)

        elif primitive == "diagonal":
            speed = np.random.uniform(0.5, MAX_LINEAR_SPEED)
            angle = np.random.uniform(-np.pi, np.pi)
            vx[a:b] = speed * np.cos(angle)
            vy[a:b] = speed * np.sin(angle)
            omega[a:b] = 0.0

        if length > 5 and i > 0:
            blend_len = min(10, length)
            prev_vx = vx[a - 1]
            prev_vy = vy[a - 1]
            prev_omega = omega[a - 1]

            vx[a:a + blend_len] = smooth_step_profile(prev_vx, vx[a], blend_len)
            vy[a:a + blend_len] = smooth_step_profile(prev_vy, vy[a], blend_len)
            omega[a:a + blend_len] = smooth_step_profile(prev_omega, omega[a], blend_len)

    vx, vy = clamp_linear_velocity(vx, vy)
    omega = clamp_angular_velocity(omega)

    return make_trajectory("train_mixed_segments", vx, vy, omega)


TRAINING_GENERATORS = {
    "forward_then_backward": gen_forward_then_backward,
    "curve": gen_curve,
    "circle_like": gen_circle_like,
    "forward_fixed_orientation": gen_forward_fixed_orientation,
    "spin_while_world_x": gen_spin_while_world_x,
    "spin_in_place": gen_spin_in_place,
    "zigzag": gen_zigzag,
    "forward_then_stop": gen_forward_then_stop,
    "lateral_motion": gen_lateral_motion,
    "diagonal_motion": gen_diagonal_motion,
    "s_curve": gen_s_curve,
    "mixed_segments": gen_mixed_segments,
}


# def generate_training_trajectory(idx=None):
#     generator = np.random.choice(TRAINING_GENERATORS)
#     traj = generator()

#     if idx is not None:
#         traj["name"] = f"{idx:05d}_{traj['name']}"

#     return traj

def generate_training_trajectory(
    idx=None,
    enabled_trajectories=None,
):
    if enabled_trajectories is None:
        enabled_trajectories = list(TRAINING_GENERATORS.keys())

    if len(enabled_trajectories) == 0:
        raise ValueError(
            "enabled_trajectories must contain at least one trajectory type."
        )

    invalid_names = [
        name
        for name in enabled_trajectories
        if name not in TRAINING_GENERATORS
    ]

    if invalid_names:
        raise ValueError(
            f"Unknown trajectory types: {invalid_names}. "
            f"Available: {list(TRAINING_GENERATORS.keys())}"
        )

    selected_name = np.random.choice(enabled_trajectories)

    generator = TRAINING_GENERATORS[selected_name]

    traj = generator()

    if idx is not None:
        traj["name"] = f"{idx:05d}_{traj['name']}"

    return traj


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


def plot_trajectory(traj, output_dir, step=10):
    pos = traj["pos"]
    vel = traj["vel"]
    theta = traj["theta"]

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

    image_filename = os.path.join(output_dir, f"{traj['name']}.png")
    plt.savefig(image_filename, dpi=300, bbox_inches="tight")
    plt.close()


# def generate_dataset(num_trajectories=200, sample=True, save_plots=True):
#     if sample:
#         os.makedirs(OUTPUT_DIR, exist_ok=True)

#     trajectories = []

#     for idx in range(num_trajectories):
#         traj = generate_training_trajectory(idx=idx)
#         trajectories.append(traj)

#         if sample:
#             save_trajectory_csv(traj, OUTPUT_DIR)

#             if save_plots:
#                 plot_trajectory(traj, OUTPUT_DIR)

#     return trajectories

def generate_dataset(
    num_trajectories=200,
    sample=True,
    save_plots=True,
    enabled_trajectories=None,
):
    if sample:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    trajectories = []

    for idx in range(num_trajectories):
        traj = generate_training_trajectory(
            idx=idx,
            enabled_trajectories=enabled_trajectories,
        )

        trajectories.append(traj)

        if sample:
            save_trajectory_csv(
                traj,
                OUTPUT_DIR,
            )

            if save_plots:
                plot_trajectory(
                    traj,
                    OUTPUT_DIR,
                )

    return trajectories


if __name__ == "__main__":
    np.random.seed(42)

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

    generate_dataset(
        num_trajectories=20,
        sample=True,
        save_plots=True,
        enabled_trajectories=training_trajectory_types
    )