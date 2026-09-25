import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import argparse


benchmark_modes = [
    "policy",
    "offset",
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--controller",
        choices=benchmark_modes,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--training-randomization",
        choices=[
            "rand",
            "no_rand",
        ],
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--folder-path",
        type=str,
        default="records_multiple_4000_small_new",
    )

    parser.add_argument(
        "--num-racetracks",
        type=int,
        default=4,
    )

    return parser.parse_args()


args = parse_args()


mode = benchmark_modes.index(
    args.controller
)

seed = args.seed

randomization = (
    args.training_randomization
)

num_racetracks = args.num_racetracks

DT = 0.02

folder_path = args.folder_path


OUTPUT_DIR = os.path.join(
    args.output_root,
    benchmark_modes[mode],
    randomization,
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


def get_errors(reference, policy, metric="mae"):
    diff = reference - policy

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError("metric must be 'mae' or 'mse'")

    err_x = np.mean(err[:, 0])
    err_y = np.mean(err[:, 1])
    err_z = np.mean(err[:, 2])
    err_xy = np.mean(err[:, :2])
    err_xyz = np.mean(err)

    return err_x, err_y, err_z, err_xy, err_xyz


def get_perstep_position_errors(policy_pos, reference_vel, dt, metric="mae"):
    if len(policy_pos) < 2 or len(reference_vel) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    actual_step = policy_pos[1:] - policy_pos[:-1]
    ref_step = reference_vel[:-1] * dt

    diff = ref_step - actual_step

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError("metric must be 'mae' or 'mse'")

    err_x = np.mean(err[:, 0])
    err_y = np.mean(err[:, 1])
    err_z = np.mean(err[:, 2])
    err_xy = np.mean(err[:, :2])
    err_xyz = np.mean(err)

    return err_x, err_y, err_z, err_xy, err_xyz


def get_timestep_errors(reference, policy, metric="mae"):
    """
    Computes timestep-wise errors.

    Returns:
        Array with shape (T, 5):
        [x_error, y_error, z_error, 2d_error, 3d_error]

    This function does not average over time.
    It keeps one error row per timestep.
    """
    diff = reference - policy

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError("metric must be 'mae' or 'mse'")

    err_x = err[:, 0]
    err_y = err[:, 1]
    err_z = err[:, 2]
    err_xy = np.mean(err[:, :2], axis=1)
    err_xyz = np.mean(err, axis=1)

    return np.column_stack([err_x, err_y, err_z, err_xy, err_xyz])


def compute_mean_and_ci(data, confidence=0.95):
    n = data.shape[0]
    mean = np.mean(data, axis=0)

    if n < 2:
        return mean, np.zeros_like(mean)

    sem = stats.sem(data, axis=0, ddof=1)
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    ci = sem * t_crit

    return mean, ci


def compute_timestep_mean_and_ci(error_sequences, indices, confidence=0.95):
    """
    Computes mean and confidence interval per timestep across selected evaluations.

    Args:
        error_sequences:
            List of arrays. Each array has shape (T, 5).
        indices:
            Evaluation indices that should be averaged.
        confidence:
            Confidence interval level.

    Returns:
        mean:
            Array with shape (T_max, 5)
        ci:
            Array with shape (T_max, 5)
        n_valid:
            Array with shape (T_max,), number of evaluations available per timestep
    """
    selected = [np.asarray(error_sequences[int(idx)], dtype=float) for idx in indices]

    if len(selected) == 0:
        raise ValueError("No timestep error sequences selected.")

    max_len = max(seq.shape[0] for seq in selected)
    padded = np.full((len(selected), max_len, 5), np.nan, dtype=float)

    for i, seq in enumerate(selected):
        padded[i, :seq.shape[0], :] = seq

    mean = np.nanmean(padded, axis=0)
    ci = np.zeros_like(mean)

    n_valid = np.sum(~np.isnan(padded[:, :, 0]), axis=0).astype(int)

    for timestep in range(max_len):
        for dim in range(5):
            values = padded[:, timestep, dim]
            values = values[~np.isnan(values)]

            n = len(values)

            if n < 2:
                ci[timestep, dim] = 0.0
            else:
                sem = stats.sem(values, ddof=1)
                alpha = 1.0 - confidence
                t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
                ci[timestep, dim] = sem * t_crit

    return mean, ci, n_valid


def compute_statistics(results, metrics, indices=None):
    stats_results = {}

    for metric in metrics:
        r = results[metric]

        if indices is None:
            ik_abs = r["ik_abs_pos_errors"]
            offset_abs = r["offset_abs_pos_errors"]
            ik_step = r["ik_perstep_pos_errors"]
            offset_step = r["offset_perstep_pos_errors"]
            ik_vel = r["ik_vel_errors"]
            offset_vel = r["offset_vel_errors"]
        else:
            ik_abs = r["ik_abs_pos_errors"][indices]
            offset_abs = r["offset_abs_pos_errors"][indices]
            ik_step = r["ik_perstep_pos_errors"][indices]
            offset_step = r["offset_perstep_pos_errors"][indices]
            ik_vel = r["ik_vel_errors"][indices]
            offset_vel = r["offset_vel_errors"][indices]

        stats_results[metric] = {}

        stats_results[metric]["ik_abs_mean"], stats_results[metric]["ik_abs_ci"] = compute_mean_and_ci(
            ik_abs
        )
        stats_results[metric]["offset_abs_mean"], stats_results[metric]["offset_abs_ci"] = compute_mean_and_ci(
            offset_abs
        )

        stats_results[metric]["ik_step_mean"], stats_results[metric]["ik_step_ci"] = compute_mean_and_ci(
            ik_step
        )
        stats_results[metric]["offset_step_mean"], stats_results[metric]["offset_step_ci"] = compute_mean_and_ci(
            offset_step
        )

        stats_results[metric]["ik_vel_mean"], stats_results[metric]["ik_vel_ci"] = compute_mean_and_ci(
            ik_vel
        )
        stats_results[metric]["offset_vel_mean"], stats_results[metric]["offset_vel_ci"] = compute_mean_and_ci(
            offset_vel
        )

    return stats_results


def load_csv_checked(path):
    data = np.loadtxt(path, delimiter=",")

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    if data.shape[1] < 3:
        raise ValueError(f"File {path} has shape {data.shape}, expected at least 3 columns.")

    return data[:, :3]


def trim_to_same_length(*arrays):
    min_len = min(len(arr) for arr in arrays)
    return [arr[:min_len] for arr in arrays]


def print_metric_block(title, eval_method, ik_mean, ik_ci, off_mean, off_ci):
    print(f"{10 * '-'} {title} {10 * '-'}")
    print(f"IK {eval_method} X:  {ik_mean[0]} ± {ik_ci[0]}")
    print(f"IK {eval_method} Y:  {ik_mean[1]} ± {ik_ci[1]}")
    print(f"IK {eval_method} Z:  {ik_mean[2]} ± {ik_ci[2]}")
    print(f"IK {eval_method} 2D: {ik_mean[3]} ± {ik_ci[3]}")
    print(f"IK {eval_method} 3D: {ik_mean[4]} ± {ik_ci[4]}")

    print(f"{benchmark_modes[mode].upper()} {eval_method} X:  {off_mean[0]} ± {off_ci[0]}")
    print(f"{benchmark_modes[mode].upper()} {eval_method} Y:  {off_mean[1]} ± {off_ci[1]}")
    print(f"{benchmark_modes[mode].upper()} {eval_method} Z:  {off_mean[2]} ± {off_ci[2]}")
    print(f"{benchmark_modes[mode].upper()} {eval_method} 2D: {off_mean[3]} ± {off_ci[3]}")
    print(f"{benchmark_modes[mode].upper()} {eval_method} 3D: {off_mean[4]} ± {off_ci[4]}")
    print()


def print_all_statistics(stats_results, metrics, title_prefix="ALL"):
    for metric in metrics:
        eval_method = metric.upper()
        r = stats_results[metric]

        print_metric_block(
            f"{title_prefix} OMNI Absolute Position Error ({eval_method})",
            eval_method,
            r["ik_abs_mean"],
            r["ik_abs_ci"],
            r["offset_abs_mean"],
            r["offset_abs_ci"],
        )

        print_metric_block(
            f"{title_prefix} OMNI Per-Step Position Error ({eval_method})",
            eval_method,
            r["ik_step_mean"],
            r["ik_step_ci"],
            r["offset_step_mean"],
            r["offset_step_ci"],
        )

        print_metric_block(
            f"{title_prefix} OMNI Velocity Error ({eval_method})",
            eval_method,
            r["ik_vel_mean"],
            r["ik_vel_ci"],
            r["offset_vel_mean"],
            r["offset_vel_ci"],
        )


def plot_single_metric_bar(
    ik_mean,
    ik_ci,
    off_mean,
    off_ci,
    eval_method,
    metric_name,
    filename,
):
    labels = ["3D"]

    ik_values = [ik_mean[4]]
    off_values = [off_mean[4]]

    ik_ci_values = [ik_ci[4]]
    off_ci_values = [off_ci[4]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 5))

    plt.bar(
        x - width / 2,
        ik_values,
        width,
        yerr=ik_ci_values,
        capsize=5,
        label="IK",
    )

    plt.bar(
        x + width / 2,
        off_values,
        width,
        yerr=off_ci_values,
        capsize=5,
        label=benchmark_modes[mode].upper(),
    )

    plt.xticks(x, labels)
    plt.ylabel(eval_method)
    plt.title(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_single_metric_lines(
    ik_errors,
    off_errors,
    eval_method,
    metric_name,
    filename,
):
    traj_idx = np.arange(ik_errors.shape[0])

    plt.figure(figsize=(10, 5))

    plt.plot(
        traj_idx,
        ik_errors[:, 4],
        marker="o",
        label="IK",
    )

    plt.plot(
        traj_idx,
        off_errors[:, 4],
        marker="x",
        label=benchmark_modes[mode].upper(),
    )

    plt.xlabel("Trajectory Index")
    plt.ylabel(eval_method)
    plt.title(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_benchmark_csv(stats_results, metrics, seed, csv_path, racetrack_id="all"):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seed",
            "randomization",
            "racetrack_id",
            "metric",
            "evaluation",
            "controller",
            "x_mean",
            "y_mean",
            "z_mean",
            "2d_mean",
            "3d_mean",
            "x_ci",
            "y_ci",
            "z_ci",
            "2d_ci",
            "3d_ci",
        ])

        for metric in metrics:
            r = stats_results[metric]

            entries = [
                (
                    "absolute_position",
                    "IK",
                    r["ik_abs_mean"],
                    r["ik_abs_ci"],
                ),
                (
                    "absolute_position",
                    benchmark_modes[mode].upper(),
                    r["offset_abs_mean"],
                    r["offset_abs_ci"],
                ),
                (
                    "perstep_position",
                    "IK",
                    r["ik_step_mean"],
                    r["ik_step_ci"],
                ),
                (
                    "perstep_position",
                    benchmark_modes[mode].upper(),
                    r["offset_step_mean"],
                    r["offset_step_ci"],
                ),
                (
                    "velocity",
                    "IK",
                    r["ik_vel_mean"],
                    r["ik_vel_ci"],
                ),
                (
                    "velocity",
                    benchmark_modes[mode].upper(),
                    r["offset_vel_mean"],
                    r["offset_vel_ci"],
                ),
            ]

            for evaluation, controller, mean, ci in entries:
                writer.writerow([
                    seed,
                    randomization,
                    racetrack_id,
                    metric.upper(),
                    evaluation,
                    controller,
                    mean[0],
                    mean[1],
                    mean[2],
                    mean[3],
                    mean[4],
                    ci[0],
                    ci[1],
                    ci[2],
                    ci[3],
                    ci[4],
                ])

    print(f"Benchmark saved to {csv_path}")


def save_raw_per_evaluation_csv(results, metrics, seed, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seed",
            "randomization",
            "eval_index",
            "racetrack_id",
            "metric",
            "evaluation",
            "controller",
            "x",
            "y",
            "z",
            "2d",
            "3d",
        ])

        evals_per_racetrack = n_dirs // num_racetracks

        for metric in metrics:
            for eval_index in range(n_dirs):
                racetrack_id = eval_index // evals_per_racetrack

                rows = [
                    (
                        "absolute_position",
                        "IK",
                        results[metric]["ik_abs_pos_errors"][eval_index],
                    ),
                    (
                        "absolute_position",
                        benchmark_modes[mode].upper(),
                        results[metric]["offset_abs_pos_errors"][eval_index],
                    ),
                    (
                        "perstep_position",
                        "IK",
                        results[metric]["ik_perstep_pos_errors"][eval_index],
                    ),
                    (
                        "perstep_position",
                        benchmark_modes[mode].upper(),
                        results[metric]["offset_perstep_pos_errors"][eval_index],
                    ),
                    (
                        "velocity",
                        "IK",
                        results[metric]["ik_vel_errors"][eval_index],
                    ),
                    (
                        "velocity",
                        benchmark_modes[mode].upper(),
                        results[metric]["offset_vel_errors"][eval_index],
                    ),
                ]

                for evaluation, controller, values in rows:
                    writer.writerow([
                        seed,
                        randomization,
                        eval_index,
                        racetrack_id,
                        metric.upper(),
                        evaluation,
                        controller,
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                    ])

    print(f"Raw per-evaluation benchmark saved to {csv_path}")


def save_timestep_error_csv(
    timestep_results,
    metrics,
    seed,
    csv_path,
    randomization,
    learned_controller_label,
    num_racetracks,
    evals_per_racetrack,
    n_dirs,
    dt,
):
    """
    Saves timestep-wise averaged error to a separate CSV.

    The CSV contains:
    - racetrack_id = "all" for the average over all racetracks
    - racetrack_id = 0, 1, 2, ... for the average per racetrack

    Only these evaluations are saved:
    - absolute_position
    - velocity

    Per-step position error is intentionally not saved here.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seed",
            "randomization",
            "racetrack_id",
            "timestep",
            "time_s",
            "metric",
            "evaluation",
            "controller",
            "n_evaluations",
            "x_mean",
            "y_mean",
            "z_mean",
            "2d_mean",
            "3d_mean",
            "x_ci",
            "y_ci",
            "z_ci",
            "2d_ci",
            "3d_ci",
        ])

        scopes = []

        all_indices = np.arange(n_dirs)
        scopes.append(("all", all_indices))

        for racetrack_id in range(num_racetracks):
            start_idx = racetrack_id * evals_per_racetrack
            end_idx = start_idx + evals_per_racetrack
            indices = np.arange(start_idx, end_idx)
            scopes.append((racetrack_id, indices))

        for racetrack_id, indices in tqdm(scopes, desc="Saving timestep error CSV"):
            for metric in metrics:
                entries = [
                    (
                        "absolute_position",
                        "IK",
                        "ik_abs_pos_errors",
                    ),
                    (
                        "absolute_position",
                        learned_controller_label,
                        "offset_abs_pos_errors",
                    ),
                    (
                        "velocity",
                        "IK",
                        "ik_vel_errors",
                    ),
                    (
                        "velocity",
                        learned_controller_label,
                        "offset_vel_errors",
                    ),
                ]

                for evaluation, controller, key in entries:
                    mean, ci, n_valid = compute_timestep_mean_and_ci(
                        timestep_results[metric][key],
                        indices,
                    )

                    for timestep in range(mean.shape[0]):
                        writer.writerow([
                            seed,
                            randomization,
                            racetrack_id,
                            timestep,
                            timestep * dt,
                            metric.upper(),
                            evaluation,
                            controller,
                            n_valid[timestep],
                            mean[timestep, 0],
                            mean[timestep, 1],
                            mean[timestep, 2],
                            mean[timestep, 3],
                            mean[timestep, 4],
                            ci[timestep, 0],
                            ci[timestep, 1],
                            ci[timestep, 2],
                            ci[timestep, 3],
                            ci[timestep, 4],
                        ])

    print(f"Timestep-wise error benchmark saved to {csv_path}")


directories = [
    name for name in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, name))
]

directories.sort(key=lambda x: int(x.split("_")[1]))

n_dirs = len(directories)

if n_dirs % num_racetracks != 0:
    raise ValueError(
        f"Number of evaluations ({n_dirs}) must be divisible by num_racetracks ({num_racetracks})."
    )

evals_per_racetrack = n_dirs // num_racetracks

metrics = ["mae", "mse"]

results = {}

for metric in metrics:
    results[metric] = {
        "ik_abs_pos_errors": np.zeros((n_dirs, 5)),
        "offset_abs_pos_errors": np.zeros((n_dirs, 5)),
        "ik_perstep_pos_errors": np.zeros((n_dirs, 5)),
        "offset_perstep_pos_errors": np.zeros((n_dirs, 5)),
        "ik_vel_errors": np.zeros((n_dirs, 5)),
        "offset_vel_errors": np.zeros((n_dirs, 5)),
    }

timestep_results = {}

for metric in metrics:
    timestep_results[metric] = {
        "ik_abs_pos_errors": [],
        "offset_abs_pos_errors": [],
        "ik_vel_errors": [],
        "offset_vel_errors": [],
    }


with tqdm(total=n_dirs) as pbar:
    for i, directory_name in enumerate(directories):
        directory = os.path.join(folder_path, directory_name)

        reference_omni = load_csv_checked(f"{directory}/trajectory_reference_omni.csv")
        ik_omni = load_csv_checked(f"{directory}/trajectory_ik_omni.csv")
        offset_omni = load_csv_checked(
            f"{directory}/trajectory_{benchmark_modes[mode]}_omni.csv"
        )

        reference_omni_vel = load_csv_checked(
            f"{directory}/trajectory_reference_omni_vel.csv"
        )
        ik_omni_vel = load_csv_checked(f"{directory}/trajectory_ik_omni_vel.csv")
        offset_omni_vel = load_csv_checked(
            f"{directory}/trajectory_{benchmark_modes[mode]}_omni_vel.csv"
        )

        (
            reference_omni,
            ik_omni,
            offset_omni,
            reference_omni_vel,
            ik_omni_vel,
            offset_omni_vel,
        ) = trim_to_same_length(
            reference_omni,
            ik_omni,
            offset_omni,
            reference_omni_vel,
            ik_omni_vel,
            offset_omni_vel,
        )

        for metric in metrics:
            results[metric]["ik_abs_pos_errors"][i] = get_errors(
                reference_omni,
                ik_omni,
                metric=metric,
            )

            results[metric]["offset_abs_pos_errors"][i] = get_errors(
                reference_omni,
                offset_omni,
                metric=metric,
            )

            results[metric]["ik_perstep_pos_errors"][i] = get_perstep_position_errors(
                ik_omni,
                reference_omni_vel,
                DT,
                metric=metric,
            )

            results[metric]["offset_perstep_pos_errors"][i] = get_perstep_position_errors(
                offset_omni,
                reference_omni_vel,
                DT,
                metric=metric,
            )

            results[metric]["ik_vel_errors"][i] = get_errors(
                reference_omni_vel,
                ik_omni_vel,
                metric=metric,
            )

            results[metric]["offset_vel_errors"][i] = get_errors(
                reference_omni_vel,
                offset_omni_vel,
                metric=metric,
            )

            timestep_results[metric]["ik_abs_pos_errors"].append(
                get_timestep_errors(
                    reference_omni,
                    ik_omni,
                    metric=metric,
                )
            )

            timestep_results[metric]["offset_abs_pos_errors"].append(
                get_timestep_errors(
                    reference_omni,
                    offset_omni,
                    metric=metric,
                )
            )

            timestep_results[metric]["ik_vel_errors"].append(
                get_timestep_errors(
                    reference_omni_vel,
                    ik_omni_vel,
                    metric=metric,
                )
            )

            timestep_results[metric]["offset_vel_errors"].append(
                get_timestep_errors(
                    reference_omni_vel,
                    offset_omni_vel,
                    metric=metric,
                )
            )

        pbar.update(1)


print(f"\nUsing DT = {DT}")
print(f"Seed = {seed}")
print(f"Randomization = {randomization}")
print(f"Benchmark mode = {benchmark_modes[mode].upper()}")
print(f"Number of evaluations = {n_dirs}")
print(f"Number of racetracks = {num_racetracks}")
print(f"Evaluations per racetrack = {evals_per_racetrack}\n")


all_stats = compute_statistics(results, metrics)


save_benchmark_csv(
    all_stats,
    metrics,
    seed,
    csv_path=os.path.join(OUTPUT_DIR, f"benchmark_{seed}.csv"),
    racetrack_id="all",
)


save_raw_per_evaluation_csv(
    results,
    metrics,
    seed,
    csv_path=os.path.join(OUTPUT_DIR, f"benchmark_{seed}_raw_per_evaluation.csv"),
)


save_timestep_error_csv(
    timestep_results,
    metrics,
    seed,
    csv_path=os.path.join(OUTPUT_DIR, f"benchmark_{seed}_timestep_errors.csv"),
    randomization=randomization,
    learned_controller_label=benchmark_modes[mode].upper(),
    num_racetracks=num_racetracks,
    evals_per_racetrack=evals_per_racetrack,
    n_dirs=n_dirs,
    dt=DT,
)


for racetrack_id in range(num_racetracks):
    start_idx = racetrack_id * evals_per_racetrack
    end_idx = start_idx + evals_per_racetrack
    indices = np.arange(start_idx, end_idx)

    racetrack_stats = compute_statistics(results, metrics, indices=indices)

    print_all_statistics(
        racetrack_stats,
        metrics,
        title_prefix=f"RACETRACK {racetrack_id}",
    )

    save_benchmark_csv(
        racetrack_stats,
        metrics,
        seed,
        csv_path=os.path.join(
            OUTPUT_DIR,
            f"benchmark_{seed}_racetrack_{racetrack_id}.csv",
        ),
        racetrack_id=racetrack_id,
    )


print_all_statistics(all_stats, metrics, title_prefix="ALL")


for metric in metrics:
    eval_method = metric.upper()
    r = all_stats[metric]

    plot_single_metric_bar(
        r["ik_abs_mean"],
        r["ik_abs_ci"],
        r["offset_abs_mean"],
        r["offset_abs_ci"],
        eval_method=eval_method,
        metric_name=f"Omni Absolute Position Error 3D {eval_method}",
        filename=f"bar_omni_absolute_position_{metric}.png",
    )

    plot_single_metric_bar(
        r["ik_step_mean"],
        r["ik_step_ci"],
        r["offset_step_mean"],
        r["offset_step_ci"],
        eval_method=eval_method,
        metric_name=f"Omni Per-Step Position Error 3D {eval_method}",
        filename=f"bar_omni_perstep_position_{metric}.png",
    )

    plot_single_metric_bar(
        r["ik_vel_mean"],
        r["ik_vel_ci"],
        r["offset_vel_mean"],
        r["offset_vel_ci"],
        eval_method=eval_method,
        metric_name=f"Omni Velocity Error 3D {eval_method}",
        filename=f"bar_omni_velocity_{metric}.png",
    )

    plot_single_metric_lines(
        results[metric]["ik_abs_pos_errors"],
        results[metric]["offset_abs_pos_errors"],
        eval_method=eval_method,
        metric_name=f"Omni Absolute Position Error 3D {eval_method}",
        filename=f"line_omni_absolute_position_{metric}.png",
    )

    plot_single_metric_lines(
        results[metric]["ik_perstep_pos_errors"],
        results[metric]["offset_perstep_pos_errors"],
        eval_method=eval_method,
        metric_name=f"Omni Per-Step Position Error 3D {eval_method}",
        filename=f"line_omni_perstep_position_{metric}.png",
    )

    plot_single_metric_lines(
        results[metric]["ik_vel_errors"],
        results[metric]["offset_vel_errors"],
        eval_method=eval_method,
        metric_name=f"Omni Velocity Error 3D {eval_method}",
        filename=f"line_omni_velocity_{metric}.png",
    )