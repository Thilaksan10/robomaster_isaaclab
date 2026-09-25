import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import argparse

benchmark_modes = ["policy", "offset"]
# mode = 0

# seed = 24
# randomization = "no_rand"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--controller",
        choices=benchmark_modes,
        required=True,
        help="Controller that was evaluated: policy or offset",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Training seed of the evaluated model",
    )

    parser.add_argument(
        "--training-randomization",
        choices=["rand", "no_rand"],
        required=True,
        help="Whether the evaluated model was trained with domain randomization",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory for this evaluation environment",
    )

    return parser.parse_args()

args = parse_args()

mode = benchmark_modes.index(args.controller)
seed = args.seed
randomization = args.training_randomization

DT = 0.02

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)



folder_path = os.path.join(
    PROJECT_ROOT,
    "record",
    "edge_cases",
)


OUTPUT_DIR = os.path.join(
    args.output_root,
    benchmark_modes[mode],
    randomization,
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


plots_root = os.path.join(
    OUTPUT_DIR,
    "plots",
)

os.makedirs(
    plots_root,
    exist_ok=True,
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise RuntimeError(f"Could not create directory: {path}")


def save_plot(filename):
    ensure_dir(os.path.dirname(filename))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def safe_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(name))


def get_errors(reference, policy, metric="mae"):
    diff = reference - policy

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError("metric must be 'mae' or 'mse'")

    return np.array([
        np.mean(err[:, 0]),
        np.mean(err[:, 1]),
        np.mean(err[:, 2]),
        np.mean(err[:, :2]),
        np.mean(err),
    ])


def get_error_timeseries_3d(reference, policy, metric="mae"):
    diff = reference - policy

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError("metric must be 'mae' or 'mse'")

    return np.mean(err, axis=1)


def get_error_timeseries_all_dims(reference, policy, metric="mae"):
    """
    Computes timestep-wise errors.

    Returns:
        Array with shape (T, 5):
        [x_error, y_error, z_error, 2d_error, 3d_error]

    This keeps the timestep dimension and does not average over time.
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
    err_2d = np.mean(err[:, :2], axis=1)
    err_3d = np.mean(err, axis=1)

    return np.column_stack([
        err_x,
        err_y,
        err_z,
        err_2d,
        err_3d,
    ])


def get_perstep_position_errors(policy_pos, reference_vel, dt, metric="mae"):
    if len(policy_pos) < 2 or len(reference_vel) < 2:
        return np.zeros(5)

    actual_step = policy_pos[1:] - policy_pos[:-1]
    ref_step = reference_vel[:-1] * dt

    diff = ref_step - actual_step

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError("metric must be 'mae' or 'mse'")

    return np.array([
        np.mean(err[:, 0]),
        np.mean(err[:, 1]),
        np.mean(err[:, 2]),
        np.mean(err[:, :2]),
        np.mean(err),
    ])


def get_perstep_position_error_timeseries_3d(
    policy_pos,
    reference_vel,
    dt,
    metric="mae",
):
    if len(policy_pos) < 2 or len(reference_vel) < 2:
        return np.zeros(1)

    actual_step = policy_pos[1:] - policy_pos[:-1]
    ref_step = reference_vel[:-1] * dt

    diff = ref_step - actual_step

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError("metric must be 'mae' or 'mse'")

    return np.mean(err, axis=1)


def compute_mean_and_ci(data, confidence=0.95):
    data = np.asarray(data, dtype=float)

    n = data.shape[0]
    mean = np.mean(data, axis=0)

    if n < 2:
        return mean, np.zeros_like(mean)

    sem = stats.sem(data, axis=0, ddof=1)
    t_crit = stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1)
    ci = sem * t_crit

    return mean, ci


def compute_timestep_mean_and_ci(error_sequences, indices, confidence=0.95):
    """
    Computes timestep-wise mean and confidence interval across selected edge cases.

    Args:
        error_sequences:
            List of arrays. Each array has shape (T, 5).
        indices:
            Indices of selected edge cases.

    Returns:
        mean:
            Array with shape (T_max, 5)
        ci:
            Array with shape (T_max, 5)
        n_valid:
            Number of valid edge cases per timestep, shape (T_max,)
    """
    selected = [
        np.asarray(error_sequences[int(idx)], dtype=float)
        for idx in indices
    ]

    if len(selected) == 0:
        raise ValueError("No timestep error sequences selected.")

    max_len = max(seq.shape[0] for seq in selected)

    padded = np.full(
        (
            len(selected),
            max_len,
            5,
        ),
        np.nan,
        dtype=float,
    )

    for i, seq in enumerate(selected):
        padded[i, :seq.shape[0], :] = seq

    mean = np.nanmean(
        padded,
        axis=0,
    )

    ci = np.zeros_like(mean)

    n_valid = np.sum(
        ~np.isnan(padded[:, :, 0]),
        axis=0,
    ).astype(int)

    for timestep in range(max_len):
        for dim in range(5):
            values = padded[:, timestep, dim]
            values = values[~np.isnan(values)]

            n = len(values)

            if n < 2:
                ci[timestep, dim] = 0.0
            else:
                sem = stats.sem(
                    values,
                    ddof=1,
                )

                t_crit = stats.t.ppf(
                    1.0 - (1.0 - confidence) / 2.0,
                    df=n - 1,
                )

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
            ik_abs,
            confidence=0.95,
        )
        stats_results[metric]["offset_abs_mean"], stats_results[metric]["offset_abs_ci"] = compute_mean_and_ci(
            offset_abs,
            confidence=0.95,
        )

        stats_results[metric]["ik_step_mean"], stats_results[metric]["ik_step_ci"] = compute_mean_and_ci(
            ik_step,
            confidence=0.95,
        )
        stats_results[metric]["offset_step_mean"], stats_results[metric]["offset_step_ci"] = compute_mean_and_ci(
            offset_step,
            confidence=0.95,
        )

        stats_results[metric]["ik_vel_mean"], stats_results[metric]["ik_vel_ci"] = compute_mean_and_ci(
            ik_vel,
            confidence=0.95,
        )
        stats_results[metric]["offset_vel_mean"], stats_results[metric]["offset_vel_ci"] = compute_mean_and_ci(
            offset_vel,
            confidence=0.95,
        )

    return stats_results


def save_benchmark_csv(
    stats_results,
    metrics,
    seed,
    csv_path,
    edgecase_id="all",
    edgecase_name="all",
):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seed",
            "randomization",
            "benchmark_type",
            "edgecase_id",
            "edgecase_name",
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
                    "edgecase",
                    edgecase_id,
                    edgecase_name,
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

    # print(f"Benchmark saved to {csv_path}")


def save_raw_per_edgecase_csv(
    results,
    metrics,
    seed,
    edge_case_names,
    csv_path,
):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seed",
            "randomization",
            "benchmark_type",
            "eval_index",
            "edgecase_id",
            "edgecase_name",
            "metric",
            "evaluation",
            "controller",
            "x",
            "y",
            "z",
            "2d",
            "3d",
        ])

        for metric in metrics:
            for eval_index, edgecase_name in enumerate(edge_case_names):
                edgecase_id = eval_index

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
                        "edgecase",
                        eval_index,
                        edgecase_id,
                        edgecase_name,
                        metric.upper(),
                        evaluation,
                        controller,
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                    ])

    # print(f"Raw per-edgecase benchmark saved to {csv_path}")


def save_timestep_error_csv(
    timestep_results,
    metrics,
    seed,
    edge_case_names,
    csv_path,
):
    """
    Saves timestep-wise errors for:

    - all edge cases together
    - each individual edge case

    Only these evaluations are saved:
    - absolute_position
    - velocity

    Both MAE and MSE are saved.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seed",
            "randomization",
            "benchmark_type",
            "edgecase_id",
            "edgecase_name",
            "timestep",
            "time_s",
            "metric",
            "evaluation",
            "controller",
            "n_edgecases",
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

        all_indices = np.arange(
            len(edge_case_names)
        )

        scopes.append(
            (
                "all",
                "all",
                all_indices,
            )
        )

        for edgecase_id, edgecase_name in enumerate(edge_case_names):
            scopes.append(
                (
                    edgecase_id,
                    edgecase_name,
                    np.array([edgecase_id]),
                )
            )

        for edgecase_id, edgecase_name, indices in tqdm(
            scopes,
            desc="Saving timestep edgecase errors",
        ):
            for metric in metrics:
                entries = [
                    (
                        "absolute_position",
                        "IK",
                        "ik_abs_pos_errors",
                    ),
                    (
                        "absolute_position",
                        benchmark_modes[mode].upper(),
                        "offset_abs_pos_errors",
                    ),
                    (
                        "velocity",
                        "IK",
                        "ik_vel_errors",
                    ),
                    (
                        "velocity",
                        benchmark_modes[mode].upper(),
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
                            "edgecase",
                            edgecase_id,
                            edgecase_name,
                            timestep,
                            timestep * DT,
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

    # print(f"Timestep edgecase errors saved to {csv_path}")


def load_csv_checked(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    data = np.loadtxt(path, delimiter=",")

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    if data.shape[1] < 3:
        raise ValueError(
            f"File {path} has shape {data.shape}, expected at least 3 columns."
        )

    return data[:, :3]


def trim_to_same_length(*arrays):
    min_len = min(len(arr) for arr in arrays)
    return [arr[:min_len] for arr in arrays]


def print_metric_block(title, eval_method, ik_mean, ik_ci, off_mean, off_ci):
    label = benchmark_modes[mode].upper()

    # print(f"{10 * '-'} {title} {10 * '-'}")
    # print(f"IK {eval_method} X:  {ik_mean[0]} ± {ik_ci[0]}")
    # print(f"IK {eval_method} Y:  {ik_mean[1]} ± {ik_ci[1]}")
    # print(f"IK {eval_method} Z:  {ik_mean[2]} ± {ik_ci[2]}")
    # print(f"IK {eval_method} 2D: {ik_mean[3]} ± {ik_ci[3]}")
    # print(f"IK {eval_method} 3D: {ik_mean[4]} ± {ik_ci[4]}")

    # print(f"{label} {eval_method} X:  {off_mean[0]} ± {off_ci[0]}")
    # print(f"{label} {eval_method} Y:  {off_mean[1]} ± {off_ci[1]}")
    # print(f"{label} {eval_method} Z:  {off_mean[2]} ± {off_ci[2]}")
    # print(f"{label} {eval_method} 2D: {off_mean[3]} ± {off_ci[3]}")
    # print(f"{label} {eval_method} 3D: {off_mean[4]} ± {off_ci[4]}")
    # print()


def print_all_statistics(stats_results, metrics, title_prefix="ALL"):
    for metric in metrics:
        eval_method = metric.upper()
        r = stats_results[metric]

        # print_metric_block(
        #     f"{title_prefix} Edge Cases Absolute Position Error ({eval_method})",
        #     eval_method,
        #     r["ik_abs_mean"],
        #     r["ik_abs_ci"],
        #     r["offset_abs_mean"],
        #     r["offset_abs_ci"],
        # )

        # print_metric_block(
        #     f"{title_prefix} Edge Cases Per-Step Position Error ({eval_method})",
        #     eval_method,
        #     r["ik_step_mean"],
        #     r["ik_step_ci"],
        #     r["offset_step_mean"],
        #     r["offset_step_ci"],
        # )

        # print_metric_block(
        #     f"{title_prefix} Edge Cases Velocity Error ({eval_method})",
        #     eval_method,
        #     r["ik_vel_mean"],
        #     r["ik_vel_ci"],
        #     r["offset_vel_mean"],
        #     r["offset_vel_ci"],
        # )


def find_edge_case_directories(folder_path):
    ignored = {"plots"}

    directories = [
        name for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name)) and name not in ignored
    ]

    directories.sort()
    return directories


def plot_overall_bar_3d(ik_mean, ik_ci, off_mean, off_ci, title, ylabel, filename):
    labels = ["3D"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 5))

    plt.bar(
        x - width / 2,
        [ik_mean[4]],
        width,
        yerr=[ik_ci[4]],
        capsize=5,
        label="IK",
    )

    plt.bar(
        x + width / 2,
        [off_mean[4]],
        width,
        yerr=[off_ci[4]],
        capsize=5,
        label=benchmark_modes[mode].upper(),
    )

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    save_plot(filename)


def plot_overall_edgecase_lines_3d(
    edge_case_names,
    ik_errors,
    off_errors,
    title,
    ylabel,
    filename,
):
    x = np.arange(len(edge_case_names))

    plt.figure(figsize=(12, 5))

    plt.plot(
        x,
        ik_errors[:, 4],
        marker="o",
        label="IK",
    )

    plt.plot(
        x,
        off_errors[:, 4],
        marker="x",
        label=benchmark_modes[mode].upper(),
    )

    plt.xticks(x, edge_case_names, rotation=45, ha="right")
    plt.xlabel("Edge Case")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    save_plot(filename)


def plot_single_edgecase_summary_bar_3d(
    ik_values,
    off_values,
    title,
    ylabel,
    filename,
):
    labels = ["3D"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 5))

    plt.bar(
        x - width / 2,
        [ik_values[4]],
        width,
        label="IK",
    )

    plt.bar(
        x + width / 2,
        [off_values[4]],
        width,
        label=benchmark_modes[mode].upper(),
    )

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    save_plot(filename)


def plot_single_edgecase_timeseries_3d(
    ik_ts,
    off_ts,
    title,
    ylabel,
    filename,
):
    min_len = min(len(ik_ts), len(off_ts))
    steps = np.arange(min_len)

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps,
        ik_ts[:min_len],
        label="IK",
    )

    plt.plot(
        steps,
        off_ts[:min_len],
        label=benchmark_modes[mode].upper(),
    )

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    save_plot(filename)


def main():
    ensure_dir(folder_path)
    ensure_dir(plots_root)
    ensure_dir(OUTPUT_DIR)

    # print(f"Current working directory: {os.getcwd()}")
    # print(f"Script directory: {SCRIPT_DIR}")
    # print(f"Project root: {PROJECT_ROOT}")
    # print(f"Edge cases path: {folder_path}")
    # print(f"Plots path: {plots_root}")
    # print(f"Output CSV path: {OUTPUT_DIR}")

    directories = find_edge_case_directories(folder_path)
    n_dirs = len(directories)

    if n_dirs == 0:
        raise RuntimeError(f"No edge case folders found in {folder_path}")

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

    with tqdm(total=n_dirs, desc="Evaluating edge cases") as pbar:
        for i, edge_case_name in enumerate(directories):
            directory = os.path.join(folder_path, edge_case_name)
            edge_plot_dir = os.path.join(plots_root, edge_case_name)
            ensure_dir(edge_plot_dir)

            reference = load_csv_checked(
                os.path.join(directory, "trajectory_reference.csv")
            )
            ik = load_csv_checked(
                os.path.join(directory, "trajectory_ik.csv")
            )
            tested = load_csv_checked(
                os.path.join(directory, f"trajectory_{benchmark_modes[mode]}.csv")
            )

            reference_vel = load_csv_checked(
                os.path.join(directory, "trajectory_reference_vel.csv")
            )
            ik_vel = load_csv_checked(
                os.path.join(directory, "trajectory_ik_vel.csv")
            )
            tested_vel = load_csv_checked(
                os.path.join(directory, f"trajectory_{benchmark_modes[mode]}_vel.csv")
            )

            (
                reference,
                ik,
                tested,
                reference_vel,
                ik_vel,
                tested_vel,
            ) = trim_to_same_length(
                reference,
                ik,
                tested,
                reference_vel,
                ik_vel,
                tested_vel,
            )

            for metric in metrics:
                eval_method = metric.upper()

                ik_abs = get_errors(reference, ik, metric=metric)
                tested_abs = get_errors(reference, tested, metric=metric)

                ik_step = get_perstep_position_errors(
                    ik,
                    reference_vel,
                    DT,
                    metric=metric,
                )
                tested_step = get_perstep_position_errors(
                    tested,
                    reference_vel,
                    DT,
                    metric=metric,
                )

                ik_vel_err = get_errors(reference_vel, ik_vel, metric=metric)
                tested_vel_err = get_errors(reference_vel, tested_vel, metric=metric)

                results[metric]["ik_abs_pos_errors"][i] = ik_abs
                results[metric]["offset_abs_pos_errors"][i] = tested_abs
                results[metric]["ik_perstep_pos_errors"][i] = ik_step
                results[metric]["offset_perstep_pos_errors"][i] = tested_step
                results[metric]["ik_vel_errors"][i] = ik_vel_err
                results[metric]["offset_vel_errors"][i] = tested_vel_err

                timestep_results[metric]["ik_abs_pos_errors"].append(
                    get_error_timeseries_all_dims(
                        reference,
                        ik,
                        metric=metric,
                    )
                )

                timestep_results[metric]["offset_abs_pos_errors"].append(
                    get_error_timeseries_all_dims(
                        reference,
                        tested,
                        metric=metric,
                    )
                )

                timestep_results[metric]["ik_vel_errors"].append(
                    get_error_timeseries_all_dims(
                        reference_vel,
                        ik_vel,
                        metric=metric,
                    )
                )

                timestep_results[metric]["offset_vel_errors"].append(
                    get_error_timeseries_all_dims(
                        reference_vel,
                        tested_vel,
                        metric=metric,
                    )
                )

                plot_single_edgecase_summary_bar_3d(
                    ik_abs,
                    tested_abs,
                    title=f"{edge_case_name} Absolute Position Error 3D {eval_method}",
                    ylabel=eval_method,
                    filename=os.path.join(
                        edge_plot_dir,
                        f"{edge_case_name}_absolute_position_summary_3d_{metric}.png",
                    ),
                )

                plot_single_edgecase_summary_bar_3d(
                    ik_step,
                    tested_step,
                    title=f"{edge_case_name} Per-Step Position Error 3D {eval_method}",
                    ylabel=eval_method,
                    filename=os.path.join(
                        edge_plot_dir,
                        f"{edge_case_name}_perstep_position_summary_3d_{metric}.png",
                    ),
                )

                plot_single_edgecase_summary_bar_3d(
                    ik_vel_err,
                    tested_vel_err,
                    title=f"{edge_case_name} Velocity Error 3D {eval_method}",
                    ylabel=eval_method,
                    filename=os.path.join(
                        edge_plot_dir,
                        f"{edge_case_name}_velocity_summary_3d_{metric}.png",
                    ),
                )

                ik_abs_ts = get_error_timeseries_3d(reference, ik, metric=metric)
                tested_abs_ts = get_error_timeseries_3d(reference, tested, metric=metric)

                ik_step_ts = get_perstep_position_error_timeseries_3d(
                    ik,
                    reference_vel,
                    DT,
                    metric=metric,
                )
                tested_step_ts = get_perstep_position_error_timeseries_3d(
                    tested,
                    reference_vel,
                    DT,
                    metric=metric,
                )

                ik_vel_ts = get_error_timeseries_3d(
                    reference_vel,
                    ik_vel,
                    metric=metric,
                )
                tested_vel_ts = get_error_timeseries_3d(
                    reference_vel,
                    tested_vel,
                    metric=metric,
                )

                plot_single_edgecase_timeseries_3d(
                    ik_abs_ts,
                    tested_abs_ts,
                    title=f"{edge_case_name} Absolute Position Error 3D {eval_method}",
                    ylabel=eval_method,
                    filename=os.path.join(
                        edge_plot_dir,
                        f"{edge_case_name}_absolute_position_timeseries_3d_{metric}.png",
                    ),
                )

                plot_single_edgecase_timeseries_3d(
                    ik_step_ts,
                    tested_step_ts,
                    title=f"{edge_case_name} Per-Step Position Error 3D {eval_method}",
                    ylabel=eval_method,
                    filename=os.path.join(
                        edge_plot_dir,
                        f"{edge_case_name}_perstep_position_timeseries_3d_{metric}.png",
                    ),
                )

                plot_single_edgecase_timeseries_3d(
                    ik_vel_ts,
                    tested_vel_ts,
                    title=f"{edge_case_name} Velocity Error 3D {eval_method}",
                    ylabel=eval_method,
                    filename=os.path.join(
                        edge_plot_dir,
                        f"{edge_case_name}_velocity_timeseries_3d_{metric}.png",
                    ),
                )

            pbar.update(1)

    # print(f"\nUsing DT = {DT}")
    # print(f"Seed = {seed}")
    # print(f"Randomization = {randomization}")
    # print(f"Benchmark mode = {benchmark_modes[mode].upper()}")
    # print(f"Number of evaluated edge cases = {n_dirs}")
    # print(f"Folder path: {folder_path}")
    # print(f"Plots path: {plots_root}")
    # print(f"Output CSV path: {OUTPUT_DIR}\n")

    all_stats = compute_statistics(results, metrics)

    save_benchmark_csv(
        all_stats,
        metrics,
        seed,
        csv_path=os.path.join(OUTPUT_DIR, f"benchmark_{seed}.csv"),
        edgecase_id="all",
        edgecase_name="all",
    )

    save_raw_per_edgecase_csv(
        results,
        metrics,
        seed,
        directories,
        csv_path=os.path.join(OUTPUT_DIR, f"benchmark_{seed}_raw_per_edgecase.csv"),
    )

    save_timestep_error_csv(
        timestep_results,
        metrics,
        seed,
        directories,
        csv_path=os.path.join(
            OUTPUT_DIR,
            f"benchmark_{seed}_timestep_errors.csv",
        ),
    )

    for edgecase_id, edgecase_name in enumerate(directories):
        edgecase_stats = compute_statistics(
            results,
            metrics,
            indices=np.array([edgecase_id]),
        )

        save_benchmark_csv(
            edgecase_stats,
            metrics,
            seed,
            csv_path=os.path.join(
                OUTPUT_DIR,
                f"benchmark_{seed}_edgecase_{safe_filename(edgecase_name)}.csv",
            ),
            edgecase_id=edgecase_id,
            edgecase_name=edgecase_name,
        )

    print_all_statistics(all_stats, metrics, title_prefix="ALL")

    for edgecase_id, edgecase_name in enumerate(directories):
        edgecase_stats = compute_statistics(
            results,
            metrics,
            indices=np.array([edgecase_id]),
        )

        print_all_statistics(
            edgecase_stats,
            metrics,
            title_prefix=f"EDGECASE {edgecase_id}: {edgecase_name}",
        )

    for metric in metrics:
        eval_method = metric.upper()
        r = all_stats[metric]

        plot_overall_bar_3d(
            r["ik_abs_mean"],
            r["ik_abs_ci"],
            r["offset_abs_mean"],
            r["offset_abs_ci"],
            title=f"Overall Edge Cases Absolute Position Error 3D {eval_method}",
            ylabel=eval_method,
            filename=os.path.join(
                plots_root,
                f"overall_absolute_position_3d_{metric}.png",
            ),
        )

        plot_overall_bar_3d(
            r["ik_step_mean"],
            r["ik_step_ci"],
            r["offset_step_mean"],
            r["offset_step_ci"],
            title=f"Overall Edge Cases Per-Step Position Error 3D {eval_method}",
            ylabel=eval_method,
            filename=os.path.join(
                plots_root,
                f"overall_perstep_position_3d_{metric}.png",
            ),
        )

        plot_overall_bar_3d(
            r["ik_vel_mean"],
            r["ik_vel_ci"],
            r["offset_vel_mean"],
            r["offset_vel_ci"],
            title=f"Overall Edge Cases Velocity Error 3D {eval_method}",
            ylabel=eval_method,
            filename=os.path.join(
                plots_root,
                f"overall_velocity_3d_{metric}.png",
            ),
        )

        plot_overall_edgecase_lines_3d(
            directories,
            results[metric]["ik_abs_pos_errors"],
            results[metric]["offset_abs_pos_errors"],
            title=f"Edge Case Absolute Position Error 3D {eval_method}",
            ylabel=eval_method,
            filename=os.path.join(
                plots_root,
                f"edgecase_absolute_position_3d_{metric}.png",
            ),
        )

        plot_overall_edgecase_lines_3d(
            directories,
            results[metric]["ik_perstep_pos_errors"],
            results[metric]["offset_perstep_pos_errors"],
            title=f"Edge Case Per-Step Position Error 3D {eval_method}",
            ylabel=eval_method,
            filename=os.path.join(
                plots_root,
                f"edgecase_perstep_position_3d_{metric}.png",
            ),
        )

        plot_overall_edgecase_lines_3d(
            directories,
            results[metric]["ik_vel_errors"],
            results[metric]["offset_vel_errors"],
            title=f"Edge Case Velocity Error 3D {eval_method}",
            ylabel=eval_method,
            filename=os.path.join(
                plots_root,
                f"edgecase_velocity_3d_{metric}.png",
            ),
        )


if __name__ == "__main__":
    main()