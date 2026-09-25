import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats


# ============================================================
# Configuration
# ============================================================

RECORDING_FOLDERS = [
    "records_multiple_1000_validation",
    "records_multiple_1000_validation_big",
]

OUTPUT_ROOT = "preliminary"

SEED = None
TRAINING_RANDOMIZATION = None
MODEL = None

CONTROLLERS = [
    "ik",
    "offset",
    "policy",
]

CONTROLLER_LABELS = {
    "ik": "IK",
    "offset": "Hybrid Offset Policy",
    "policy": "Fully Learned Policy",
}

METRICS = ["mae", "mse"]

DT = 0.02


# ============================================================
# Error functions
# ============================================================

def get_errors(reference, controller, metric="mae"):
    diff = reference - controller

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


def get_timestep_errors(reference, controller, metric="mae"):
    """
    Return error for every timestep.

    Shape:
        (T, 5)

    Columns:
        x, y, z, 2D, 3D
    """

    diff = reference - controller

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError(
            "metric must be 'mae' or 'mse'"
        )

    err_x = err[:, 0]
    err_y = err[:, 1]
    err_z = err[:, 2]

    err_xy = np.mean(
        err[:, :2],
        axis=1,
    )

    err_xyz = np.mean(
        err,
        axis=1,
    )

    return np.column_stack([
        err_x,
        err_y,
        err_z,
        err_xy,
        err_xyz,
    ])


def compute_mean_and_ci(data, confidence=0.95):
    n = data.shape[0]
    mean = np.mean(data, axis=0)

    if n < 2:
        return mean, np.zeros_like(mean)

    sem = stats.sem(data, axis=0, ddof=1)
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(
        1.0 - alpha / 2.0,
        df=n - 1,
    )
    ci = sem * t_crit

    return mean, ci

def compute_timestep_mean_and_ci(
    error_sequences,
    confidence=0.95,
):
    """
    Average timestep errors across all evaluations.

    error_sequences:
        list of arrays with shape (T, 5)

    Returns:
        mean: (T_max, 5)
        ci:   (T_max, 5)
    """

    max_len = max(
        sequence.shape[0]
        for sequence in error_sequences
    )

    padded = np.full(
        (
            len(error_sequences),
            max_len,
            5,
        ),
        np.nan,
        dtype=float,
    )

    for i, sequence in enumerate(
        error_sequences
    ):
        padded[
            i,
            :sequence.shape[0],
            :
        ] = sequence

    mean = np.nanmean(
        padded,
        axis=0,
    )

    ci = np.zeros_like(mean)

    for timestep in range(max_len):

        for dim in range(5):

            values = padded[
                :,
                timestep,
                dim,
            ]

            values = values[
                ~np.isnan(values)
            ]

            n = len(values)

            if n < 2:
                ci[timestep, dim] = 0.0
                continue

            sem = stats.sem(
                values,
                ddof=1,
            )

            alpha = 1.0 - confidence

            t_crit = stats.t.ppf(
                1.0 - alpha / 2.0,
                df=n - 1,
            )

            ci[timestep, dim] = (
                sem * t_crit
            )

    return mean, ci


# ============================================================
# Loading helpers
# ============================================================

def load_csv_checked(path):
    data = np.loadtxt(
        path,
        delimiter=",",
    )

    if data.ndim == 1:
        data = np.expand_dims(
            data,
            axis=0,
        )

    if data.shape[1] < 3:
        raise ValueError(
            f"File {path} has shape {data.shape}, "
            "expected at least 3 columns."
        )

    return data[:, :3]


def trim_to_same_length(*arrays):
    min_len = min(
        len(arr)
        for arr in arrays
    )

    return [
        arr[:min_len]
        for arr in arrays
    ]


def directory_sort_key(name):
    """
    Keeps the same behavior as the existing evaluation benchmark
    for folders such as eval_0, eval_1, trajectory_0, ...

    Falls back to alphabetical sorting if no numeric suffix exists.
    """
    try:
        return (
            0,
            int(name.split("_")[-1]),
        )
    except ValueError:
        return (1, name)


# ============================================================
# Result storage
# ============================================================

def initialize_results(n_dirs):
    results = {}

    for metric in METRICS:
        results[metric] = {}

        for controller in CONTROLLERS:
            results[metric][
                f"{controller}_abs_pos_errors"
            ] = np.zeros(
                (n_dirs, 5)
            )

            results[metric][
                f"{controller}_vel_errors"
            ] = np.zeros(
                (n_dirs, 5)
            )

    return results


# ============================================================
# Statistics
# ============================================================

def compute_statistics(results):
    statistics = {}

    for metric in METRICS:
        statistics[metric] = {}

        for controller in CONTROLLERS:
            abs_key = (
                f"{controller}_abs_pos_errors"
            )
            vel_key = (
                f"{controller}_vel_errors"
            )

            (
                statistics[metric][
                    f"{controller}_abs_mean"
                ],
                statistics[metric][
                    f"{controller}_abs_ci"
                ],
            ) = compute_mean_and_ci(
                results[metric][abs_key]
            )

            (
                statistics[metric][
                    f"{controller}_vel_mean"
                ],
                statistics[metric][
                    f"{controller}_vel_ci"
                ],
            ) = compute_mean_and_ci(
                results[metric][vel_key]
            )

    return statistics


# ============================================================
# Printing
# ============================================================

def print_statistics(
    stats_results,
    folder_name,
):
    print()
    print("=" * 100)
    print(
        f"RESULTS: {folder_name}"
    )
    print(
        f"Seed: {SEED} | "
        f"Training: {TRAINING_RANDOMIZATION} | "
        f"Model: {MODEL}"
    )
    print("=" * 100)

    for metric in METRICS:
        eval_method = metric.upper()

        print()
        print(
            f"{10 * '-'} "
            f"ABSOLUTE POSITION ERROR "
            f"({eval_method}) "
            f"{10 * '-'}"
        )

        for controller in CONTROLLERS:
            mean = stats_results[metric][
                f"{controller}_abs_mean"
            ]
            ci = stats_results[metric][
                f"{controller}_abs_ci"
            ]

            label = CONTROLLER_LABELS[
                controller
            ]

            print(
                f"{label} X:  "
                f"{mean[0]} ± {ci[0]}"
            )
            print(
                f"{label} Y:  "
                f"{mean[1]} ± {ci[1]}"
            )
            print(
                f"{label} Z:  "
                f"{mean[2]} ± {ci[2]}"
            )
            print(
                f"{label} 2D: "
                f"{mean[3]} ± {ci[3]}"
            )
            print(
                f"{label} 3D: "
                f"{mean[4]} ± {ci[4]}"
            )
            print()

        print(
            f"{10 * '-'} "
            f"VELOCITY ERROR "
            f"({eval_method}) "
            f"{10 * '-'}"
        )

        for controller in CONTROLLERS:
            mean = stats_results[metric][
                f"{controller}_vel_mean"
            ]
            ci = stats_results[metric][
                f"{controller}_vel_ci"
            ]

            label = CONTROLLER_LABELS[
                controller
            ]

            print(
                f"{label} X:  "
                f"{mean[0]} ± {ci[0]}"
            )
            print(
                f"{label} Y:  "
                f"{mean[1]} ± {ci[1]}"
            )
            print(
                f"{label} Z:  "
                f"{mean[2]} ± {ci[2]}"
            )
            print(
                f"{label} 2D: "
                f"{mean[3]} ± {ci[3]}"
            )
            print(
                f"{label} 3D: "
                f"{mean[4]} ± {ci[4]}"
            )
            print()


# ============================================================
# CSV output
# ============================================================

def save_summary_csv(
    stats_results,
    output_dir,
    folder_name,
):
    csv_path = os.path.join(
        output_dir,
        "benchmark_summary.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "seed",
            "training_randomization",
            "model",
            "recording_folder",
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

        for metric in METRICS:
            for evaluation, suffix in [
                (
                    "absolute_position",
                    "abs",
                ),
                (
                    "velocity",
                    "vel",
                ),
            ]:
                for controller in CONTROLLERS:
                    mean = stats_results[
                        metric
                    ][
                        f"{controller}_{suffix}_mean"
                    ]

                    ci = stats_results[
                        metric
                    ][
                        f"{controller}_{suffix}_ci"
                    ]

                    writer.writerow([
                        SEED,
                        TRAINING_RANDOMIZATION,
                        MODEL,
                        folder_name,
                        metric.upper(),
                        evaluation,
                        CONTROLLER_LABELS[
                            controller
                        ],
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

    print(
        f"Summary saved to: {csv_path}"
    )


def save_raw_per_evaluation_csv(
    results,
    directories,
    output_dir,
    folder_name,
):
    csv_path = os.path.join(
        output_dir,
        "benchmark_raw_per_evaluation.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "seed",
            "training_randomization",
            "model",
            "recording_folder",
            "eval_index",
            "eval_directory",
            "metric",
            "evaluation",
            "controller",
            "x",
            "y",
            "z",
            "2d",
            "3d",
        ])

        for metric in METRICS:
            for i, directory_name in enumerate(
                directories
            ):
                for evaluation, suffix in [
                    (
                        "absolute_position",
                        "abs_pos",
                    ),
                    (
                        "velocity",
                        "vel",
                    ),
                ]:
                    for controller in CONTROLLERS:
                        values = results[
                            metric
                        ][
                            f"{controller}_{suffix}_errors"
                        ][i]

                        writer.writerow([
                            SEED,
                            TRAINING_RANDOMIZATION,
                            MODEL,
                            folder_name,
                            i,
                            directory_name,
                            metric.upper(),
                            evaluation,
                            CONTROLLER_LABELS[
                                controller
                            ],
                            values[0],
                            values[1],
                            values[2],
                            values[3],
                            values[4],
                        ])

    print(
        f"Raw results saved to: "
        f"{csv_path}"
    )


# ============================================================
# Plotting
# ============================================================

def plot_3_controller_bar(
    stats_results,
    metric,
    evaluation,
    output_dir,
    filename,
    title,
):
    """
    One aggregate 3D bar plot:
        IK
        Hybrid Offset Policy
        Fully Learned Policy
    """

    if evaluation == "absolute_position":
        suffix = "abs"
        ylabel = metric.upper()
    elif evaluation == "velocity":
        suffix = "vel"
        ylabel = metric.upper()
    else:
        raise ValueError(
            "Unknown evaluation type."
        )

    values = []
    cis = []
    labels = []

    for controller in CONTROLLERS:
        mean = stats_results[metric][
            f"{controller}_{suffix}_mean"
        ]
        ci = stats_results[metric][
            f"{controller}_{suffix}_ci"
        ]

        values.append(mean[4])
        cis.append(ci[4])
        labels.append(
            CONTROLLER_LABELS[controller]
        )

    x = np.arange(
        len(CONTROLLERS)
    )

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        x,
        values,
        yerr=cis,
        capsize=5,
        width=0.6,
        color=["tab:blue", "tab:orange", "tab:green"],
    )

    plt.xticks(
        x,
        labels,
        rotation=10,
        ha="right",
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.6f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        filename,
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Plot saved to: {path}"
    )


# ============================================================
# Process ONE complete recording folder
# ============================================================

def process_recording_folder(
    folder_path,
):
    folder_name = os.path.basename(
        os.path.normpath(folder_path)
    )

    output_dir = os.path.join(
        OUTPUT_ROOT,
        folder_name,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    directories = [
        name
        for name in os.listdir(folder_path)
        if os.path.isdir(
            os.path.join(
                folder_path,
                name,
            )
        )
    ]

    directories.sort(
        key=directory_sort_key
    )

    n_dirs = len(directories)

    if n_dirs == 0:
        raise RuntimeError(
            f"No evaluation directories "
            f"found in {folder_path}"
        )

    print()
    print("=" * 100)
    print(
        f"Processing: {folder_path}"
    )
    print(
        f"Number of evaluations: {n_dirs}"
    )
    print("=" * 100)

    results = initialize_results(
        n_dirs
    )

    timestep_results = {}

    for metric in METRICS:

        timestep_results[metric] = {}

        for controller in CONTROLLERS:

            timestep_results[metric][
                f"{controller}_abs_pos_errors"
            ] = []

            timestep_results[metric][
                f"{controller}_vel_errors"
            ] = []

    # ========================================================
    # IMPORTANT:
    # Everything is loaded/calculated together here.
    # One pass through every evaluation directory.
    # ========================================================

    with tqdm(
        total=n_dirs,
        desc=folder_name,
    ) as pbar:

        for i, directory_name in enumerate(
            directories
        ):
            directory = os.path.join(
                folder_path,
                directory_name,
            )

            # ------------------------------------------------
            # Load POSITION data for ALL controllers
            # ------------------------------------------------

            reference_omni = load_csv_checked(
                os.path.join(
                    directory,
                    "trajectory_reference_omni.csv",
                )
            )

            ik_omni = load_csv_checked(
                os.path.join(
                    directory,
                    "trajectory_ik_omni.csv",
                )
            )

            offset_omni = load_csv_checked(
                os.path.join(
                    directory,
                    "trajectory_offset_omni.csv",
                )
            )

            policy_omni = load_csv_checked(
                os.path.join(
                    directory,
                    "trajectory_policy_omni.csv",
                )
            )

            # ------------------------------------------------
            # Load VELOCITY data for ALL controllers
            # ------------------------------------------------

            reference_omni_vel = load_csv_checked(
                os.path.join(
                    directory,
                    "trajectory_reference_omni_vel.csv",
                )
            )

            ik_omni_vel = load_csv_checked(
                os.path.join(
                    directory,
                    "trajectory_ik_omni_vel.csv",
                )
            )

            offset_omni_vel = load_csv_checked(
                os.path.join(
                    directory,
                    "trajectory_offset_omni_vel.csv",
                )
            )

            policy_omni_vel = load_csv_checked(
                os.path.join(
                    directory,
                    "trajectory_policy_omni_vel.csv",
                )
            )

            # ------------------------------------------------
            # Trim EVERYTHING simultaneously
            # ------------------------------------------------

            (
                reference_omni,
                ik_omni,
                offset_omni,
                policy_omni,
                reference_omni_vel,
                ik_omni_vel,
                offset_omni_vel,
                policy_omni_vel,
            ) = trim_to_same_length(
                reference_omni,
                ik_omni,
                offset_omni,
                policy_omni,
                reference_omni_vel,
                ik_omni_vel,
                offset_omni_vel,
                policy_omni_vel,
            )

            # ------------------------------------------------
            # Calculate ALL metrics for ALL controllers
            # ------------------------------------------------

            for metric in METRICS:

                results[metric][
                    "ik_abs_pos_errors"
                ][i] = get_errors(
                    reference_omni,
                    ik_omni,
                    metric=metric,
                )

                results[metric][
                    "offset_abs_pos_errors"
                ][i] = get_errors(
                    reference_omni,
                    offset_omni,
                    metric=metric,
                )

                results[metric][
                    "policy_abs_pos_errors"
                ][i] = get_errors(
                    reference_omni,
                    policy_omni,
                    metric=metric,
                )

                results[metric][
                    "ik_vel_errors"
                ][i] = get_errors(
                    reference_omni_vel,
                    ik_omni_vel,
                    metric=metric,
                )

                results[metric][
                    "offset_vel_errors"
                ][i] = get_errors(
                    reference_omni_vel,
                    offset_omni_vel,
                    metric=metric,
                )

                results[metric][
                    "policy_vel_errors"
                ][i] = get_errors(
                    reference_omni_vel,
                    policy_omni_vel,
                    metric=metric,
                )

                # ========================================================
                # Timestep-wise position errors
                # ========================================================

                timestep_results[metric][
                    "ik_abs_pos_errors"
                ].append(
                    get_timestep_errors(
                        reference_omni,
                        ik_omni,
                        metric=metric,
                    )
                )

                timestep_results[metric][
                    "offset_abs_pos_errors"
                ].append(
                    get_timestep_errors(
                        reference_omni,
                        offset_omni,
                        metric=metric,
                    )
                )

                timestep_results[metric][
                    "policy_abs_pos_errors"
                ].append(
                    get_timestep_errors(
                        reference_omni,
                        policy_omni,
                        metric=metric,
                    )
                )


                # ========================================================
                # Timestep-wise velocity errors
                # ========================================================

                timestep_results[metric][
                    "ik_vel_errors"
                ].append(
                    get_timestep_errors(
                        reference_omni_vel,
                        ik_omni_vel,
                        metric=metric,
                    )
                )

                timestep_results[metric][
                    "offset_vel_errors"
                ].append(
                    get_timestep_errors(
                        reference_omni_vel,
                        offset_omni_vel,
                        metric=metric,
                    )
                )

                timestep_results[metric][
                    "policy_vel_errors"
                ].append(
                    get_timestep_errors(
                        reference_omni_vel,
                        policy_omni_vel,
                        metric=metric,
                    )
                )

            pbar.update(1)

    # ========================================================
    # Statistics AFTER all evaluations are loaded
    # ========================================================

    stats_results = compute_statistics(
        results
    )

    print_statistics(
        stats_results,
        folder_name,
    )

    save_summary_csv(
        stats_results,
        output_dir,
        folder_name,
    )

    save_raw_per_evaluation_csv(
        results,
        directories,
        output_dir,
        folder_name,
    )

    # ========================================================
    # FOUR plots for this complete recording folder
    # ========================================================

    plot_3_controller_bar(
        stats_results,
        metric="mae",
        evaluation="absolute_position",
        output_dir=output_dir,
        filename=(
            "bar_omni_absolute_position_mae.png"
        ),
        title=(
            "Omni Absolute Position Error "
            "MAE"
        ),
    )

    plot_3_controller_bar(
        stats_results,
        metric="mse",
        evaluation="absolute_position",
        output_dir=output_dir,
        filename=(
            "bar_omni_absolute_position_mse.png"
        ),
        title=(
            "Omni Absolute Position Error "
            "MSE"
        ),
    )

    plot_3_controller_bar(
        stats_results,
        metric="mae",
        evaluation="velocity",
        output_dir=output_dir,
        filename=(
            "bar_omni_velocity_mae.png"
        ),
        title=(
            "Omni Velocity Error "
            "MAE"
        ),
    )

    plot_3_controller_bar(
        stats_results,
        metric="mse",
        evaluation="velocity",
        output_dir=output_dir,
        filename=(
            "bar_omni_velocity_mse.png"
        ),
        title=(
            "Omni Velocity Error "
            "MSE"
        ),
    )

    # ========================================================
    # Average timestep plots
    # ========================================================

    plot_timestep_average(
        timestep_results,
        metric="mae",
        evaluation="absolute_position",
        output_dir=output_dir,
        filename=(
            "timestep_absolute_position_mae.png"
        ),
        title=(
            "Average Absolute Position Error "
            "per Timestep -  MAE"
        ),
    )

    plot_timestep_average(
        timestep_results,
        metric="mse",
        evaluation="absolute_position",
        output_dir=output_dir,
        filename=(
            "timestep_absolute_position_mse.png"
        ),
        title=(
            "Average Absolute Position Error "
            "per Timestep -  MSE"
        ),
    )

    plot_timestep_average(
        timestep_results,
        metric="mae",
        evaluation="velocity",
        output_dir=output_dir,
        filename=(
            "timestep_velocity_mae.png"
        ),
        title=(
            "Average Velocity Error "
            "per Timestep -  MAE"
        ),
    )

    plot_timestep_average(
        timestep_results,
        metric="mse",
        evaluation="velocity",
        output_dir=output_dir,
        filename=(
            "timestep_velocity_mse.png"
        ),
        title=(
            "Average Velocity Error "
            "per Timestep -  MSE"
        ),
    )

def plot_timestep_average(
    timestep_results,
    metric,
    evaluation,
    output_dir,
    filename,
    title,
):

    if evaluation == "absolute_position":
        suffix = "abs_pos"
    elif evaluation == "velocity":
        suffix = "vel"
    else:
        raise ValueError(
            "Unknown evaluation."
        )

    colors = {
        "ik": "tab:blue",
        "offset": "tab:orange",
        "policy": "tab:green",
    }

    plt.figure(
        figsize=(10, 5)
    )

    for controller in CONTROLLERS:

        sequences = timestep_results[
            metric
        ][
            f"{controller}_{suffix}_errors"
        ]

        mean, ci = (
            compute_timestep_mean_and_ci(
                sequences
            )
        )

        time = (
            np.arange(mean.shape[0])
            * DT
        )

        # 3D metric = column 4
        mean_3d = mean[:, 4]
        ci_3d = ci[:, 4]

        plt.plot(
            time,
            mean_3d,
            linewidth=2,
            color=colors[controller],
            label=CONTROLLER_LABELS[
                controller
            ],
        )

        lower_bound = np.maximum(
            mean_3d - ci_3d,
            0.0,
        )

        upper_bound = (
            mean_3d + ci_3d
        )

        plt.fill_between(
            time,
            lower_bound,
            upper_bound,
            color=colors[controller],
            alpha=0.15,
        )

    plt.xlabel("Time [s]")
    plt.ylabel(
        f" {metric.upper()} Error"
    )

    plt.title(title)

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        filename,
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Timestep plot saved to: {path}"
    )


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    for folder_path in RECORDING_FOLDERS:
        if not os.path.isdir(
            folder_path
        ):
            print(
                f"WARNING: folder does not exist: "
                f"{folder_path}"
            )
            continue

        process_recording_folder(
            folder_path
        )

    print()
    print("=" * 100)
    print("Finished.")
    print(
        f"All results saved below: "
        f"{OUTPUT_ROOT}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
