import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from tqdm import tqdm
import argparse

from matplotlib.lines import Line2D

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir",
        type=str,
        default="seeded_benchmarks",
    )

    return parser.parse_args()


args = parse_args()

BASE_DIR = args.base_dir

CONTROLLERS = {
    "offset": {
        "name": "OFFSET",
        "base_dir": os.path.join(BASE_DIR, "offset"),
    },
    "policy": {
        "name": "POLICY",
        "base_dir": os.path.join(BASE_DIR, "policy"),
    },
}

RANDOMIZED_LABEL = "RAND"
NON_RANDOMIZED_LABEL = "NON_RAND"

VALUE_COLUMN = "3d_mean"
CI_COLUMN = "3d_ci"



CONTROLLER_DISPLAY_NAMES = {
    "IK": "IK",
    "OFFSET": "Hybrid Offset",
    "POLICY": "Fully Learned",
}


def controller_display_name(controller):
    return CONTROLLER_DISPLAY_NAMES.get(
        str(controller).upper(),
        str(controller),
    )


def learned_controller_display_name(
    controller,
    condition,
):
    name = controller_display_name(controller)

    # Only domain-randomized training gets an extra label.
    if condition == RANDOMIZED_LABEL:
        return f"{name} (DR)"

    return name


def metric_display_name(metric):
    metric = str(metric).upper()

    if metric == "MSE":
        return "MSE"

    return f"3D {metric}"


TIMESTEP_PLOT_MODE = "combined"

# Use ["MAE"] for exactly 5 timestep graphs.
# Change to ["MAE", "MSE"] if you also want MSE timestep graphs.
TIMESTEP_METRICS_TO_PLOT = ["MAE", "MSE"]

# Only these two evaluations are used for timestep plots.
TIMESTEP_EVALUATIONS_TO_PLOT = [
    "absolute_position",
    "velocity",
]



for controller_key in CONTROLLERS:
    controller_base = CONTROLLERS[controller_key]["base_dir"]

    CONTROLLERS[controller_key]["rand_dir"] = os.path.join(
        controller_base,
        "rand",
    )

    CONTROLLERS[controller_key]["no_rand_dir"] = os.path.join(
        controller_base,
        "no_rand",
    )

    CONTROLLERS[controller_key]["plot_dir"] = os.path.join(
        controller_base,
        "plots",
    )

    CONTROLLERS[controller_key]["timestep_plot_dir"] = os.path.join(
        controller_base,
        "plots",
        "timestep",
    )

    os.makedirs(
        CONTROLLERS[controller_key]["plot_dir"],
        exist_ok=True,
    )

    os.makedirs(
        CONTROLLERS[controller_key]["timestep_plot_dir"],
        exist_ok=True,
    )


# Combined OFFSET + POLICY plots
COMBINED_PLOT_DIR = os.path.join(
    BASE_DIR,
    "plots",
)

COMBINED_TIMESTEP_PLOT_DIR = os.path.join(
    BASE_DIR,
    "plots",
    "timestep",
)

os.makedirs(
    COMBINED_PLOT_DIR,
    exist_ok=True,
)

os.makedirs(
    COMBINED_TIMESTEP_PLOT_DIR,
    exist_ok=True,
)




# Controller-specific plots:
# IK vs NON_RAND vs RAND
CONDITION_COLORS = {
    "IK": "tab:blue",
    "NON_RAND": "tab:orange",
    "RAND": "tab:red",
}

# Combined plots:
# IK vs OFFSET vs POLICY
COMBINED_COLORS = {
    "IK": "tab:blue",
    "OFFSET_NON_RAND": "tab:orange",
    "OFFSET_RAND": "tab:red",
    "POLICY_NON_RAND": "tab:green",
    "POLICY_RAND": "tab:purple",
}


def load_raw_per_evaluation_csvs(
    directory,
    condition_name,
    drive_mode,
):
    """
    Load benchmark_<seed>_raw_per_evaluation.csv files.

    These files contain one error value per evaluation and are used
    to calculate pooled means and 95% confidence intervals across
    all evaluations from all seeds.
    """

    if not os.path.isdir(directory):
        print(
            f"[SKIP] Directory does not exist:\n"
            f"       {directory}"
        )
        return None

    csv_files = glob.glob(
        os.path.join(
            directory,
            "benchmark_*_raw_per_evaluation.csv",
        )
    )

    if len(csv_files) == 0:
        print(
            f"[SKIP] No raw per-evaluation CSV files found in:\n"
            f"       {directory}"
        )
        return None

    dfs = []

    description = (
        f"Loading {drive_mode.upper()} "
        f"{condition_name} raw evaluations"
    )

    for csv_file in tqdm(
        csv_files,
        desc=description,
    ):
        df = pd.read_csv(csv_file)

        required_columns = [
            "seed",
            "eval_index",
            "racetrack_id",
            "metric",
            "evaluation",
            "controller",
            "3d",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if len(missing_columns) > 0:
            raise ValueError(
                f"Raw file {csv_file} is missing columns: "
                f"{missing_columns}"
            )

        df["racetrack_id"] = df["racetrack_id"].apply(
            normalize_racetrack_id
        )

        df["metric"] = (
            df["metric"]
            .astype(str)
            .str.upper()
        )

        df["evaluation"] = df["evaluation"].astype(str)

        df["controller"] = (
            df["controller"]
            .astype(str)
            .str.upper()
        )

        df["condition"] = condition_name
        df["drive_mode"] = drive_mode.upper()
        df["source_file"] = os.path.basename(csv_file)

        dfs.append(df)

    return pd.concat(
        dfs,
        ignore_index=True,
    )

def compute_pooled_mean_and_ci(
        values,
        confidence=0.95,
        n_bootstrap=10000,
        random_state=42,
    ):
        """
        Calculate mean and percentile-bootstrap confidence interval
        from pooled per-evaluation error values.
        """

        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[
            ~np.isnan(values)
        ]

        if len(values) == 0:
            return np.nan, (np.nan, np.nan)

        mean = np.mean(values)

        if len(values) < 2:
            return mean, (mean, mean)

        rng = np.random.default_rng(
            random_state
        )

        bootstrap_means = np.empty(
            n_bootstrap,
            dtype=float,
        )

        for i in range(n_bootstrap):
            sample = rng.choice(
                values,
                size=len(values),
                replace=True,
            )

            bootstrap_means[i] = np.mean(
                sample
            )

        alpha = 1.0 - confidence

        lower = np.percentile(
            bootstrap_means,
            100.0 * alpha / 2.0,
        )

        upper = np.percentile(
            bootstrap_means,
            100.0 * (1.0 - alpha / 2.0),
        )

        return mean, (lower, upper)

def aggregate_raw_evaluations(df):
    """
    Pool all per-evaluation errors from all seeds.

    Produces statistics:
    - per racetrack
    - over all racetracks

    The returned DataFrame has the same basic structure expected
    by the existing plotting functions.
    """

    rows = []

    base_group_cols = [
        "drive_mode",
        "condition",
        "metric",
        "evaluation",
        "controller",
    ]


    racetrack_group_cols = (
        base_group_cols
        + ["racetrack_id"]
    )

    groups = list(
        df.groupby(
            racetrack_group_cols
        )
    )

    for group_values, group in tqdm(
        groups,
        desc="Pooling per-racetrack evaluations",
    ):
        row = dict(
            zip(
                racetrack_group_cols,
                group_values,
            )
        )

        values = group["3d"].values

        mean, ci = compute_pooled_mean_and_ci(
            values
        )

        row["num_seeds"] = (
            group["seed"].nunique()
        )

        row["num_evaluations"] = len(
            values
        )

        row[VALUE_COLUMN] = mean

        # Store bounds, not symmetric +/- CI
        row["3d_ci_lower"] = ci[0]
        row["3d_ci_upper"] = ci[1]

        rows.append(row)


    groups = list(
        df.groupby(
            base_group_cols
        )
    )

    for group_values, group in tqdm(
        groups,
        desc="Pooling all-racetrack evaluations",
    ):
        row = dict(
            zip(
                base_group_cols,
                group_values,
            )
        )

        row["racetrack_id"] = "all"

        values = group["3d"].values

        mean, ci = compute_pooled_mean_and_ci(
            values
        )

        row["num_seeds"] = (
            group["seed"].nunique()
        )

        row["num_evaluations"] = len(
            values
        )

        row[VALUE_COLUMN] = mean
        row["3d_ci_lower"] = ci[0]
        row["3d_ci_upper"] = ci[1]

        rows.append(row)

    return pd.DataFrame(rows)

def normalize_racetrack_id(value):
    if pd.isna(value):
        return "all"

    value = str(value)

    if value.lower() == "all":
        return "all"

    try:
        value_float = float(value)
        if value_float.is_integer():
            return str(int(value_float))
    except ValueError:
        pass

    return value


def sort_racetrack_ids(racetrack_ids):
    racetrack_ids = [
        normalize_racetrack_id(r)
        for r in racetrack_ids
    ]

    numeric_ids = []
    other_ids = []
    has_all = False

    for racetrack_id in racetrack_ids:
        if racetrack_id == "all":
            has_all = True
        else:
            try:
                numeric_ids.append(
                    (
                        int(racetrack_id),
                        racetrack_id,
                    )
                )
            except ValueError:
                other_ids.append(racetrack_id)

    sorted_ids = [
        racetrack_id
        for _, racetrack_id in sorted(
            set(numeric_ids),
            key=lambda x: x[0],
        )
    ]

    sorted_ids.extend(
        sorted(
            set(other_ids)
        )
    )

    if has_all:
        sorted_ids.append("all")

    return sorted_ids


def racetrack_label(racetrack_id):
    racetrack_id = normalize_racetrack_id(racetrack_id)

    if racetrack_id == "all":
        return "All racetracks"

    return f"Racetrack {racetrack_id}"


def racetrack_filename_part(racetrack_id):
    racetrack_id = normalize_racetrack_id(racetrack_id)

    if racetrack_id == "all":
        return "all"

    return f"racetrack_{racetrack_id}"


def safe_name(value):
    return str(value).lower().replace("/", "_").replace(" ", "_")



def load_benchmark_csvs(
    directory,
    condition_name,
    drive_mode,
):
    """
    Load all normal benchmark CSVs from one directory.

    Important:
    timestep CSVs are excluded here because they have a different structure.
    """

    if not os.path.isdir(directory):
        print(
            f"[SKIP] Directory does not exist:\n"
            f"       {directory}"
        )
        return None

    csv_files = glob.glob(
        os.path.join(
            directory,
            "benchmark_*.csv",
        )
    )

    csv_files = [
        f
        for f in csv_files
        if "raw_per_evaluation" not in os.path.basename(f)
        and "timestep_errors" not in os.path.basename(f)
    ]

    if len(csv_files) == 0:
        print(
            f"[SKIP] No benchmark CSV files found in:\n"
            f"       {directory}"
        )
        return None

    dfs = []

    description = (
        f"Loading {drive_mode.upper()} "
        f"{condition_name} benchmarks"
    )

    for csv_file in tqdm(
        csv_files,
        desc=description,
    ):
        df = pd.read_csv(csv_file)

        if "racetrack_id" not in df.columns:
            df["racetrack_id"] = "all"

        df["racetrack_id"] = df["racetrack_id"].apply(
            normalize_racetrack_id
        )

        df["metric"] = df["metric"].astype(str).str.upper()
        df["evaluation"] = df["evaluation"].astype(str)
        df["controller"] = df["controller"].astype(str).str.upper()

        df["condition"] = condition_name
        df["drive_mode"] = drive_mode.upper()
        df["source_file"] = os.path.basename(csv_file)

        dfs.append(df)

    if len(dfs) == 0:
        return None

    return pd.concat(
        dfs,
        ignore_index=True,
    )


def load_timestep_error_csvs(
    directory,
    condition_name,
    drive_mode,
):
    """
    Load benchmark_<seed>_timestep_errors.csv files.

    These files are generated by benchmark.py and contain timestep-wise
    averaged absolute_position and velocity errors.
    """

    if not os.path.isdir(directory):
        print(
            f"[SKIP] Directory does not exist for timestep data:\n"
            f"       {directory}"
        )
        return None

    csv_files = glob.glob(
        os.path.join(
            directory,
            "benchmark_*_timestep_errors.csv",
        )
    )

    if len(csv_files) == 0:
        print(
            f"[SKIP] No timestep error CSV files found in:\n"
            f"       {directory}"
        )
        return None

    dfs = []

    description = (
        f"Loading {drive_mode.upper()} "
        f"{condition_name} timestep errors"
    )

    for csv_file in tqdm(
        csv_files,
        desc=description,
    ):
        df = pd.read_csv(csv_file)

        required_columns = [
            "seed",
            "racetrack_id",
            "timestep",
            "time_s",
            "metric",
            "evaluation",
            "controller",
            "3d_mean",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if len(missing_columns) > 0:
            raise ValueError(
                f"Timestep file {csv_file} is missing columns: "
                f"{missing_columns}"
            )

        df["racetrack_id"] = df["racetrack_id"].apply(
            normalize_racetrack_id
        )

        df["metric"] = df["metric"].astype(str).str.upper()
        df["evaluation"] = df["evaluation"].astype(str)
        df["controller"] = df["controller"].astype(str).str.upper()

        df["condition"] = condition_name
        df["drive_mode"] = drive_mode.upper()
        df["source_file"] = os.path.basename(csv_file)

        dfs.append(df)

    if len(dfs) == 0:
        return None

    return pd.concat(
        dfs,
        ignore_index=True,
    )


def compute_mean_and_ci(
    values,
    confidence=0.95,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        ~np.isnan(values)
    ]

    n = len(values)

    if n == 0:
        return np.nan, 0.0

    mean = np.mean(values)

    if n < 2:
        return mean, 0.0

    sem = stats.sem(
        values,
        ddof=1,
    )

    alpha = 1.0 - confidence

    t_crit = stats.t.ppf(
        1.0 - alpha / 2.0,
        df=n - 1,
    )

    ci = sem * t_crit

    return mean, ci


def aggregate_across_seeds(df):
    """
    Average normal benchmark values across seeds.
    """

    group_cols = [
        "drive_mode",
        "condition",
        "racetrack_id",
        "metric",
        "evaluation",
        "controller",
    ]

    rows = []

    groups = list(
        df.groupby(group_cols)
    )

    for group_values, group in tqdm(
        groups,
        desc="Averaging benchmarks",
    ):
        row = dict(
            zip(
                group_cols,
                group_values,
            )
        )

        row["num_seeds"] = group["seed"].nunique()

        mean, ci = compute_mean_and_ci(
            group[VALUE_COLUMN].values
        )

        row[VALUE_COLUMN] = mean
        row[CI_COLUMN] = ci

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_timestep_across_seeds(df):
    """
    Average timestep-wise values across seeds.

    Input rows are already averaged over evaluations inside benchmark.py.
    This function averages those timestep means across seeds.
    """

    group_cols = [
        "drive_mode",
        "condition",
        "racetrack_id",
        "timestep",
        "time_s",
        "metric",
        "evaluation",
        "controller",
    ]

    rows = []

    groups = list(
        df.groupby(group_cols)
    )

    for group_values, group in tqdm(
        groups,
        desc="Averaging timestep errors",
    ):
        row = dict(
            zip(
                group_cols,
                group_values,
            )
        )

        row["num_seeds"] = group["seed"].nunique()

        if "n_evaluations" in group.columns:
            row["mean_n_evaluations"] = np.mean(
                group["n_evaluations"].values
            )
        else:
            row["mean_n_evaluations"] = np.nan

        mean, ci = compute_mean_and_ci(
            group[VALUE_COLUMN].values
        )

        row[VALUE_COLUMN] = mean
        row[CI_COLUMN] = ci

        rows.append(row)

    averaged_df = pd.DataFrame(rows)

    if not averaged_df.empty:
        averaged_df = averaged_df.sort_values(
            [
                "drive_mode",
                "condition",
                "racetrack_id",
                "metric",
                "evaluation",
                "controller",
                "timestep",
            ]
        )

    return averaged_df


def get_value(
    agg_df,
    condition,
    controller,
    racetrack_id,
    metric,
    evaluation,
    drive_mode=None,
):
    racetrack_id = normalize_racetrack_id(racetrack_id)

    mask = (
        (agg_df["condition"] == condition)
        & (agg_df["controller"] == controller)
        & (
            agg_df["racetrack_id"].astype(str)
            == str(racetrack_id)
        )
        & (agg_df["metric"] == metric)
        & (agg_df["evaluation"] == evaluation)
    )

    if drive_mode is not None:
        mask = mask & (
            agg_df["drive_mode"] == drive_mode
        )

    row = agg_df[mask]

    if row.empty:
        return None, None

    mean = row[VALUE_COLUMN].iloc[0]

    lower = row["3d_ci_lower"].iloc[0]
    upper = row["3d_ci_upper"].iloc[0]

    yerr = (
        mean - lower,
        upper - mean,
    )

    return mean, yerr


def get_timestep_series(
    timestep_df,
    condition,
    controller,
    racetrack_id,
    metric,
    evaluation,
    drive_mode=None,
):
    racetrack_id = normalize_racetrack_id(racetrack_id)

    mask = (
        (timestep_df["condition"] == condition)
        & (timestep_df["controller"] == controller)
        & (
            timestep_df["racetrack_id"].astype(str)
            == str(racetrack_id)
        )
        & (timestep_df["metric"] == metric)
        & (timestep_df["evaluation"] == evaluation)
    )

    if drive_mode is not None:
        mask = mask & (
            timestep_df["drive_mode"] == drive_mode
        )

    series_df = timestep_df[mask].copy()

    if series_df.empty:
        return None

    series_df = series_df.sort_values(
        "timestep"
    )

    return {
        "time_s": series_df["time_s"].values,
        "mean": series_df[VALUE_COLUMN].values,
        "ci": series_df[CI_COLUMN].values,
    }


def plot_controller_metric_over_racetracks(
    plot_data,
    metric,
    evaluation,
    controller_name,
    plot_dir,
):
    """
    Plot:

        IK
        NON_RAND controller
        RAND controller

    Example for OFFSET:
        IK vs NON_RAND OFFSET vs RAND OFFSET
    """

    display_controller = controller_display_name(
        controller_name
    )

    display_metric = metric_display_name(
        metric
    )

    racetrack_ids = list(
        plot_data.keys()
    )

    labels = [
        "Average"
        if str(r) == "all"
        else f"Racetrack {r}"
        for r in racetrack_ids
    ]

    ik_values = [
        plot_data[r]["IK"][0]
        for r in racetrack_ids
    ]

    non_rand_values = [
        plot_data[r]["NON_RAND"][0]
        for r in racetrack_ids
    ]

    rand_values = [
        plot_data[r]["RAND"][0]
        for r in racetrack_ids
    ]

    ik_errors = [
        plot_data[r]["IK"][1]
        for r in racetrack_ids
    ]

    non_rand_errors = [
        plot_data[r]["NON_RAND"][1]
        for r in racetrack_ids
    ]

    rand_errors = [
        plot_data[r]["RAND"][1]
        for r in racetrack_ids
    ]

    ik_errors = np.array(
        ik_errors
    ).T

    non_rand_errors = np.array(
        non_rand_errors
    ).T

    rand_errors = np.array(
        rand_errors
    ).T

    x = np.arange(
        len(labels)
    )

    width = 0.25

    plt.figure(
        figsize=(
            max(
                10,
                len(labels) * 1.5,
            ),
            6,
        )
    )

    plt.bar(
        x - width,
        ik_values,
        width,
        yerr=ik_errors,
        capsize=5,
        label="IK",
        color=CONDITION_COLORS["IK"],
    )

    plt.bar(
        x,
        non_rand_values,
        width,
        yerr=non_rand_errors,
        capsize=5,
        label=learned_controller_display_name(
            controller_name,
            NON_RANDOMIZED_LABEL,
        ),
        color=CONDITION_COLORS["NON_RAND"],
    )

    plt.bar(
        x + width,
        rand_values,
        width,
        yerr=rand_errors,
        capsize=5,
        label=learned_controller_display_name(
            controller_name,
            RANDOMIZED_LABEL,
        ),
        color=CONDITION_COLORS["RAND"],
    )

    plt.xticks(
        x,
        labels,
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        display_metric
    )

    plt.title(
        f"{evaluation} | "
        f"{display_metric} | "
        f"IK vs {display_controller}"
    )

    plt.legend(font_size=16)

    plt.tight_layout()

    safe_metric = metric.lower()

    safe_eval = evaluation.replace(
        "/",
        "_",
    )

    filename = (
        f"{safe_metric}_"
        f"{safe_eval}_"
        f"3d_all_racetracks.png"
    )

    plt.savefig(
        os.path.join(
            plot_dir,
            filename,
        )
    )

    plt.close()


def generate_controller_plots(
    agg_df,
    controller_name,
    plot_dir,
):
    """
    Generate the current existing plot type separately
    for OFFSET and POLICY.
    """

    controller_df = agg_df[
        agg_df["drive_mode"]
        == controller_name
    ]

    racetrack_ids = sort_racetrack_ids(
        controller_df[
            "racetrack_id"
        ].astype(str).unique()
    )

    metrics = sorted(
        controller_df[
            "metric"
        ].unique()
    )

    evaluations = sorted(
        controller_df[
            "evaluation"
        ].unique()
    )

    total_plots = (
        len(metrics)
        * len(evaluations)
    )

    description = (
        f"Generating {controller_name} plots"
    )

    with tqdm(
        total=total_plots,
        desc=description,
    ) as pbar:

        for metric in metrics:

            for evaluation in evaluations:

                plot_data = {}

                for racetrack_id in racetrack_ids:

                    # IK from NON_RAND
                    ik_non_rand_value, ik_non_rand_ci = get_value(
                        agg_df=controller_df,
                        condition=NON_RANDOMIZED_LABEL,
                        controller="IK",
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode=controller_name,
                    )

                    # IK from RAND
                    ik_rand_value, ik_rand_ci = get_value(
                        agg_df=controller_df,
                        condition=RANDOMIZED_LABEL,
                        controller="IK",
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode=controller_name,
                    )

                    # NON_RANDOMIZED learned
                    non_rand_value, non_rand_ci = get_value(
                        agg_df=controller_df,
                        condition=NON_RANDOMIZED_LABEL,
                        controller=controller_name,
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode=controller_name,
                    )

                    # RANDOMIZED learned
                    rand_value, rand_ci = get_value(
                        agg_df=controller_df,
                        condition=RANDOMIZED_LABEL,
                        controller=controller_name,
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode=controller_name,
                    )


                    if ik_non_rand_value is not None:
                        ik_value = ik_non_rand_value
                        ik_ci = ik_non_rand_ci

                    else:
                        ik_value = ik_rand_value
                        ik_ci = ik_rand_ci

                    if (
                        ik_value is None
                        or non_rand_value is None
                        or rand_value is None
                    ):
                        continue

                    plot_data[
                        racetrack_id
                    ] = {
                        "IK": (
                            ik_value,
                            ik_ci,
                        ),
                        "NON_RAND": (
                            non_rand_value,
                            non_rand_ci,
                        ),
                        "RAND": (
                            rand_value,
                            rand_ci,
                        ),
                    }

                if len(plot_data) > 0:

                    plot_controller_metric_over_racetracks(
                        plot_data=plot_data,
                        metric=metric,
                        evaluation=evaluation,
                        controller_name=controller_name,
                        plot_dir=plot_dir,
                    )

                pbar.update(1)

def get_controller_ik_timestep_series(
    timestep_df,
    racetrack_id,
    metric,
    evaluation,
    drive_mode,
):
    """
    Get IK timestep series for a controller-specific plot.

    Priority:
    1. IK from NON_RAND
    2. IK from RAND
    """

    search_order = [
        NON_RANDOMIZED_LABEL,
        RANDOMIZED_LABEL,
    ]

    for condition in search_order:
        series = get_timestep_series(
            timestep_df=timestep_df,
            condition=condition,
            controller="IK",
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode=drive_mode,
        )

        if series is not None:
            return series

    return None


def get_combined_ik_timestep_series(
    timestep_df,
    racetrack_id,
    metric,
    evaluation,
):
    """
    Get one IK timestep series for the combined plot.

    Priority:
    1. OFFSET NON_RAND IK
    2. OFFSET RAND IK
    3. POLICY NON_RAND IK
    4. POLICY RAND IK
    """

    search_order = [
        ("OFFSET", NON_RANDOMIZED_LABEL),
        ("OFFSET", RANDOMIZED_LABEL),
        ("POLICY", NON_RANDOMIZED_LABEL),
        ("POLICY", RANDOMIZED_LABEL),
    ]

    for drive_mode, condition in search_order:
        series = get_timestep_series(
            timestep_df=timestep_df,
            condition=condition,
            controller="IK",
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode=drive_mode,
        )

        if series is not None:
            return series

    return None


def add_timestep_line(
    ax,
    series,
    label,
    color,
):
    if series is None:
        return False

    time_s = series["time_s"]
    mean = series["mean"]
    ci = series["ci"]

    ax.plot(
        time_s,
        mean,
        label=label,
        color=color,
        linewidth=2.0,
    )

    if ci is not None and np.any(ci > 0.0):
        ax.fill_between(
            time_s,
            mean - ci,
            mean + ci,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    return True


def plot_controller_timestep_scope(
    timestep_df,
    controller_name,
    racetrack_id,
    metric,
    plot_dir,
):
    """
    Create one timestep plot for one racetrack/all for one controller.

    The figure contains:
    - absolute_position subplot
    - velocity subplot
    """
    display_controller = controller_display_name(
        controller_name
    )

    display_metric = metric_display_name(
        metric
    )

    fig, axes = plt.subplots(
        len(TIMESTEP_EVALUATIONS_TO_PLOT),
        1,
        figsize=(12, 8),
        sharex=True,
    )

    if len(TIMESTEP_EVALUATIONS_TO_PLOT) == 1:
        axes = [axes]

    plotted_anything = False

    for ax, evaluation in zip(
        axes,
        TIMESTEP_EVALUATIONS_TO_PLOT,
    ):
        ik_series = get_controller_ik_timestep_series(
            timestep_df=timestep_df,
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode=controller_name,
        )

        non_rand_series = get_timestep_series(
            timestep_df=timestep_df,
            condition=NON_RANDOMIZED_LABEL,
            controller=controller_name,
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode=controller_name,
        )

        rand_series = get_timestep_series(
            timestep_df=timestep_df,
            condition=RANDOMIZED_LABEL,
            controller=controller_name,
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode=controller_name,
        )

        plotted_anything |= add_timestep_line(
            ax=ax,
            series=ik_series,
            label="IK",
            color=CONDITION_COLORS["IK"],
        )

        plotted_anything |= add_timestep_line(
            ax=ax,
            series=non_rand_series,
            label=learned_controller_display_name(
                controller_name,
                NON_RANDOMIZED_LABEL,
            ),
            color=CONDITION_COLORS["NON_RAND"],
        )

        plotted_anything |= add_timestep_line(
            ax=ax,
            series=rand_series,
            label=learned_controller_display_name(
                controller_name,
                RANDOMIZED_LABEL,
            ),
            color=CONDITION_COLORS["RAND"],
        )

        ax.set_ylabel(
            f"{evaluation}\n{display_metric}"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend(
            loc="best",
        )

    if not plotted_anything:
        plt.close(fig)
        return

    axes[-1].set_xlabel(
        "Time [s]"
    )

    fig.suptitle(
        f"{display_controller} timestep error | "
        f"{racetrack_label(racetrack_id)} | "
        f"{display_metric}"
    )

    fig.tight_layout()

    filename = (
        f"timestep_"
        f"{metric.lower()}_"
        f"3d_"
        f"{racetrack_filename_part(racetrack_id)}.png"
    )

    plt.savefig(
        os.path.join(
            plot_dir,
            filename,
        )
    )

    plt.close(fig)


def generate_controller_timestep_plots(
    timestep_df,
    controller_name,
    plot_dir,
):
    """
    Generate 5 timestep plots for a complete controller:
    - racetrack 0
    - racetrack 1
    - racetrack 2
    - racetrack 3
    - all
    """

    controller_df = timestep_df[
        timestep_df["drive_mode"] == controller_name
    ]

    if controller_df.empty:
        print(
            f"[SKIP] No timestep data available for {controller_name}."
        )
        return

    racetrack_ids = sort_racetrack_ids(
        controller_df["racetrack_id"].unique()
    )

    expected_plots = (
        len(racetrack_ids)
        * len(TIMESTEP_METRICS_TO_PLOT)
    )

    with tqdm(
        total=expected_plots,
        desc=f"Generating {controller_name} timestep plots",
    ) as pbar:

        for metric in TIMESTEP_METRICS_TO_PLOT:

            for racetrack_id in racetrack_ids:

                plot_controller_timestep_scope(
                    timestep_df=controller_df,
                    controller_name=controller_name,
                    racetrack_id=racetrack_id,
                    metric=metric,
                    plot_dir=plot_dir,
                )

                pbar.update(1)


def plot_combined_timestep_scope(
    timestep_df,
    racetrack_id,
    metric,
    plot_dir,
):
    """
    Create one combined timestep plot for one racetrack/all.

    The figure contains:
    - absolute_position subplot
    - velocity subplot

    Lines:
    - IK
    - OFFSET NON_RAND
    - OFFSET RAND
    - POLICY NON_RAND
    - POLICY RAND
    """

    display_metric = metric_display_name(
        metric
    )

    fig, axes = plt.subplots(
        len(TIMESTEP_EVALUATIONS_TO_PLOT),
        1,
        figsize=(13, 8),
        sharex=True,
    )

    if len(TIMESTEP_EVALUATIONS_TO_PLOT) == 1:
        axes = [axes]

    plotted_anything = False

    for ax, evaluation in zip(
        axes,
        TIMESTEP_EVALUATIONS_TO_PLOT,
    ):
        ik_series = get_combined_ik_timestep_series(
            timestep_df=timestep_df,
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
        )

        offset_non_rand_series = get_timestep_series(
            timestep_df=timestep_df,
            condition=NON_RANDOMIZED_LABEL,
            controller="OFFSET",
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode="OFFSET",
        )

        offset_rand_series = get_timestep_series(
            timestep_df=timestep_df,
            condition=RANDOMIZED_LABEL,
            controller="OFFSET",
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode="OFFSET",
        )

        policy_non_rand_series = get_timestep_series(
            timestep_df=timestep_df,
            condition=NON_RANDOMIZED_LABEL,
            controller="POLICY",
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode="POLICY",
        )

        policy_rand_series = get_timestep_series(
            timestep_df=timestep_df,
            condition=RANDOMIZED_LABEL,
            controller="POLICY",
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode="POLICY",
        )

        plotted_anything |= add_timestep_line(
            ax=ax,
            series=ik_series,
            label="IK",
            color=COMBINED_COLORS["IK"],
        )

        plotted_anything |= add_timestep_line(
            ax=ax,
            series=offset_non_rand_series,
            label="Hybrid Offset",
            color=COMBINED_COLORS["OFFSET_NON_RAND"],
        )

        plotted_anything |= add_timestep_line(
            ax=ax,
            series=offset_rand_series,
            label="Hybrid Offset (DR)",
            color=COMBINED_COLORS["OFFSET_RAND"],
        )

        plotted_anything |= add_timestep_line(
            ax=ax,
            series=policy_non_rand_series,
            label="Fully Learned",
            color=COMBINED_COLORS["POLICY_NON_RAND"],
        )

        plotted_anything |= add_timestep_line(
            ax=ax,
            series=policy_rand_series,
            label="Fully Learned (DR)",
            color=COMBINED_COLORS["POLICY_RAND"],
        )

        ax.set_ylabel(
            f"{evaluation}\n{display_metric}"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend(
            loc="best",
        )

    if not plotted_anything:
        plt.close(fig)
        return

    axes[-1].set_xlabel(
        "Time [s]"
    )

    fig.suptitle(
        f"Timestep error | "
        f"{racetrack_label(racetrack_id)} | "
        f"{display_metric} | "
        f"IK vs Hybrid Offset vs Fully Learned"
    )

    fig.tight_layout()

    filename = (
        f"timestep_"
        f"{metric.lower()}_"
        f"3d_"
        f"{racetrack_filename_part(racetrack_id)}_"
        f"ik_vs_offset_vs_policy.png"
    )

    plt.savefig(
        os.path.join(
            plot_dir,
            filename,
        )
    )

    plt.close(fig)


def generate_combined_timestep_plots(
    timestep_df,
):
    """
    Generate exactly 5 combined timestep plots when both OFFSET and POLICY
    timestep data are available.
    """

    racetrack_ids = sort_racetrack_ids(
        timestep_df["racetrack_id"].unique()
    )

    expected_plots = (
        len(racetrack_ids)
        * len(TIMESTEP_METRICS_TO_PLOT)
    )

    with tqdm(
        total=expected_plots,
        desc="Generating combined timestep plots",
    ) as pbar:

        for metric in TIMESTEP_METRICS_TO_PLOT:

            for racetrack_id in racetrack_ids:

                plot_combined_timestep_scope(
                    timestep_df=timestep_df,
                    racetrack_id=racetrack_id,
                    metric=metric,
                    plot_dir=COMBINED_TIMESTEP_PLOT_DIR,
                )

                pbar.update(1)


def get_combined_ik_value(
    agg_df,
    racetrack_id,
    metric,
    evaluation,
):
    """
    Get one IK value for the combined plot.

    Priority:
    1. OFFSET NON_RAND IK
    2. OFFSET RAND IK
    3. POLICY NON_RAND IK
    4. POLICY RAND IK

    Since IK is the common baseline, only one IK bar is shown.
    """

    search_order = [
        ("OFFSET", NON_RANDOMIZED_LABEL),
        ("OFFSET", RANDOMIZED_LABEL),
        ("POLICY", NON_RANDOMIZED_LABEL),
        ("POLICY", RANDOMIZED_LABEL),
    ]

    for drive_mode, condition in search_order:

        value, ci = get_value(
            agg_df=agg_df,
            condition=condition,
            controller="IK",
            racetrack_id=racetrack_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode=drive_mode,
        )

        if value is not None:
            return value, ci

    return None, None


def plot_combined_metric_over_racetracks(
    plot_data,
    metric,
    evaluation,
):
    """
    Generate one combined plot containing:

        IK
        OFFSET NON_RAND
        OFFSET RAND
        POLICY NON_RAND
        POLICY RAND

    for every racetrack and the overall Average.
    """

    display_metric = metric_display_name(
        metric
    )

    racetrack_ids = list(
        plot_data.keys()
    )

    labels = [
        "Average"
        if str(r) == "all"
        else f"Racetrack {r}"
        for r in racetrack_ids
    ]

    ik_values = [
        plot_data[r]["IK"][0]
        for r in racetrack_ids
    ]

    offset_non_rand_values = [
        plot_data[r]["OFFSET_NON_RAND"][0]
        for r in racetrack_ids
    ]

    offset_rand_values = [
        plot_data[r]["OFFSET_RAND"][0]
        for r in racetrack_ids
    ]

    policy_non_rand_values = [
        plot_data[r]["POLICY_NON_RAND"][0]
        for r in racetrack_ids
    ]

    policy_rand_values = [
        plot_data[r]["POLICY_RAND"][0]
        for r in racetrack_ids
    ]

    ik_errors = [
        plot_data[r]["IK"][1]
        for r in racetrack_ids
    ]

    offset_non_rand_errors = [
        plot_data[r]["OFFSET_NON_RAND"][1]
        for r in racetrack_ids
    ]

    offset_rand_errors = [
        plot_data[r]["OFFSET_RAND"][1]
        for r in racetrack_ids
    ]

    policy_non_rand_errors = [
        plot_data[r]["POLICY_NON_RAND"][1]
        for r in racetrack_ids
    ]

    policy_rand_errors = [
        plot_data[r]["POLICY_RAND"][1]
        for r in racetrack_ids
    ]

    ik_errors = np.array(
        ik_errors
    ).T

    offset_non_rand_errors = np.array(
        offset_non_rand_errors
    ).T

    offset_rand_errors = np.array(
        offset_rand_errors
    ).T

    policy_non_rand_errors = np.array(
        policy_non_rand_errors
    ).T

    policy_rand_errors = np.array(
        policy_rand_errors
    ).T

    x = np.arange(
        len(labels)
    )

    width = 0.16

    plt.figure(
        figsize=(
            max(
                12,
                len(labels) * 2.0,
            ),
            6,
        )
    )

    plt.bar(
        x - 2 * width,
        ik_values,
        width,
        yerr=ik_errors,
        capsize=4,
        label="IK",
        color=COMBINED_COLORS["IK"],
    )

    plt.bar(
        x - width,
        offset_non_rand_values,
        width,
        yerr=offset_non_rand_errors,
        capsize=4,
        label="Hybrid Offset",
        color=COMBINED_COLORS["OFFSET_NON_RAND"],
    )

    plt.bar(
        x,
        offset_rand_values,
        width,
        yerr=offset_rand_errors,
        capsize=4,
        label="Hybrid Offset (DR)",
        color=COMBINED_COLORS["OFFSET_RAND"],
    )

    plt.bar(
        x + width,
        policy_non_rand_values,
        width,
        yerr=policy_non_rand_errors,
        capsize=4,
        label="Fully Learned",
        color=COMBINED_COLORS["POLICY_NON_RAND"],
    )

    plt.bar(
        x + 2 * width,
        policy_rand_values,
        width,
        yerr=policy_rand_errors,
        capsize=4,
        label="Fully Learned (DR)",
        color=COMBINED_COLORS["POLICY_RAND"],
    )

    plt.xticks(
        x,
        labels,
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        display_metric
    )

    plt.title(
        f"{evaluation} | "
        f"{display_metric} | "
        f"IK vs Hybrid Offset vs Fully Learned"
    )

    plt.legend(font_size=16)

    plt.tight_layout()

    safe_metric = metric.lower()

    safe_eval = evaluation.replace(
        "/",
        "_",
    )

    filename = (
        f"{safe_metric}_"
        f"{safe_eval}_"
        f"3d_ik_vs_offset_vs_policy.png"
    )

    plt.savefig(
        os.path.join(
            COMBINED_PLOT_DIR,
            filename,
        )
    )

    plt.close()


def generate_combined_plots(
    agg_df,
):
    """
    Generate combined bar plots.
    """

    racetrack_ids = sort_racetrack_ids(
        agg_df[
            "racetrack_id"
        ].astype(str).unique()
    )

    metrics = sorted(
        agg_df[
            "metric"
        ].unique()
    )

    evaluations = sorted(
        agg_df[
            "evaluation"
        ].unique()
    )

    total_plots = (
        len(metrics)
        * len(evaluations)
    )

    with tqdm(
        total=total_plots,
        desc="Generating combined plots",
    ) as pbar:

        for metric in metrics:

            for evaluation in evaluations:

                plot_data = {}

                for racetrack_id in racetrack_ids:

                    ik_value, ik_ci = get_combined_ik_value(
                        agg_df=agg_df,
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                    )

                    offset_non_rand_value, offset_non_rand_ci = get_value(
                        agg_df=agg_df,
                        condition=NON_RANDOMIZED_LABEL,
                        controller="OFFSET",
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode="OFFSET",
                    )

                    offset_rand_value, offset_rand_ci = get_value(
                        agg_df=agg_df,
                        condition=RANDOMIZED_LABEL,
                        controller="OFFSET",
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode="OFFSET",
                    )

                    policy_non_rand_value, policy_non_rand_ci = get_value(
                        agg_df=agg_df,
                        condition=NON_RANDOMIZED_LABEL,
                        controller="POLICY",
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode="POLICY",
                    )

                    policy_rand_value, policy_rand_ci = get_value(
                        agg_df=agg_df,
                        condition=RANDOMIZED_LABEL,
                        controller="POLICY",
                        racetrack_id=racetrack_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode="POLICY",
                    )

                    if (
                        ik_value is None
                        or offset_non_rand_value is None
                        or offset_rand_value is None
                        or policy_non_rand_value is None
                        or policy_rand_value is None
                    ):
                        continue

                    plot_data[
                        racetrack_id
                    ] = {
                        "IK": (
                            ik_value,
                            ik_ci,
                        ),
                        "OFFSET_NON_RAND": (
                            offset_non_rand_value,
                            offset_non_rand_ci,
                        ),
                        "OFFSET_RAND": (
                            offset_rand_value,
                            offset_rand_ci,
                        ),
                        "POLICY_NON_RAND": (
                            policy_non_rand_value,
                            policy_non_rand_ci,
                        ),
                        "POLICY_RAND": (
                            policy_rand_value,
                            policy_rand_ci,
                        ),
                    }

                if len(plot_data) > 0:

                    plot_combined_metric_over_racetracks(
                        plot_data=plot_data,
                        metric=metric,
                        evaluation=evaluation,
                    )

                pbar.update(1)


def generate_seed_level_overall_mse_plot(df):
    """
    Generate one compact figure showing the overall 3D MSE
    of every learned controller separately for each seed.

    Figure:
        - Absolute position MSE
        - Velocity MSE

    Controllers:
        - Hybrid Offset NON_RAND
        - Hybrid Offset RAND
        - Fully Learned NON_RAND
        - Fully Learned RAND

    IK is intentionally excluded.

    Seed-level values are used directly, i.e. no averaging
    across seeds is performed before plotting.
    """

    plot_df = df[
        (
            df["racetrack_id"]
            .astype(str)
            .apply(normalize_racetrack_id)
            == "all"
        )
        & (
            df["metric"].astype(str).str.upper()
            == "MSE"
        )
        & (
            df["controller"].astype(str).str.upper()
            != "IK"
        )
        & (
            df["evaluation"].isin(
                [
                    "absolute_position",
                    "velocity",
                ]
            )
        )
    ].copy()


    if plot_df.empty:
        print(
            "[SKIP] No seed-level overall MSE data found."
        )
        return


    controller_configs = [
        {
            "drive_mode": "OFFSET",
            "condition": NON_RANDOMIZED_LABEL,
            "label": "Hybrid Offset",
            "color": COMBINED_COLORS[
                "OFFSET_NON_RAND"
            ],
        },
        {
            "drive_mode": "OFFSET",
            "condition": RANDOMIZED_LABEL,
            "label": "Hybrid Offset\n(DR)",
            "color": COMBINED_COLORS[
                "OFFSET_RAND"
            ],
        },
        {
            "drive_mode": "POLICY",
            "condition": NON_RANDOMIZED_LABEL,
            "label": "Fully Learned",
            "color": COMBINED_COLORS[
                "POLICY_NON_RAND"
            ],
        },
        {
            "drive_mode": "POLICY",
            "condition": RANDOMIZED_LABEL,
            "label": "Fully Learned\n(DR)",
            "color": COMBINED_COLORS[
                "POLICY_RAND"
            ],
        },
    ]


    seeds = sorted(
        plot_df["seed"]
        .dropna()
        .astype(int)
        .unique()
    )

    marker_list = [
        "o",
        "s",
        "^",
        "D",
        "P",
        "X",
    ]

    seed_markers = {
        seed: marker_list[
            i % len(marker_list)
        ]
        for i, seed in enumerate(seeds)
    }


    if len(seeds) == 1:
        seed_offsets = {
            seeds[0]: 0.0
        }
    else:
        offsets = np.linspace(
            -0.14,
            0.14,
            len(seeds),
        )

        seed_offsets = {
            seed: offset
            for seed, offset in zip(
                seeds,
                offsets,
            )
        }


    evaluations = [
        (
            "absolute_position",
            "Absolute Position MSE",
        ),
        (
            "velocity",
            "Velocity MSE",
        ),
    ]


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )


    x = np.arange(
        len(controller_configs)
    )


    for ax, (
        evaluation,
        title,
    ) in zip(
        axes,
        evaluations,
    ):

        evaluation_df = plot_df[
            plot_df["evaluation"]
            == evaluation
        ]


        for controller_index, config in enumerate(
            controller_configs
        ):

            controller_df = evaluation_df[
                (
                    evaluation_df["drive_mode"]
                    == config["drive_mode"]
                )
                & (
                    evaluation_df["condition"]
                    == config["condition"]
                )
                & (
                    evaluation_df["controller"]
                    == config["drive_mode"]
                )
            ]


            if controller_df.empty:
                continue


            seed_values = []

            for seed in seeds:

                seed_rows = controller_df[
                    controller_df["seed"].astype(int)
                    == seed
                ]

                if seed_rows.empty:
                    continue

                value = seed_rows[
                    VALUE_COLUMN
                ].mean()


                seed_values.append(
                    value
                )


                ax.scatter(
                    controller_index
                    + seed_offsets[seed],
                    value,
                    color=config["color"],
                    marker=seed_markers[seed],
                    s=55,
                    zorder=3,
                )

            if len(seed_values) > 0:

                mean_value = np.mean(
                    seed_values
                )

                ax.hlines(
                    mean_value,
                    controller_index - 0.20,
                    controller_index + 0.20,
                    color="black",
                    linewidth=1.5,
                    zorder=2,
                )


        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                config["label"]
                for config in controller_configs
            ],
            fontsize=8,
        )

        ax.set_ylabel(
            "MSE"
        )

        ax.set_title(
            title
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )


    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=seed_markers[seed],
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6,
            label=f"Seed {seed}",
        )
        for seed in seeds
    ]


    # Mean indicator
    seed_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            label="Seed mean",
        )
    )


    fig.legend(
        handles=seed_handles,
        loc="upper center",
        ncol=len(seed_handles),
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )


    fig.suptitle(
        "Seed-Level Overall MSE of Learned Controllers",
        y=1.10,
    )


    fig.tight_layout()


    output_path = os.path.join(
        COMBINED_PLOT_DIR,
        "seed_level_overall_mse.png",
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    print(
        f"\nSeed-level overall MSE plot saved to:\n"
        f"{output_path}"
    )


def main():

    offset_rand_df = load_benchmark_csvs(
        directory=CONTROLLERS["offset"]["rand_dir"],
        condition_name=RANDOMIZED_LABEL,
        drive_mode="OFFSET",
    )

    offset_no_rand_df = load_benchmark_csvs(
        directory=CONTROLLERS["offset"]["no_rand_dir"],
        condition_name=NON_RANDOMIZED_LABEL,
        drive_mode="OFFSET",
    )

    offset_complete = (
        offset_rand_df is not None
        and offset_no_rand_df is not None
    )


    policy_rand_df = load_benchmark_csvs(
        directory=CONTROLLERS["policy"]["rand_dir"],
        condition_name=RANDOMIZED_LABEL,
        drive_mode="POLICY",
    )

    policy_no_rand_df = load_benchmark_csvs(
        directory=CONTROLLERS["policy"]["no_rand_dir"],
        condition_name=NON_RANDOMIZED_LABEL,
        drive_mode="POLICY",
    )

    policy_complete = (
        policy_rand_df is not None
        and policy_no_rand_df is not None
    )


    offset_rand_timestep_df = load_timestep_error_csvs(
        directory=CONTROLLERS["offset"]["rand_dir"],
        condition_name=RANDOMIZED_LABEL,
        drive_mode="OFFSET",
    )

    offset_no_rand_timestep_df = load_timestep_error_csvs(
        directory=CONTROLLERS["offset"]["no_rand_dir"],
        condition_name=NON_RANDOMIZED_LABEL,
        drive_mode="OFFSET",
    )

    offset_timestep_complete = (
        offset_rand_timestep_df is not None
        and offset_no_rand_timestep_df is not None
    )


    policy_rand_timestep_df = load_timestep_error_csvs(
        directory=CONTROLLERS["policy"]["rand_dir"],
        condition_name=RANDOMIZED_LABEL,
        drive_mode="POLICY",
    )

    policy_no_rand_timestep_df = load_timestep_error_csvs(
        directory=CONTROLLERS["policy"]["no_rand_dir"],
        condition_name=NON_RANDOMIZED_LABEL,
        drive_mode="POLICY",
    )

    policy_timestep_complete = (
        policy_rand_timestep_df is not None
        and policy_no_rand_timestep_df is not None
    )


    offset_rand_raw_df = load_raw_per_evaluation_csvs(
        directory=CONTROLLERS["offset"]["rand_dir"],
        condition_name=RANDOMIZED_LABEL,
        drive_mode="OFFSET",
    )

    offset_no_rand_raw_df = load_raw_per_evaluation_csvs(
        directory=CONTROLLERS["offset"]["no_rand_dir"],
        condition_name=NON_RANDOMIZED_LABEL,
        drive_mode="OFFSET",
    )

    policy_rand_raw_df = load_raw_per_evaluation_csvs(
        directory=CONTROLLERS["policy"]["rand_dir"],
        condition_name=RANDOMIZED_LABEL,
        drive_mode="POLICY",
    )

    policy_no_rand_raw_df = load_raw_per_evaluation_csvs(
        directory=CONTROLLERS["policy"]["no_rand_dir"],
        condition_name=NON_RANDOMIZED_LABEL,
        drive_mode="POLICY",
    )


    print("\n----------------------------------")
    print("BENCHMARK DATA STATUS")
    print("----------------------------------")

    print(
        f"OFFSET RAND:              "
        f"{'AVAILABLE' if offset_rand_df is not None else 'MISSING'}"
    )

    print(
        f"OFFSET NON_RAND:          "
        f"{'AVAILABLE' if offset_no_rand_df is not None else 'MISSING'}"
    )

    print(
        f"POLICY RAND:              "
        f"{'AVAILABLE' if policy_rand_df is not None else 'MISSING'}"
    )

    print(
        f"POLICY NON_RAND:          "
        f"{'AVAILABLE' if policy_no_rand_df is not None else 'MISSING'}"
    )

    print()

    print(
        f"OFFSET timestep RAND:     "
        f"{'AVAILABLE' if offset_rand_timestep_df is not None else 'MISSING'}"
    )

    print(
        f"OFFSET timestep NON_RAND: "
        f"{'AVAILABLE' if offset_no_rand_timestep_df is not None else 'MISSING'}"
    )

    print(
        f"POLICY timestep RAND:     "
        f"{'AVAILABLE' if policy_rand_timestep_df is not None else 'MISSING'}"
    )

    print(
        f"POLICY timestep NON_RAND: "
        f"{'AVAILABLE' if policy_no_rand_timestep_df is not None else 'MISSING'}"
    )

    print()

    print(
        f"OFFSET plots:             "
        f"{'WILL BE GENERATED' if offset_complete else 'SKIPPED'}"
    )

    print(
        f"POLICY plots:             "
        f"{'WILL BE GENERATED' if policy_complete else 'SKIPPED'}"
    )

    print(
        f"Combined plots:           "
        f"{'WILL BE GENERATED' if offset_complete and policy_complete else 'SKIPPED'}"
    )

    print(
        f"Timestep plot mode:       "
        f"{TIMESTEP_PLOT_MODE}"
    )

    print("----------------------------------\n")


    if not offset_complete and not policy_complete:
        print(
            "No complete normal benchmark set found.\n"
            "A controller requires both RAND and NON_RAND "
            "benchmark data before normal plots can be generated."
        )

    else:
        available_dfs = []

        if offset_complete:
            available_dfs.extend(
                [
                    offset_rand_df,
                    offset_no_rand_df,
                ]
            )

        if policy_complete:
            available_dfs.extend(
                [
                    policy_rand_df,
                    policy_no_rand_df,
                ]
            )

        all_df = pd.concat(
            available_dfs,
            ignore_index=True,
        )


        if offset_complete and policy_complete:

            print(
                "\nGenerating seed-level overall MSE plot..."
            )

            generate_seed_level_overall_mse_plot(
                all_df
            )

        raw_dfs = []

        if offset_complete:
            raw_dfs.extend([
                offset_rand_raw_df,
                offset_no_rand_raw_df,
            ])

        if policy_complete:
            raw_dfs.extend([
                policy_rand_raw_df,
                policy_no_rand_raw_df,
            ])

        raw_dfs = [
            df
            for df in raw_dfs
            if df is not None
        ]

        all_raw_df = pd.concat(
            raw_dfs,
            ignore_index=True,
        )

        learned_raw_df = all_raw_df[
            all_raw_df["controller"] != "IK"
        ].copy()

        ik_raw_df = all_raw_df[
            all_raw_df["controller"] == "IK"
        ].copy()


        ik_raw_df = ik_raw_df.drop_duplicates(
            subset=[
                "drive_mode",
                "condition",
                "eval_index",
                "racetrack_id",
                "metric",
                "evaluation",
                "3d",
            ]
        )

        all_raw_df = pd.concat(
            [
                learned_raw_df,
                ik_raw_df,
            ],
            ignore_index=True,
        )

        averaged_df = aggregate_raw_evaluations(
            all_raw_df
        )

        if offset_complete:

            print(
                "\nGenerating OFFSET benchmark results..."
            )

            offset_averaged_df = averaged_df[
                averaged_df["drive_mode"]
                == "OFFSET"
            ]

            offset_averaged_csv_path = os.path.join(
                CONTROLLERS["offset"]["base_dir"],
                "averaged_benchmarks_rand_vs_nonrand.csv",
            )

            offset_averaged_df.to_csv(
                offset_averaged_csv_path,
                index=False,
            )

            generate_controller_plots(
                agg_df=averaged_df,
                controller_name="OFFSET",
                plot_dir=CONTROLLERS["offset"]["plot_dir"],
            )

            print(
                f"\nOFFSET averaged benchmark saved to:\n"
                f"{offset_averaged_csv_path}"
            )

            print(
                f"\nOFFSET plots saved to:\n"
                f"{CONTROLLERS['offset']['plot_dir']}"
            )

        else:

            print(
                "\nSkipping OFFSET plots because "
                "OFFSET benchmarks are incomplete."
            )

        if policy_complete:

            print(
                "\nGenerating POLICY benchmark results..."
            )

            policy_averaged_df = averaged_df[
                averaged_df["drive_mode"]
                == "POLICY"
            ]

            policy_averaged_csv_path = os.path.join(
                CONTROLLERS["policy"]["base_dir"],
                "averaged_benchmarks_rand_vs_nonrand.csv",
            )

            policy_averaged_df.to_csv(
                policy_averaged_csv_path,
                index=False,
            )

            generate_controller_plots(
                agg_df=averaged_df,
                controller_name="POLICY",
                plot_dir=CONTROLLERS["policy"]["plot_dir"],
            )

            print(
                f"\nPOLICY averaged benchmark saved to:\n"
                f"{policy_averaged_csv_path}"
            )

            print(
                f"\nPOLICY plots saved to:\n"
                f"{CONTROLLERS['policy']['plot_dir']}"
            )

        else:

            print(
                "\nSkipping POLICY plots because "
                "POLICY benchmarks are incomplete."
            )

        if offset_complete and policy_complete:

            print(
                "\nGenerating combined "
                "IK vs OFFSET vs POLICY plots..."
            )

            combined_averaged_csv_path = os.path.join(
                BASE_DIR,
                "averaged_benchmarks_all_controllers.csv",
            )

            averaged_df.to_csv(
                combined_averaged_csv_path,
                index=False,
            )

            generate_combined_plots(
                agg_df=averaged_df,
            )

            print(
                f"\nCombined averaged benchmark saved to:\n"
                f"{combined_averaged_csv_path}"
            )

            print(
                f"\nCombined IK vs OFFSET vs POLICY "
                f"plots saved to:\n"
                f"{COMBINED_PLOT_DIR}"
            )

        else:

            print(
                "\nSkipping combined IK vs OFFSET vs POLICY plots "
                "because both OFFSET and POLICY benchmark sets "
                "must be complete."
            )


    timestep_available_dfs = []

    if offset_timestep_complete:
        timestep_available_dfs.extend(
            [
                offset_rand_timestep_df,
                offset_no_rand_timestep_df,
            ]
        )

    if policy_timestep_complete:
        timestep_available_dfs.extend(
            [
                policy_rand_timestep_df,
                policy_no_rand_timestep_df,
            ]
        )

    if len(timestep_available_dfs) == 0:
        print(
            "\nSkipping timestep plots because no complete "
            "timestep benchmark set was found."
        )
        print("\nFinished.")
        return

    all_timestep_df = pd.concat(
        timestep_available_dfs,
        ignore_index=True,
    )

    averaged_timestep_df = aggregate_timestep_across_seeds(
        all_timestep_df
    )

    # Save all timestep averages
    combined_timestep_csv_path = os.path.join(
        BASE_DIR,
        "averaged_timestep_errors_all_available_controllers.csv",
    )

    averaged_timestep_df.to_csv(
        combined_timestep_csv_path,
        index=False,
    )

    print(
        f"\nAveraged timestep errors saved to:\n"
        f"{combined_timestep_csv_path}"
    )

    # Save controller-specific timestep averages
    if offset_timestep_complete:
        offset_timestep_averaged_df = averaged_timestep_df[
            averaged_timestep_df["drive_mode"] == "OFFSET"
        ]

        offset_timestep_csv_path = os.path.join(
            CONTROLLERS["offset"]["base_dir"],
            "averaged_timestep_errors_rand_vs_nonrand.csv",
        )

        offset_timestep_averaged_df.to_csv(
            offset_timestep_csv_path,
            index=False,
        )

        print(
            f"\nOFFSET averaged timestep errors saved to:\n"
            f"{offset_timestep_csv_path}"
        )

    if policy_timestep_complete:
        policy_timestep_averaged_df = averaged_timestep_df[
            averaged_timestep_df["drive_mode"] == "POLICY"
        ]

        policy_timestep_csv_path = os.path.join(
            CONTROLLERS["policy"]["base_dir"],
            "averaged_timestep_errors_rand_vs_nonrand.csv",
        )

        policy_timestep_averaged_df.to_csv(
            policy_timestep_csv_path,
            index=False,
        )

        print(
            f"\nPOLICY averaged timestep errors saved to:\n"
            f"{policy_timestep_csv_path}"
        )


    if TIMESTEP_PLOT_MODE not in [
        "combined",
        "controller",
        "both",
    ]:
        raise ValueError(
            "TIMESTEP_PLOT_MODE must be one of: "
            "'combined', 'controller', 'both'"
        )

    should_generate_controller_timestep = (
        TIMESTEP_PLOT_MODE == "controller"
        or TIMESTEP_PLOT_MODE == "both"
    )

    should_generate_combined_timestep = (
        TIMESTEP_PLOT_MODE == "combined"
        or TIMESTEP_PLOT_MODE == "both"
    )

    if should_generate_controller_timestep:

        if offset_timestep_complete:
            generate_controller_timestep_plots(
                timestep_df=averaged_timestep_df,
                controller_name="OFFSET",
                plot_dir=CONTROLLERS["offset"]["timestep_plot_dir"],
            )

            print(
                f"\nOFFSET timestep plots saved to:\n"
                f"{CONTROLLERS['offset']['timestep_plot_dir']}"
            )

        if policy_timestep_complete:
            generate_controller_timestep_plots(
                timestep_df=averaged_timestep_df,
                controller_name="POLICY",
                plot_dir=CONTROLLERS["policy"]["timestep_plot_dir"],
            )

            print(
                f"\nPOLICY timestep plots saved to:\n"
                f"{CONTROLLERS['policy']['timestep_plot_dir']}"
            )

    if should_generate_combined_timestep:

        if offset_timestep_complete and policy_timestep_complete:
            generate_combined_timestep_plots(
                timestep_df=averaged_timestep_df,
            )

            print(
                f"\nCombined timestep plots saved to:\n"
                f"{COMBINED_TIMESTEP_PLOT_DIR}"
            )

        elif offset_timestep_complete:
            print(
                "\nCombined timestep plots require OFFSET and POLICY. "
                "Only OFFSET timestep data is complete, so generating "
                "5 OFFSET timestep plots instead."
            )

            generate_controller_timestep_plots(
                timestep_df=averaged_timestep_df,
                controller_name="OFFSET",
                plot_dir=CONTROLLERS["offset"]["timestep_plot_dir"],
            )

            print(
                f"\nOFFSET timestep plots saved to:\n"
                f"{CONTROLLERS['offset']['timestep_plot_dir']}"
            )

        elif policy_timestep_complete:
            print(
                "\nCombined timestep plots require OFFSET and POLICY. "
                "Only POLICY timestep data is complete, so generating "
                "5 POLICY timestep plots instead."
            )

            generate_controller_timestep_plots(
                timestep_df=averaged_timestep_df,
                controller_name="POLICY",
                plot_dir=CONTROLLERS["policy"]["timestep_plot_dir"],
            )

            print(
                f"\nPOLICY timestep plots saved to:\n"
                f"{CONTROLLERS['policy']['timestep_plot_dir']}"
            )

    print("\nFinished.")


if __name__ == "__main__":
    main()