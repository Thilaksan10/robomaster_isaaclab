import os
import glob
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.lines import Line2D



def parse_args():
    parser = argparse.ArgumentParser()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join(
            project_root,
            "record",
            "seeded_benchmarks_edgecases",
        ),
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
CI_LOWER_COLUMN = "3d_ci_lower"
CI_UPPER_COLUMN = "3d_ci_upper"

REQUIRED_NORMAL_COLUMNS = [
    "seed",
    "randomization",
    "benchmark_type",
    "edgecase_id",
    "edgecase_name",
    "metric",
    "evaluation",
    "controller",
    "3d_mean",
]

REQUIRED_RAW_COLUMNS = [
    "seed",
    "eval_index",
    "edgecase_id",
    "edgecase_name",
    "metric",
    "evaluation",
    "controller",
    "3d",
]

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

    if condition == RANDOMIZED_LABEL:
        return f"{name} (DR)"

    return name


def metric_display_name(metric):
    metric = str(metric).upper()

    if metric == "MSE":
        return "MSE"

    return f"3D {metric}"


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

    os.makedirs(
        CONTROLLERS[controller_key]["plot_dir"],
        exist_ok=True,
    )


COMBINED_PLOT_DIR = os.path.join(
    BASE_DIR,
    "plots",
)

os.makedirs(
    COMBINED_PLOT_DIR,
    exist_ok=True,
)



CONDITION_COLORS = {
    "IK": "tab:blue",
    "NON_RAND": "tab:orange",
    "RAND": "tab:red",
}

COMBINED_COLORS = {
    "IK": "tab:blue",
    "OFFSET_NON_RAND": "tab:orange",
    "OFFSET_RAND": "tab:red",
    "POLICY_NON_RAND": "tab:green",
    "POLICY_RAND": "tab:purple",
}



def normalize_edgecase_id(value):
    if pd.isna(value):
        return "all"

    value_str = str(value)

    if value_str.lower() == "all":
        return "all"

    try:
        value_float = float(value_str)
        if value_float.is_integer():
            return str(int(value_float))
    except ValueError:
        pass

    return value_str


def normalize_edgecase_name(value):
    if pd.isna(value):
        return "all"

    value_str = str(value)

    if value_str.lower() == "all":
        return "all"

    return value_str


def sort_edgecase_ids(edgecase_ids):
    edgecase_ids = [
        normalize_edgecase_id(edgecase_id)
        for edgecase_id in edgecase_ids
    ]

    edgecase_ids = list(dict.fromkeys(edgecase_ids))

    numeric_ids = []
    other_ids = []
    has_all = False

    for edgecase_id in edgecase_ids:
        if edgecase_id == "all":
            has_all = True
        else:
            try:
                numeric_ids.append(
                    (
                        int(edgecase_id),
                        edgecase_id,
                    )
                )
            except ValueError:
                other_ids.append(edgecase_id)

    sorted_ids = [
        edgecase_id
        for _, edgecase_id in sorted(
            numeric_ids,
            key=lambda item: item[0],
        )
    ]

    sorted_ids.extend(
        sorted(other_ids)
    )

    if has_all:
        sorted_ids.append("all")

    return sorted_ids


def make_safe_filename(value):
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def get_edgecase_label(
    edgecase_id,
    edgecase_name,
):
    edgecase_id = normalize_edgecase_id(edgecase_id)
    edgecase_name = normalize_edgecase_name(edgecase_name)

    if edgecase_id == "all":
        return "Average"

    if (
        edgecase_name == "all"
        or edgecase_name == edgecase_id
    ):
        return f"Edge case {edgecase_id}"

    return edgecase_name


def validate_normal_columns(df, csv_file):
    missing_columns = [
        column
        for column in REQUIRED_NORMAL_COLUMNS
        if column not in df.columns
    ]

    if len(missing_columns) > 0:
        raise ValueError(
            f"\nCSV file is missing required columns:\n"
            f"File: {csv_file}\n"
            f"Missing: {missing_columns}\n\n"
            f"Rerun benchmark_edgecase.py first."
        )


def load_benchmark_csvs(
    directory,
    condition_name,
    drive_mode,
):
    """
    Load the normal per-seed edge-case benchmark CSV files.

    These are kept for the seed-level overall MSE figure.
    The main bar plots are generated from the raw per-edgecase CSVs.
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
        csv_file
        for csv_file in csv_files
        if "raw_per_edgecase" not in os.path.basename(csv_file)
        and "raw_per_evaluation" not in os.path.basename(csv_file)
        and "timestep_errors" not in os.path.basename(csv_file)
    ]

    if len(csv_files) == 0:
        print(
            f"[SKIP] No normal benchmark CSV files found in:\n"
            f"       {directory}"
        )
        return None

    dfs = []

    description = (
        f"Loading {drive_mode.upper()} "
        f"{condition_name} edge-case benchmarks"
    )

    for csv_file in tqdm(
        csv_files,
        desc=description,
    ):
        df = pd.read_csv(csv_file)

        validate_normal_columns(
            df,
            csv_file,
        )

        df["edgecase_id"] = df["edgecase_id"].apply(
            normalize_edgecase_id
        )
        df["edgecase_name"] = df["edgecase_name"].apply(
            normalize_edgecase_name
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



def load_raw_per_edgecase_csvs(
    directory,
    condition_name,
    drive_mode,
):
    """
    Load benchmark_<seed>_raw_per_edgecase.csv files.

    Each row contains one 3D error value for one edge case from one
    training seed. These values are pooled across seeds and used to
    calculate the main mean values and bootstrap 95% confidence intervals.
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
            "benchmark_*_raw_per_edgecase.csv",
        )
    )

    if len(csv_files) == 0:
        print(
            f"[SKIP] No raw per-edgecase CSV files found in:\n"
            f"       {directory}"
        )
        return None

    dfs = []

    description = (
        f"Loading {drive_mode.upper()} "
        f"{condition_name} raw edge-case evaluations"
    )

    for csv_file in tqdm(
        csv_files,
        desc=description,
    ):
        df = pd.read_csv(csv_file)

        missing_columns = [
            column
            for column in REQUIRED_RAW_COLUMNS
            if column not in df.columns
        ]

        if len(missing_columns) > 0:
            raise ValueError(
                f"Raw file {csv_file} is missing columns: "
                f"{missing_columns}"
            )

        df["edgecase_id"] = df["edgecase_id"].apply(
            normalize_edgecase_id
        )
        df["edgecase_name"] = df["edgecase_name"].apply(
            normalize_edgecase_name
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

def compute_pooled_mean_and_ci(
    values,
    confidence=0.95,
    n_bootstrap=10000,
    random_state=42,
):
    """
    Calculate the arithmetic mean and a percentile-bootstrap confidence
    interval from pooled non-negative per-edgecase error values.

    The returned confidence interval contains absolute lower/upper bounds,
    not a symmetric +/- value.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
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


def aggregate_raw_edgecases(df):
    """
    Pool the raw per-edgecase values from all available seeds.

    Statistics are produced for:
      - each individual edge case
      - the overall average across all edge cases

    The overall value pools all edge-case-level observations from all seeds.
    """

    rows = []

    base_group_cols = [
        "drive_mode",
        "condition",
        "metric",
        "evaluation",
        "controller",
    ]

    edgecase_group_cols = (
        base_group_cols
        + ["edgecase_id"]
    )

    groups = list(
        df.groupby(
            edgecase_group_cols,
            dropna=False,
        )
    )

    for group_values, group in tqdm(
        groups,
        desc="Pooling per-edgecase evaluations",
    ):
        row = dict(
            zip(
                edgecase_group_cols,
                group_values,
            )
        )

        names = [
            normalize_edgecase_name(name)
            for name in group["edgecase_name"].dropna().unique()
        ]

        row["edgecase_name"] = (
            names[0]
            if len(names) > 0
            else row["edgecase_id"]
        )

        values = group["3d"].values

        mean, ci = compute_pooled_mean_and_ci(
            values
        )

        row["num_seeds"] = group["seed"].nunique()
        row["num_evaluations"] = len(values)
        row[VALUE_COLUMN] = mean
        row[CI_LOWER_COLUMN] = ci[0]
        row[CI_UPPER_COLUMN] = ci[1]

        rows.append(row)

    groups = list(
        df.groupby(
            base_group_cols,
            dropna=False,
        )
    )

    for group_values, group in tqdm(
        groups,
        desc="Pooling all edge-case evaluations",
    ):
        row = dict(
            zip(
                base_group_cols,
                group_values,
            )
        )

        row["edgecase_id"] = "all"
        row["edgecase_name"] = "all"

        values = group["3d"].values

        mean, ci = compute_pooled_mean_and_ci(
            values
        )

        row["num_seeds"] = group["seed"].nunique()
        row["num_evaluations"] = len(values)
        row[VALUE_COLUMN] = mean
        row[CI_LOWER_COLUMN] = ci[0]
        row[CI_UPPER_COLUMN] = ci[1]

        rows.append(row)

    aggregated_df = pd.DataFrame(rows)

    if not aggregated_df.empty:
        aggregated_df = aggregated_df.sort_values(
            [
                "drive_mode",
                "condition",
                "metric",
                "evaluation",
                "controller",
                "edgecase_id",
            ]
        )

    return aggregated_df


def get_value(
    agg_df,
    condition,
    controller,
    edgecase_id,
    metric,
    evaluation,
    drive_mode=None,
):
    edgecase_id = normalize_edgecase_id(edgecase_id)
    controller = str(controller).upper()
    metric = str(metric).upper()

    mask = (
        (agg_df["condition"] == condition)
        & (agg_df["controller"] == controller)
        & (
            agg_df["edgecase_id"].astype(str)
            == edgecase_id
        )
        & (agg_df["metric"] == metric)
        & (agg_df["evaluation"] == evaluation)
    )

    if drive_mode is not None:
        mask = mask & (
            agg_df["drive_mode"]
            == str(drive_mode).upper()
        )

    row = agg_df[mask]

    if row.empty:
        return None, None, None

    mean = row[VALUE_COLUMN].iloc[0]
    lower = row[CI_LOWER_COLUMN].iloc[0]
    upper = row[CI_UPPER_COLUMN].iloc[0]

    yerr = (
        max(0.0, mean - lower),
        max(0.0, upper - mean),
    )

    return (
        mean,
        yerr,
        row["edgecase_name"].iloc[0],
    )


def plot_controller_metric_over_edgecases(
    plot_data,
    metric,
    evaluation,
    controller_name,
    plot_dir,
):
    display_controller = controller_display_name(
        controller_name
    )
    display_metric = metric_display_name(
        metric
    )

    edgecase_ids = list(
        plot_data.keys()
    )

    labels = [
        get_edgecase_label(
            edgecase_id,
            plot_data[edgecase_id]["edgecase_name"],
        )
        for edgecase_id in edgecase_ids
    ]

    ik_values = [
        plot_data[edgecase_id]["IK"][0]
        for edgecase_id in edgecase_ids
    ]

    non_rand_values = [
        plot_data[edgecase_id]["NON_RAND"][0]
        for edgecase_id in edgecase_ids
    ]

    rand_values = [
        plot_data[edgecase_id]["RAND"][0]
        for edgecase_id in edgecase_ids
    ]

    ik_errors = np.array(
        [
            plot_data[edgecase_id]["IK"][1]
            for edgecase_id in edgecase_ids
        ]
    ).T

    non_rand_errors = np.array(
        [
            plot_data[edgecase_id]["NON_RAND"][1]
            for edgecase_id in edgecase_ids
        ]
    ).T

    rand_errors = np.array(
        [
            plot_data[edgecase_id]["RAND"][1]
            for edgecase_id in edgecase_ids
        ]
    ).T

    x = np.arange(
        len(labels)
    )
    width = 0.25

    plt.figure(
        figsize=(
            max(
                12,
                len(labels) * 1.6,
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
        rotation=45,
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

    plt.legend()
    plt.tight_layout()

    safe_metric = metric.lower()
    safe_eval = make_safe_filename(
        evaluation
    )

    filename = (
        f"{safe_metric}_"
        f"{safe_eval}_"
        f"3d_all_edgecases.png"
    )

    plt.savefig(
        os.path.join(
            plot_dir,
            filename,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def generate_controller_plots(
    agg_df,
    controller_name,
    plot_dir,
):
    controller_df = agg_df[
        agg_df["drive_mode"]
        == controller_name
    ]

    edgecase_ids = sort_edgecase_ids(
        controller_df["edgecase_id"].unique()
    )

    metrics = sorted(
        controller_df["metric"].unique()
    )

    evaluations = sorted(
        controller_df["evaluation"].unique()
    )

    total_plots = (
        len(metrics)
        * len(evaluations)
    )

    with tqdm(
        total=total_plots,
        desc=f"Generating {controller_name} edge-case plots",
    ) as pbar:
        for metric in metrics:
            for evaluation in evaluations:
                plot_data = {}

                for edgecase_id in edgecase_ids:
                    (
                        ik_non_rand_value,
                        ik_non_rand_ci,
                        ik_non_rand_name,
                    ) = get_value(
                        agg_df=controller_df,
                        condition=NON_RANDOMIZED_LABEL,
                        controller="IK",
                        edgecase_id=edgecase_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode=controller_name,
                    )

                    (
                        ik_rand_value,
                        ik_rand_ci,
                        ik_rand_name,
                    ) = get_value(
                        agg_df=controller_df,
                        condition=RANDOMIZED_LABEL,
                        controller="IK",
                        edgecase_id=edgecase_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode=controller_name,
                    )

                    (
                        non_rand_value,
                        non_rand_ci,
                        non_rand_name,
                    ) = get_value(
                        agg_df=controller_df,
                        condition=NON_RANDOMIZED_LABEL,
                        controller=controller_name,
                        edgecase_id=edgecase_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode=controller_name,
                    )

                    (
                        rand_value,
                        rand_ci,
                        rand_name,
                    ) = get_value(
                        agg_df=controller_df,
                        condition=RANDOMIZED_LABEL,
                        controller=controller_name,
                        edgecase_id=edgecase_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode=controller_name,
                    )

                    if ik_non_rand_value is not None:
                        ik_value = ik_non_rand_value
                        ik_ci = ik_non_rand_ci
                        edgecase_name = ik_non_rand_name
                    else:
                        ik_value = ik_rand_value
                        ik_ci = ik_rand_ci
                        edgecase_name = ik_rand_name

                    if non_rand_name not in [
                        None,
                        "all",
                    ]:
                        edgecase_name = non_rand_name
                    elif rand_name not in [
                        None,
                        "all",
                    ]:
                        edgecase_name = rand_name

                    if (
                        ik_value is None
                        or non_rand_value is None
                        or rand_value is None
                    ):
                        continue

                    plot_data[edgecase_id] = {
                        "edgecase_name": edgecase_name,
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
                    plot_controller_metric_over_edgecases(
                        plot_data=plot_data,
                        metric=metric,
                        evaluation=evaluation,
                        controller_name=controller_name,
                        plot_dir=plot_dir,
                    )

                pbar.update(1)


def get_combined_ik_value(
    agg_df,
    edgecase_id,
    metric,
    evaluation,
):
    search_order = [
        (
            "OFFSET",
            NON_RANDOMIZED_LABEL,
        ),
        (
            "OFFSET",
            RANDOMIZED_LABEL,
        ),
        (
            "POLICY",
            NON_RANDOMIZED_LABEL,
        ),
        (
            "POLICY",
            RANDOMIZED_LABEL,
        ),
    ]

    for drive_mode, condition in search_order:
        value, ci, edgecase_name = get_value(
            agg_df=agg_df,
            condition=condition,
            controller="IK",
            edgecase_id=edgecase_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode=drive_mode,
        )

        if value is not None:
            return (
                value,
                ci,
                edgecase_name,
            )

    return None, None, None


def plot_combined_metric_over_edgecases(
    plot_data,
    metric,
    evaluation,
    filename_suffix="edgecases",
):
    display_metric = metric_display_name(
        metric
    )

    edgecase_ids = list(
        plot_data.keys()
    )

    labels = [
        get_edgecase_label(
            edgecase_id,
            plot_data[edgecase_id]["edgecase_name"],
        )
        for edgecase_id in edgecase_ids
    ]

    ik_values = [
        plot_data[edgecase_id]["IK"][0]
        for edgecase_id in edgecase_ids
    ]

    offset_non_rand_values = [
        plot_data[edgecase_id]["OFFSET_NON_RAND"][0]
        for edgecase_id in edgecase_ids
    ]

    offset_rand_values = [
        plot_data[edgecase_id]["OFFSET_RAND"][0]
        for edgecase_id in edgecase_ids
    ]

    policy_non_rand_values = [
        plot_data[edgecase_id]["POLICY_NON_RAND"][0]
        for edgecase_id in edgecase_ids
    ]

    policy_rand_values = [
        plot_data[edgecase_id]["POLICY_RAND"][0]
        for edgecase_id in edgecase_ids
    ]

    ik_errors = np.array(
        [
            plot_data[edgecase_id]["IK"][1]
            for edgecase_id in edgecase_ids
        ]
    ).T

    offset_non_rand_errors = np.array(
        [
            plot_data[edgecase_id]["OFFSET_NON_RAND"][1]
            for edgecase_id in edgecase_ids
        ]
    ).T

    offset_rand_errors = np.array(
        [
            plot_data[edgecase_id]["OFFSET_RAND"][1]
            for edgecase_id in edgecase_ids
        ]
    ).T

    policy_non_rand_errors = np.array(
        [
            plot_data[edgecase_id]["POLICY_NON_RAND"][1]
            for edgecase_id in edgecase_ids
        ]
    ).T

    policy_rand_errors = np.array(
        [
            plot_data[edgecase_id]["POLICY_RAND"][1]
            for edgecase_id in edgecase_ids
        ]
    ).T

    x = np.arange(
        len(labels)
    )
    width = 0.16

    plt.figure(
        figsize=(
            max(
                14,
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
        rotation=45,
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

    plt.legend()
    plt.tight_layout()

    safe_metric = metric.lower()
    safe_eval = make_safe_filename(
        evaluation
    )

    filename = (
        f"{safe_metric}_"
        f"{safe_eval}_"
        f"3d_ik_vs_offset_vs_policy_"
        f"{filename_suffix}.png"
    )

    plt.savefig(
        os.path.join(
            COMBINED_PLOT_DIR,
            filename,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def generate_combined_plots(
    agg_df,
):
    edgecase_ids = sort_edgecase_ids(
        agg_df["edgecase_id"].unique()
    )

    metrics = sorted(
        agg_df["metric"].unique()
    )

    evaluations = sorted(
        agg_df["evaluation"].unique()
    )

    total_plots = (
        len(metrics)
        * len(evaluations)
    )

    with tqdm(
        total=total_plots,
        desc="Generating combined edge-case plots",
    ) as pbar:
        for metric in metrics:
            for evaluation in evaluations:
                plot_data = {}

                for edgecase_id in edgecase_ids:
                    (
                        ik_value,
                        ik_ci,
                        ik_name,
                    ) = get_combined_ik_value(
                        agg_df=agg_df,
                        edgecase_id=edgecase_id,
                        metric=metric,
                        evaluation=evaluation,
                    )

                    (
                        offset_non_rand_value,
                        offset_non_rand_ci,
                        offset_non_rand_name,
                    ) = get_value(
                        agg_df=agg_df,
                        condition=NON_RANDOMIZED_LABEL,
                        controller="OFFSET",
                        edgecase_id=edgecase_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode="OFFSET",
                    )

                    (
                        offset_rand_value,
                        offset_rand_ci,
                        offset_rand_name,
                    ) = get_value(
                        agg_df=agg_df,
                        condition=RANDOMIZED_LABEL,
                        controller="OFFSET",
                        edgecase_id=edgecase_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode="OFFSET",
                    )

                    (
                        policy_non_rand_value,
                        policy_non_rand_ci,
                        policy_non_rand_name,
                    ) = get_value(
                        agg_df=agg_df,
                        condition=NON_RANDOMIZED_LABEL,
                        controller="POLICY",
                        edgecase_id=edgecase_id,
                        metric=metric,
                        evaluation=evaluation,
                        drive_mode="POLICY",
                    )

                    (
                        policy_rand_value,
                        policy_rand_ci,
                        policy_rand_name,
                    ) = get_value(
                        agg_df=agg_df,
                        condition=RANDOMIZED_LABEL,
                        controller="POLICY",
                        edgecase_id=edgecase_id,
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

                    edgecase_name = ik_name

                    for candidate_name in [
                        offset_non_rand_name,
                        offset_rand_name,
                        policy_non_rand_name,
                        policy_rand_name,
                    ]:
                        if candidate_name not in [
                            None,
                            "all",
                        ]:
                            edgecase_name = candidate_name
                            break

                    plot_data[edgecase_id] = {
                        "edgecase_name": edgecase_name,
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
                    plot_combined_metric_over_edgecases(
                        plot_data=plot_data,
                        metric=metric,
                        evaluation=evaluation,
                    )

                pbar.update(1)


def generate_seed_level_overall_mse_plot(df):
    """
    Show the overall MSE of every learned controller separately for each
    training seed. This uses the normal per-seed benchmark CSVs and is
    intentionally not pooled before plotting.
    """

    plot_df = df[
        (
            df["edgecase_id"]
            .astype(str)
            .apply(normalize_edgecase_id)
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


def normalize_edgecase_match_name(value):
    """
    Convert an edge-case name to a simple form that can
    be matched robustly, regardless of spaces, hyphens,
    underscores, brackets, etc.
    """

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def resolve_selected_edgecases(agg_df):
    """
    Find the four selected edge cases from their names.

    Selected cases:
        - Changing Velocities
        - Small-Radius Circle
        - Connected U-Curves (5)
        - Connected U-Curves (8)

    Returns:
        [
            (edgecase_id, display_name),
            ...
        ]
    """

    unique_edgecases = (
        agg_df[
            agg_df["edgecase_id"].astype(str) != "all"
        ][
            [
                "edgecase_id",
                "edgecase_name",
            ]
        ]
        .drop_duplicates()
    )

    selected = {
        "changing_velocities": None,
        "small_radius_circle": None,
        "connected_u_curves_5": None,
        "connected_u_curves_8": None,
    }


    for _, row in unique_edgecases.iterrows():

        edgecase_id = normalize_edgecase_id(
            row["edgecase_id"]
        )

        edgecase_name = normalize_edgecase_name(
            row["edgecase_name"]
        )

        match_name = normalize_edgecase_match_name(
            edgecase_name
        )


        if (
            "changing" in match_name
            and "veloc" in match_name
        ):
            selected[
                "changing_velocities"
            ] = edgecase_id
-

        if (
            "circle" in match_name
            and "small" in match_name
        ):
            selected[
                "small_radius_circle"
            ] = edgecase_id


        if (
            "connected" in match_name
            and "curve" in match_name
            and re.search(
                r"(?:^|_)5(?:_|$)",
                match_name,
            )
        ):
            selected[
                "connected_u_curves_5"
            ] = edgecase_id


        if (
            "connected" in match_name
            and "curve" in match_name
            and re.search(
                r"(?:^|_)8(?:_|$)",
                match_name,
            )
        ):
            selected[
                "connected_u_curves_8"
            ] = edgecase_id


    display_names = {
        "changing_velocities": (
            "Changing Velocities"
        ),
        "small_radius_circle": (
            "Small-Radius Circle"
        ),
        "connected_u_curves_5": (
            "Connected U-Curves (5)"
        ),
        "connected_u_curves_8": (
            "Connected U-Curves (8)"
        ),
    }


    resolved = []

    for key in [
        "changing_velocities",
        "small_radius_circle",
        "connected_u_curves_5",
        "connected_u_curves_8",
    ]:

        edgecase_id = selected[key]

        if edgecase_id is None:

            print(
                f"[WARNING] Could not find selected "
                f"edge case: {display_names[key]}"
            )

            continue


        resolved.append(
            (
                edgecase_id,
                display_names[key],
            )
        )


    print(
        "\nSelected thesis edge cases:"
    )

    for edgecase_id, display_name in resolved:

        print(
            f"  {edgecase_id}: "
            f"{display_name}"
        )


    return resolved


def build_combined_plot_data(
    agg_df,
    edgecase_ids,
    metric,
    evaluation,
):
    """
    Build plot_data in exactly the same format used by
    plot_combined_metric_over_edgecases().
    """

    plot_data = {}


    for edgecase_id in edgecase_ids:

        (
            ik_value,
            ik_ci,
            ik_name,
        ) = get_combined_ik_value(
            agg_df=agg_df,
            edgecase_id=edgecase_id,
            metric=metric,
            evaluation=evaluation,
        )


        (
            offset_non_rand_value,
            offset_non_rand_ci,
            offset_non_rand_name,
        ) = get_value(
            agg_df=agg_df,
            condition=NON_RANDOMIZED_LABEL,
            controller="OFFSET",
            edgecase_id=edgecase_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode="OFFSET",
        )


        (
            offset_rand_value,
            offset_rand_ci,
            offset_rand_name,
        ) = get_value(
            agg_df=agg_df,
            condition=RANDOMIZED_LABEL,
            controller="OFFSET",
            edgecase_id=edgecase_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode="OFFSET",
        )


        (
            policy_non_rand_value,
            policy_non_rand_ci,
            policy_non_rand_name,
        ) = get_value(
            agg_df=agg_df,
            condition=NON_RANDOMIZED_LABEL,
            controller="POLICY",
            edgecase_id=edgecase_id,
            metric=metric,
            evaluation=evaluation,
            drive_mode="POLICY",
        )


        (
            policy_rand_value,
            policy_rand_ci,
            policy_rand_name,
        ) = get_value(
            agg_df=agg_df,
            condition=RANDOMIZED_LABEL,
            controller="POLICY",
            edgecase_id=edgecase_id,
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


        edgecase_name = ik_name

        for candidate_name in [
            offset_non_rand_name,
            offset_rand_name,
            policy_non_rand_name,
            policy_rand_name,
        ]:

            if candidate_name not in [
                None,
                "all",
            ]:
                edgecase_name = candidate_name
                break


        plot_data[edgecase_id] = {

            "edgecase_name": edgecase_name,

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


    return plot_data


def generate_additional_thesis_plots(
    agg_df,
    raw_df,
):
    """
    Generate additional compact edge-case plots:

    1. Overall average across all edge cases
    2. Four selected edge cases + their selected average

    Generated independently for every metric and evaluation.
    """

    selected_edgecases = (
        resolve_selected_edgecases(
            agg_df
        )
    )


    metrics = sorted(
        agg_df["metric"].unique()
    )

    evaluations = sorted(
        agg_df["evaluation"].unique()
    )


    total_plots = (
        len(metrics)
        * len(evaluations)
        * 2
    )


    with tqdm(
        total=total_plots,
        desc="Generating additional thesis plots",
    ) as pbar:


        for metric in metrics:

            for evaluation in evaluations:

                overall_plot_data = (
                    build_combined_plot_data(
                        agg_df=agg_df,
                        edgecase_ids=[
                            "all"
                        ],
                        metric=metric,
                        evaluation=evaluation,
                    )
                )


                if len(overall_plot_data) > 0:

                    plot_combined_metric_over_edgecases(
                        plot_data=overall_plot_data,
                        metric=metric,
                        evaluation=evaluation,
                        filename_suffix="overall",
                    )

                pbar.update(1)


                selected_ids = [
                    edgecase_id
                    for (
                        edgecase_id,
                        _
                    ) in selected_edgecases
                ]


                selected_raw_df = raw_df[
                    raw_df["edgecase_id"]
                    .astype(str)
                    .isin(selected_ids)
                ].copy()


                selected_agg_df = aggregate_raw_edgecases(
                    selected_raw_df
                )


                selected_plot_data = build_combined_plot_data(
                    agg_df=selected_agg_df,
                    edgecase_ids=(
                        selected_ids
                        + ["all"]
                    ),
                    metric=metric,
                    evaluation=evaluation,
                )

                for (
                    edgecase_id,
                    display_name,
                ) in selected_edgecases:

                    if (
                        edgecase_id
                        in selected_plot_data
                    ):

                        selected_plot_data[
                            edgecase_id
                        ][
                            "edgecase_name"
                        ] = display_name

               
                if "all" in selected_plot_data:

                    selected_plot_data[
                        "selected_average"
                    ] = selected_plot_data.pop(
                        "all"
                    )

                    selected_plot_data[
                        "selected_average"
                    ][
                        "edgecase_name"
                    ] = "Selected Average"


                if len(selected_plot_data) > 0:

                    plot_combined_metric_over_edgecases(
                        plot_data=selected_plot_data,
                        metric=metric,
                        evaluation=evaluation,
                        filename_suffix=(
                            "selected_edgecases_with_average"
                        ),
                    )

                pbar.update(1)


def main():
    print("\n----------------------------------")
    print("SEEDED EDGE-CASE BENCHMARK")
    print("----------------------------------")
    print(f"Base directory: {BASE_DIR}")
    print("----------------------------------\n")


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

    offset_complete = (
        offset_rand_df is not None
        and offset_no_rand_df is not None
    )

    policy_complete = (
        policy_rand_df is not None
        and policy_no_rand_df is not None
    )


    offset_rand_raw_df = load_raw_per_edgecase_csvs(
        directory=CONTROLLERS["offset"]["rand_dir"],
        condition_name=RANDOMIZED_LABEL,
        drive_mode="OFFSET",
    )

    offset_no_rand_raw_df = load_raw_per_edgecase_csvs(
        directory=CONTROLLERS["offset"]["no_rand_dir"],
        condition_name=NON_RANDOMIZED_LABEL,
        drive_mode="OFFSET",
    )

    policy_rand_raw_df = load_raw_per_edgecase_csvs(
        directory=CONTROLLERS["policy"]["rand_dir"],
        condition_name=RANDOMIZED_LABEL,
        drive_mode="POLICY",
    )

    policy_no_rand_raw_df = load_raw_per_edgecase_csvs(
        directory=CONTROLLERS["policy"]["no_rand_dir"],
        condition_name=NON_RANDOMIZED_LABEL,
        drive_mode="POLICY",
    )

    offset_raw_complete = (
        offset_rand_raw_df is not None
        and offset_no_rand_raw_df is not None
    )

    policy_raw_complete = (
        policy_rand_raw_df is not None
        and policy_no_rand_raw_df is not None
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
        f"OFFSET raw RAND:          "
        f"{'AVAILABLE' if offset_rand_raw_df is not None else 'MISSING'}"
    )
    print(
        f"OFFSET raw NON_RAND:      "
        f"{'AVAILABLE' if offset_no_rand_raw_df is not None else 'MISSING'}"
    )
    print(
        f"POLICY raw RAND:          "
        f"{'AVAILABLE' if policy_rand_raw_df is not None else 'MISSING'}"
    )
    print(
        f"POLICY raw NON_RAND:      "
        f"{'AVAILABLE' if policy_no_rand_raw_df is not None else 'MISSING'}"
    )
    print()
    print(
        f"OFFSET plots:             "
        f"{'WILL BE GENERATED' if offset_complete and offset_raw_complete else 'SKIPPED'}"
    )
    print(
        f"POLICY plots:             "
        f"{'WILL BE GENERATED' if policy_complete and policy_raw_complete else 'SKIPPED'}"
    )
    print(
        f"Combined plots:           "
        f"{'WILL BE GENERATED' if offset_complete and policy_complete and offset_raw_complete and policy_raw_complete else 'SKIPPED'}"
    )
    print("----------------------------------\n")

    if not offset_raw_complete and not policy_raw_complete:
        raise RuntimeError(
            "No complete raw per-edgecase benchmark set was found. "
            "Rerun benchmark_edgecase.py so that "
            "benchmark_<seed>_raw_per_edgecase.csv exists for each seed."
        )


    normal_dfs = []

    if offset_complete:
        normal_dfs.extend(
            [
                offset_rand_df,
                offset_no_rand_df,
            ]
        )

    if policy_complete:
        normal_dfs.extend(
            [
                policy_rand_df,
                policy_no_rand_df,
            ]
        )

    if (
        offset_complete
        and policy_complete
        and len(normal_dfs) > 0
    ):
        all_normal_df = pd.concat(
            normal_dfs,
            ignore_index=True,
        )

        print(
            "\nGenerating seed-level overall MSE plot..."
        )

        generate_seed_level_overall_mse_plot(
            all_normal_df
        )


    raw_dfs = []

    if offset_raw_complete:
        raw_dfs.extend(
            [
                offset_rand_raw_df,
                offset_no_rand_raw_df,
            ]
        )

    if policy_raw_complete:
        raw_dfs.extend(
            [
                policy_rand_raw_df,
                policy_no_rand_raw_df,
            ]
        )

    raw_dfs = [
        df
        for df in raw_dfs
        if df is not None
    ]

    if len(raw_dfs) == 0:
        raise RuntimeError(
            "No raw per-edgecase benchmark files were found."
        )

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
            "edgecase_id",
            "edgecase_name",
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

    averaged_df = aggregate_raw_edgecases(
        all_raw_df
    )


    if offset_complete and offset_raw_complete:
        print(
            "\nGenerating Hybrid Offset edge-case benchmark results..."
        )

        offset_averaged_df = averaged_df[
            averaged_df["drive_mode"]
            == "OFFSET"
        ]

        offset_averaged_csv_path = os.path.join(
            CONTROLLERS["offset"]["base_dir"],
            "averaged_edgecase_benchmarks_rand_vs_nonrand.csv",
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
            f"\nHybrid Offset averaged edge-case benchmark saved to:\n"
            f"{offset_averaged_csv_path}"
        )

        print(
            f"\nHybrid Offset edge-case plots saved to:\n"
            f"{CONTROLLERS['offset']['plot_dir']}"
        )
    else:
        print(
            "\nSkipping Hybrid Offset plots because "
            "the benchmark/raw data is incomplete."
        )


    if policy_complete and policy_raw_complete:
        print(
            "\nGenerating Fully Learned edge-case benchmark results..."
        )

        policy_averaged_df = averaged_df[
            averaged_df["drive_mode"]
            == "POLICY"
        ]

        policy_averaged_csv_path = os.path.join(
            CONTROLLERS["policy"]["base_dir"],
            "averaged_edgecase_benchmarks_rand_vs_nonrand.csv",
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
            f"\nFully Learned averaged edge-case benchmark saved to:\n"
            f"{policy_averaged_csv_path}"
        )

        print(
            f"\nFully Learned edge-case plots saved to:\n"
            f"{CONTROLLERS['policy']['plot_dir']}"
        )
    else:
        print(
            "\nSkipping Fully Learned plots because "
            "the benchmark/raw data is incomplete."
        )


    if (
        offset_complete
        and policy_complete
        and offset_raw_complete
        and policy_raw_complete
    ):
        print(
            "\nGenerating combined IK vs Hybrid Offset "
            "vs Fully Learned edge-case plots..."
        )

        combined_averaged_csv_path = os.path.join(
            BASE_DIR,
            "averaged_edgecase_benchmarks_all_controllers.csv",
        )

        averaged_df.to_csv(
            combined_averaged_csv_path,
            index=False,
        )

        generate_combined_plots(
            agg_df=averaged_df,
        )

        print(
            f"\nCombined averaged edge-case benchmark saved to:\n"
            f"{combined_averaged_csv_path}"
        )

        print(
            f"\nCombined edge-case plots saved to:\n"
            f"{COMBINED_PLOT_DIR}"
        )

        print(
            "\nGenerating additional thesis-focused "
            "edge-case plots..."
        )

        generate_additional_thesis_plots(
            agg_df=averaged_df,
            raw_df=all_raw_df,
        )
    else:
        print(
            "\nSkipping combined plots because OFFSET and POLICY "
            "normal/raw benchmark sets must both be complete."
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()
