import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


CSV_NAME = "averaged_benchmarks_all_controllers.csv"

VALUE_COLUMN = "3d_mean"
CI_LOWER_COLUMN = "3d_ci_lower"
CI_UPPER_COLUMN = "3d_ci_upper"

NON_RANDOMIZED_LABEL = "NON_RAND"
RANDOMIZED_LABEL = "RAND"

CONTROLLERS = [
    "IK",
    "OFFSET_NON_RAND",
    "OFFSET_RAND",
    "POLICY_NON_RAND",
    "POLICY_RAND",
]

CONTROLLER_LABELS = {
    "IK": "IK",
    "OFFSET_NON_RAND": "Hybrid\nOffset",
    "OFFSET_RAND": "Hybrid Offset\n(DR)",
    "POLICY_NON_RAND": "Fully\nLearned",
    "POLICY_RAND": "Fully Learned\n(DR)",
}

CONTROLLER_COLORS = {
    "IK": "tab:blue",
    "OFFSET_NON_RAND": "tab:orange",
    "OFFSET_RAND": "tab:red",
    "POLICY_NON_RAND": "tab:green",
    "POLICY_RAND": "tab:purple",
}


def project_root_from_script():
    """
    Script is normally stored inside <project_root>/components/.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def parse_args():
    project_root = project_root_from_script()

    parser = argparse.ArgumentParser(
        description=(
            "Compare racetrack MSE between the non-optimized and optimized "
            "simulation models using grouped bars."
        )
    )

    parser.add_argument(
        "--non-optimized-root",
        type=str,
        default=os.path.join(
            project_root,
            "evaluation",
            "non_optimized",
        ),
    )

    parser.add_argument(
        "--optimized-root",
        type=str,
        default=os.path.join(
            project_root,
            "evaluation",
            "optimized",
        ),
    )

    parser.add_argument(
        "--evaluation-condition",
        choices=["no_rand", "rand", "both"],
        default="no_rand",
        help=(
            "Evaluation physics condition. "
            "'no_rand' is recommended for the simulation-model comparison."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            project_root,
            "evaluation",
            "simulation_model_analysis",
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def find_model_csv(model_root, evaluation_condition):
    model_root = os.path.abspath(model_root)

    if not os.path.isdir(model_root):
        raise FileNotFoundError(
            f"Simulation-model evaluation directory does not exist:\n"
            f"  {model_root}"
        )

    target_eval_dir = (
        "seeded_benchmark_no_rand"
        if evaluation_condition == "no_rand"
        else "seeded_benchmark_rand"
    )

    matches = []

    for current_root, _, files in os.walk(model_root):
        if CSV_NAME not in files:
            continue

        parts = [
            part.lower()
            for part in os.path.normpath(current_root).split(os.sep)
            if part
        ]

        if "racetrack" not in parts:
            continue

        if target_eval_dir not in parts:
            continue

        matches.append(
            os.path.join(current_root, CSV_NAME)
        )

    matches = sorted(matches)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"  {CSV_NAME}\n\n"
            f"below:\n"
            f"  {model_root}\n\n"
            f"for evaluation condition:\n"
            f"  {target_eval_dir}\n"
        )

    if len(matches) > 1:
        candidates = "\n".join(
            f"  - {path}"
            for path in matches
        )
        raise RuntimeError(
            f"\nMultiple matching CSVs found below:\n"
            f"  {model_root}\n\n"
            f"{candidates}\n"
        )

    return matches[0]


def normalize_dataframe(df):
    df = df.copy()

    for column in [
        "metric",
        "evaluation",
        "controller",
        "condition",
        "drive_mode",
    ]:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.upper()
            )

    if "racetrack_id" in df.columns:
        df["racetrack_id"] = (
            df["racetrack_id"]
            .astype(str)
            .str.lower()
            .replace({"nan": "all"})
        )

    return df


def load_summary_csv(path):
    df = pd.read_csv(path)
    df = normalize_dataframe(df)

    required_columns = [
        "condition",
        "drive_mode",
        "metric",
        "evaluation",
        "controller",
        "racetrack_id",
        VALUE_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nCSV is missing required columns:\n"
            f"  {path}\n\n"
            f"Missing:\n"
            f"  {missing}\n"
        )

    return df



def select_overall_rows(df, evaluation):
    return df[
        (df["metric"] == "MSE")
        & (df["evaluation"] == evaluation.upper())
        & (df["racetrack_id"] == "all")
    ].copy()


def controller_mask(df, controller_key):
    if controller_key == "IK":
        return df["controller"] == "IK"

    if controller_key == "OFFSET_NON_RAND":
        return (
            (df["drive_mode"] == "OFFSET")
            & (df["controller"] == "OFFSET")
            & (df["condition"] == NON_RANDOMIZED_LABEL)
        )

    if controller_key == "OFFSET_RAND":
        return (
            (df["drive_mode"] == "OFFSET")
            & (df["controller"] == "OFFSET")
            & (df["condition"] == RANDOMIZED_LABEL)
        )

    if controller_key == "POLICY_NON_RAND":
        return (
            (df["drive_mode"] == "POLICY")
            & (df["controller"] == "POLICY")
            & (df["condition"] == NON_RANDOMIZED_LABEL)
        )

    if controller_key == "POLICY_RAND":
        return (
            (df["drive_mode"] == "POLICY")
            & (df["controller"] == "POLICY")
            & (df["condition"] == RANDOMIZED_LABEL)
        )

    raise ValueError(
        f"Unknown controller key: {controller_key}"
    )


def collapse_rows(rows, controller_key, source_path):
    """
    Collapse duplicate rows if present.

    This is mainly useful for IK, which may appear multiple times in a combined
    benchmark CSV because it is shared across controller groups.
    """
    if rows.empty:
        raise RuntimeError(
            f"\nNo row found for controller:\n"
            f"  {CONTROLLER_LABELS[controller_key]}\n\n"
            f"CSV:\n"
            f"  {source_path}\n"
        )

    means = pd.to_numeric(
        rows[VALUE_COLUMN],
        errors="coerce",
    ).dropna()

    if len(means) == 0:
        raise RuntimeError(
            f"No numeric '{VALUE_COLUMN}' value found for "
            f"{CONTROLLER_LABELS[controller_key]}"
        )

    result = {
        "mean": float(means.mean()),
        "ci_lower": np.nan,
        "ci_upper": np.nan,
    }

    if (
        CI_LOWER_COLUMN in rows.columns
        and CI_UPPER_COLUMN in rows.columns
    ):
        lower = pd.to_numeric(
            rows[CI_LOWER_COLUMN],
            errors="coerce",
        ).dropna()

        upper = pd.to_numeric(
            rows[CI_UPPER_COLUMN],
            errors="coerce",
        ).dropna()

        if len(lower) > 0:
            result["ci_lower"] = float(lower.mean())

        if len(upper) > 0:
            result["ci_upper"] = float(upper.mean())

    return result


def extract_metric(df, evaluation, source_path):
    overall = select_overall_rows(
        df=df,
        evaluation=evaluation,
    )

    values = {}

    for controller_key in CONTROLLERS:
        rows = overall[
            controller_mask(
                overall,
                controller_key,
            )
        ]

        values[controller_key] = collapse_rows(
            rows=rows,
            controller_key=controller_key,
            source_path=source_path,
        )

    return values


def error_size(result):
    mean = result["mean"]
    lower = result["ci_lower"]
    upper = result["ci_upper"]

    if not (
        np.isfinite(lower)
        and np.isfinite(upper)
    ):
        return None

    return np.array([
        [max(0.0, mean - lower)],
        [max(0.0, upper - mean)],
    ])


def relative_change(non_optimized, optimized):
    if np.isclose(non_optimized, 0.0):
        return np.nan

    return (
        (optimized - non_optimized)
        / non_optimized
        * 100.0
    )


def set_axis_limits(ax, all_values, all_cis, percentage_positions):
    """
    Add enough vertical space for error bars and percentage labels.
    """
    values = list(all_values)

    for lower, upper in all_cis:
        if np.isfinite(lower):
            values.append(lower)
        if np.isfinite(upper):
            values.append(upper)

    values.extend(percentage_positions)

    finite = [
        value
        for value in values
        if np.isfinite(value)
    ]

    if not finite:
        return

    maximum = max(finite)

    if maximum <= 0:
        maximum = 1.0

    ax.set_ylim(
        0,
        maximum * 1.16,
    )


def plot_grouped_model_comparison(
    ax,
    non_optimized_values,
    optimized_values,
    title,
):
    x = np.arange(len(CONTROLLERS), dtype=float)

    bar_width = 0.32
    left_offset = -bar_width / 2
    right_offset = bar_width / 2

    all_values = []
    all_cis = []
    percentage_positions = []

    raw_max = 0.0

    for controller_key in CONTROLLERS:
        non_opt = non_optimized_values[controller_key]
        opt = optimized_values[controller_key]

        raw_max = max(
            raw_max,
            non_opt["mean"],
            opt["mean"],
        )

        if np.isfinite(non_opt["ci_upper"]):
            raw_max = max(
                raw_max,
                non_opt["ci_upper"],
            )

        if np.isfinite(opt["ci_upper"]):
            raw_max = max(
                raw_max,
                opt["ci_upper"],
            )

    if raw_max <= 0:
        raw_max = 1.0

    percentage_offset = raw_max * 0.035

    for i, controller_key in enumerate(CONTROLLERS):
        color = CONTROLLER_COLORS[controller_key]

        non_opt = non_optimized_values[controller_key]
        opt = optimized_values[controller_key]

        x_non = x[i] + left_offset
        x_opt = x[i] + right_offset

        y_non = non_opt["mean"]
        y_opt = opt["mean"]

        non_yerr = error_size(non_opt)
        opt_yerr = error_size(opt)

        ax.bar(
            x_non,
            y_non,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=1.1,
            yerr=non_yerr,
            capsize=3 if non_yerr is not None else 0,
            error_kw={
                "elinewidth": 1.0,
                "capthick": 1.0,
            },
            zorder=2,
        )

        ax.bar(
            x_opt,
            y_opt,
            width=bar_width,
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            hatch="///",
            yerr=opt_yerr,
            capsize=3 if opt_yerr is not None else 0,
            error_kw={
                "ecolor": color,
                "elinewidth": 1.0,
                "capthick": 1.0,
            },
            zorder=2,
        )

        ax.plot(
            [x_non, x_opt],
            [y_non, y_opt],
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=3.5,
            zorder=4,
        )

        change = relative_change(
            y_non,
            y_opt,
        )

        midpoint_x = (x_non + x_opt) / 2
        midpoint_y = (y_non + y_opt) / 2

        if y_opt >= y_non:
            label_y = midpoint_y + percentage_offset
            vertical_alignment = "bottom"
        else:
            label_y = midpoint_y + percentage_offset
            vertical_alignment = "bottom"

        if np.isfinite(change):
            label = f"{change:+.1f}%"
        else:
            label = "n/a"

        ax.text(
            midpoint_x,
            label_y,
            label,
            ha="center",
            va=vertical_alignment,
            fontsize=9,
            fontweight="bold",
            color=color,
            zorder=5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
                "pad": 0.7,
            },
        )

        all_values.extend([
            y_non,
            y_opt,
        ])

        all_cis.extend([
            (
                non_opt["ci_lower"],
                non_opt["ci_upper"],
            ),
            (
                opt["ci_lower"],
                opt["ci_upper"],
            ),
        ])

        percentage_positions.append(
            label_y + percentage_offset
        )

    set_axis_limits(
        ax=ax,
        all_values=all_values,
        all_cis=all_cis,
        percentage_positions=percentage_positions,
    )

    ax.set_xticks(
        x,
        [
            CONTROLLER_LABELS[key]
            for key in CONTROLLERS
        ],
    )

    ax.set_ylabel("MSE")
    ax.set_title(title)

    ax.grid(
        axis="y",
        alpha=0.25,
        zorder=0,
    )


def make_figure(
    non_optimized_position,
    optimized_position,
    non_optimized_velocity,
    optimized_velocity,
    output_dir,
    evaluation_condition,
    dpi,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
    )

    plot_grouped_model_comparison(
        ax=axes[0],
        non_optimized_values=non_optimized_position,
        optimized_values=optimized_position,
        title="Position MSE",
    )

    plot_grouped_model_comparison(
        ax=axes[1],
        non_optimized_values=non_optimized_velocity,
        optimized_values=optimized_velocity,
        title="Velocity MSE",
    )

    model_legend = [
        Patch(
            facecolor="0.55",
            edgecolor="0.55",
            label="Non-optimized",
        ),
        Patch(
            facecolor="white",
            edgecolor="0.35",
            hatch="///",
            linewidth=1.3,
            label="Optimized",
        ),
    ]

    fig.legend(
        handles=model_legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(
        rect=(0, 0.08, 1, 1),
    )

    suffix = (
        "no_rand_eval"
        if evaluation_condition == "no_rand"
        else "rand_eval"
    )

    png_path = os.path.join(
        output_dir,
        f"racetrack_simulation_model_grouped_{suffix}.png",
    )

    pdf_path = os.path.join(
        output_dir,
        f"racetrack_simulation_model_grouped_{suffix}.pdf",
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png_path, pdf_path


def save_summary(
    non_optimized_position,
    optimized_position,
    non_optimized_velocity,
    optimized_velocity,
    output_dir,
    evaluation_condition,
):
    rows = []

    for metric_name, non_opt_data, opt_data in [
        (
            "absolute_position",
            non_optimized_position,
            optimized_position,
        ),
        (
            "velocity",
            non_optimized_velocity,
            optimized_velocity,
        ),
    ]:
        for controller_key in CONTROLLERS:
            non_opt = non_opt_data[controller_key]
            opt = opt_data[controller_key]

            rows.append({
                "evaluation_condition": evaluation_condition,
                "metric": metric_name,
                "controller": CONTROLLER_LABELS[
                    controller_key
                ].replace("\n", " "),
                "non_optimized_mse": non_opt["mean"],
                "optimized_mse": opt["mean"],
                "relative_change_percent": relative_change(
                    non_opt["mean"],
                    opt["mean"],
                ),
            })

    summary = pd.DataFrame(rows)

    suffix = (
        "no_rand_eval"
        if evaluation_condition == "no_rand"
        else "rand_eval"
    )

    output_path = os.path.join(
        output_dir,
        f"racetrack_simulation_model_grouped_{suffix}.csv",
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    return output_path, summary


def process_condition(
    non_optimized_root,
    optimized_root,
    output_dir,
    evaluation_condition,
    dpi,
):
    non_optimized_csv = find_model_csv(
        non_optimized_root,
        evaluation_condition,
    )

    optimized_csv = find_model_csv(
        optimized_root,
        evaluation_condition,
    )

    print("\n" + "=" * 90)
    print(
        f"EVALUATION CONDITION: "
        f"{evaluation_condition.upper()}"
    )
    print("=" * 90)

    print("\nNon-optimized:")
    print(f"  {non_optimized_csv}")

    print("\nOptimized:")
    print(f"  {optimized_csv}")

    non_optimized_df = load_summary_csv(
        non_optimized_csv
    )

    optimized_df = load_summary_csv(
        optimized_csv
    )

    non_optimized_position = extract_metric(
        non_optimized_df,
        "absolute_position",
        non_optimized_csv,
    )

    optimized_position = extract_metric(
        optimized_df,
        "absolute_position",
        optimized_csv,
    )

    non_optimized_velocity = extract_metric(
        non_optimized_df,
        "velocity",
        non_optimized_csv,
    )

    optimized_velocity = extract_metric(
        optimized_df,
        "velocity",
        optimized_csv,
    )

    png_path, pdf_path = make_figure(
        non_optimized_position,
        optimized_position,
        non_optimized_velocity,
        optimized_velocity,
        output_dir,
        evaluation_condition,
        dpi,
    )

    csv_path, summary = save_summary(
        non_optimized_position,
        optimized_position,
        non_optimized_velocity,
        optimized_velocity,
        output_dir,
        evaluation_condition,
    )

    print("\nSaved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"  CSV: {csv_path}")

    print("\nChanges from non-optimized -> optimized:\n")

    for _, row in summary.iterrows():
        print(
            f"{row['metric']:18s} | "
            f"{row['controller']:24s} | "
            f"{row['non_optimized_mse']:9.4f} -> "
            f"{row['optimized_mse']:9.4f} | "
            f"{row['relative_change_percent']:+8.2f}%"
        )


def main():
    args = parse_args()

    if args.evaluation_condition == "both":
        conditions = [
            "no_rand",
            "rand",
        ]
    else:
        conditions = [
            args.evaluation_condition,
        ]

    for condition in conditions:
        process_condition(
            non_optimized_root=args.non_optimized_root,
            optimized_root=args.optimized_root,
            output_dir=args.output_dir,
            evaluation_condition=condition,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()