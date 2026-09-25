import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




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

RANDOMIZED_LABEL = "RAND"
NON_RANDOMIZED_LABEL = "NON_RAND"
VALUE_COLUMN = "3d_mean"

RACETRACK_CSV = "averaged_benchmarks_all_controllers.csv"
EDGECASE_CSV = "averaged_edgecase_benchmarks_all_controllers.csv"



def project_root_from_script():
    """
    The analysis scripts are normally placed in <project>/components.
    In that case the project root is one directory above this file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def default_evaluation_root():
    """
    All benchmark CSVs used by this analysis live below:

        <project_root>/evaluation/non_optimized

    Nothing is loaded from the record/ directory.
    """
    return os.path.join(
        project_root_from_script(),
        "evaluation",
        "non_optimized",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the relative MSE change caused by randomized "
            "physics parameters during evaluation."
        )
    )

    parser.add_argument(
        "--evaluation-root",
        type=str,
        default=default_evaluation_root(),
        help=(
            "Root directory containing the non-optimized evaluation CSVs. "
            "Default: <project>/evaluation/non_optimized"
        ),
    )

    parser.add_argument(
        "--racetrack-nominal-csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--racetrack-randomized-csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--edgecase-nominal-csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--edgecase-randomized-csv",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def find_csvs_recursive(root, filename):
    """Find all files with the requested basename below root."""
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Evaluation directory does not exist:\n  {root}"
        )

    matches = []

    for current_root, _, files in os.walk(root):
        if filename in files:
            matches.append(
                os.path.join(current_root, filename)
            )

    return sorted(matches)


def path_tokens(path):
    """Return lower-case path components with '-' normalized to '_'."""
    normalized = os.path.normpath(path).replace("-", "_").lower()
    return [part for part in normalized.split(os.sep) if part]


def classify_evaluation_csv(path):
    """
    Classify a summary CSV as nominal or randomized evaluation from its path.

    Training randomization folders such as offset/rand and policy/rand are not
    used here because the combined summary CSV is stored above those folders.
    """
    tokens = path_tokens(os.path.dirname(path))

    nominal_tokens = {
        "no_rand",
        "non_rand",
        "non_randomized",
        "without_randomization",
        "without_rand",
        "nominal",
        "seeded_benchmark_no_rand",
        "seeded_benchmarks_no_rand",
        "evaluation_no_rand",
    }

    randomized_tokens = {
        "rand",
        "randomized",
        "with_randomization",
        "with_rand",
        "seeded_benchmark_rand",
        "seeded_benchmarks_rand",
        "evaluation_rand",
    }


    if any(token in nominal_tokens for token in tokens):
        return "nominal"

    if any(token in randomized_tokens for token in tokens):
        return "randomized"

    directory_text = "/".join(tokens)

    nominal_fragments = [
        "no_rand",
        "non_rand",
        "non_randomized",
        "without_rand",
        "nominal",
    ]

    randomized_fragments = [
        "randomized_eval",
        "rand_eval",
        "evaluation_rand",
        "seeded_benchmark_rand",
        "seeded_benchmarks_rand",
    ]

    if any(fragment in directory_text for fragment in nominal_fragments):
        return "nominal"

    if any(fragment in directory_text for fragment in randomized_fragments):
        return "randomized"

    return None


def resolve_csv_pair(evaluation_root, filename, nominal_override, randomized_override, label):
    """
    Resolve one nominal/randomized evaluation CSV pair.

    Explicit command-line paths take priority. Otherwise every matching CSV is
    searched below <project>/evaluation/non_optimized and classified by the
    directory names that describe the evaluation condition.
    """
    if nominal_override is not None and randomized_override is not None:
        nominal = os.path.abspath(nominal_override)
        randomized = os.path.abspath(randomized_override)

        for path in [nominal, randomized]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"CSV does not exist:\n  {path}")

        return nominal, randomized

    matches = find_csvs_recursive(
        evaluation_root,
        filename,
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {label} summary CSV named '{filename}' was found below:\n"
            f"  {os.path.abspath(evaluation_root)}"
        )

    nominal_candidates = []
    randomized_candidates = []

    for path in matches:
        classification = classify_evaluation_csv(path)

        if classification == "nominal":
            nominal_candidates.append(path)
        elif classification == "randomized":
            randomized_candidates.append(path)

    if nominal_override is not None:
        nominal_candidates = [os.path.abspath(nominal_override)]

    if randomized_override is not None:
        randomized_candidates = [os.path.abspath(randomized_override)]

    if len(nominal_candidates) != 1 or len(randomized_candidates) != 1:
        candidate_text = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Could not uniquely determine the nominal and randomized {label} CSVs.\n\n"
            f"Search root:\n  {os.path.abspath(evaluation_root)}\n\n"
            f"Found candidates:\n{candidate_text}\n\n"
            f"Please pass the two paths explicitly with the corresponding "
            f"--{label}-nominal-csv and --{label}-randomized-csv arguments."
        )

    return nominal_candidates[0], randomized_candidates[0]


def normalize_common_columns(df):
    df = df.copy()

    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.upper()

    if "evaluation" in df.columns:
        df["evaluation"] = df["evaluation"].astype(str)

    if "controller" in df.columns:
        df["controller"] = df["controller"].astype(str).str.upper()

    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str).str.upper()

    if "drive_mode" in df.columns:
        df["drive_mode"] = df["drive_mode"].astype(str).str.upper()

    if "racetrack_id" in df.columns:
        df["racetrack_id"] = (
            df["racetrack_id"]
            .astype(str)
            .str.lower()
            .replace({"nan": "all"})
        )

    if "edgecase_id" in df.columns:
        df["edgecase_id"] = (
            df["edgecase_id"]
            .astype(str)
            .str.lower()
            .replace({"nan": "all"})
        )

    return df


def load_combined_csv(csv_path, benchmark_name):
    csv_path = os.path.abspath(csv_path)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Could not find the combined {benchmark_name} CSV:\n"
            f"  {csv_path}"
        )

    print(f"  {benchmark_name:22s}: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_common_columns(df)

    required = [
        "condition",
        "drive_mode",
        "metric",
        "evaluation",
        "controller",
        VALUE_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\n{csv_path} is missing required columns:\n"
            f"  {missing}"
        )

    return df


def filter_overall(df, benchmark_type, evaluation):
    mask = (
        (df["metric"] == "MSE")
        & (df["evaluation"] == evaluation)
    )

    if benchmark_type == "racetrack":
        if "racetrack_id" not in df.columns:
            raise ValueError("Racetrack CSV has no 'racetrack_id' column.")
        mask &= (
            df["racetrack_id"].astype(str).str.lower() == "all"
        )

    elif benchmark_type == "edgecase":
        if "edgecase_id" not in df.columns:
            raise ValueError("Edge Case CSV has no 'edgecase_id' column.")
        mask &= (
            df["edgecase_id"].astype(str).str.lower() == "all"
        )

    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    return df[mask].copy()


def first_value(df):
    if df.empty:
        return None

    values = (
        pd.to_numeric(df[VALUE_COLUMN], errors="coerce")
        .dropna()
        .values
    )

    if len(values) == 0:
        return None

    return float(values[0])


def get_ik_value(df):
    """
    Reproduce the preferred IK lookup order used in the existing scripts.
    IK can occur multiple times because it is stored alongside both learned
    controller families and both training-randomization conditions.
    """
    search_order = [
        ("OFFSET", NON_RANDOMIZED_LABEL),
        ("OFFSET", RANDOMIZED_LABEL),
        ("POLICY", NON_RANDOMIZED_LABEL),
        ("POLICY", RANDOMIZED_LABEL),
    ]

    for drive_mode, condition in search_order:
        subset = df[
            (df["controller"] == "IK")
            & (df["drive_mode"] == drive_mode)
            & (df["condition"] == condition)
        ]

        value = first_value(subset)
        if value is not None:
            return value

    return first_value(df[df["controller"] == "IK"])


def get_learned_value(df, controller, condition, drive_mode):
    subset = df[
        (df["controller"] == controller)
        & (df["condition"] == condition)
        & (df["drive_mode"] == drive_mode)
    ]

    return first_value(subset)


def get_controller_values(df, benchmark_type, evaluation):
    overall = filter_overall(
        df=df,
        benchmark_type=benchmark_type,
        evaluation=evaluation,
    )

    values = {
        "IK": get_ik_value(overall),
        "OFFSET_NON_RAND": get_learned_value(
            overall,
            controller="OFFSET",
            condition=NON_RANDOMIZED_LABEL,
            drive_mode="OFFSET",
        ),
        "OFFSET_RAND": get_learned_value(
            overall,
            controller="OFFSET",
            condition=RANDOMIZED_LABEL,
            drive_mode="OFFSET",
        ),
        "POLICY_NON_RAND": get_learned_value(
            overall,
            controller="POLICY",
            condition=NON_RANDOMIZED_LABEL,
            drive_mode="POLICY",
        ),
        "POLICY_RAND": get_learned_value(
            overall,
            controller="POLICY",
            condition=RANDOMIZED_LABEL,
            drive_mode="POLICY",
        ),
    }

    missing = [
        controller
        for controller, value in values.items()
        if value is None
    ]

    if missing:
        raise RuntimeError(
            f"\nMissing overall values for {benchmark_type} / {evaluation}:\n"
            f"  {missing}\n"
        )

    return values


def compute_relative_change(nominal_values, randomized_values):
    """
    Relative change caused by randomized evaluation physics.

        (randomized - nominal) / nominal * 100

    Positive -> tracking error increased (worse).
    Negative -> tracking error decreased.
    """
    result = {}

    for controller in CONTROLLERS:
        nominal = nominal_values[controller]
        randomized = randomized_values[controller]

        if np.isclose(nominal, 0.0):
            result[controller] = np.nan
        else:
            result[controller] = (
                (randomized - nominal)
                / nominal
                * 100.0
            )

    return result



def add_value_labels(ax, bars, values):
    """
    Add percentage labels while keeping them inside the axes.

    plot_panel() reserves vertical space before this function is called,
    so labels above positive bars and below negative bars stay visible.
    """
    y_min, y_max = ax.get_ylim()
    axis_span = max(y_max - y_min, 1.0)
    offset = axis_span * 0.018

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue

        x = bar.get_x() + bar.get_width() / 2.0

        if value >= 0:
            y = value + offset
            va = "bottom"
        else:
            y = value - offset
            va = "top"

        ax.text(
            x,
            y,
            f"{value:+.1f}%",
            ha="center",
            va=va,
            fontsize=8,
            clip_on=True,
        )


def set_panel_ylim(ax, values):
    """
    Reserve enough vertical room for percentage labels.

    Matplotlib's automatic scaling only considers the bars, not text labels.
    Without explicit padding, labels on the tallest/lowest bars can therefore
    extend beyond the plotting area.
    """
    finite_values = [
        float(value)
        for value in values
        if np.isfinite(value)
    ]

    if not finite_values:
        ax.set_ylim(-1.0, 1.0)
        return

    data_min = min(0.0, min(finite_values)) + 1
    data_max = max(0.0, max(finite_values)) + 1

    data_span = data_max - data_min
    if np.isclose(data_span, 0.0):
        data_span = max(abs(data_max), abs(data_min), 1.0)

    top_padding = 0.14 * data_span
    bottom_padding = 0.12 * data_span

    top_padding = max(top_padding, 1.5)
    bottom_padding = max(bottom_padding, 1.5)

    ax.set_ylim(
        data_min - bottom_padding,
        data_max + top_padding,
    )


def plot_panel(ax, changes, title):
    x = np.arange(len(CONTROLLERS))

    values = [
        changes[controller]
        for controller in CONTROLLERS
    ]

    colors = [
        CONTROLLER_COLORS[controller]
        for controller in CONTROLLERS
    ]

    bars = ax.bar(
        x,
        values,
        color=colors,
        width=0.72,
    )

    ax.axhline(0.0, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONTROLLER_LABELS[controller] for controller in CONTROLLERS],
        fontsize=8,
    )

    ax.set_ylabel("Relative MSE change [%]")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    set_panel_ylim(
        ax=ax,
        values=values,
    )

    add_value_labels(
        ax=ax,
        bars=bars,
        values=values,
    )


def make_figure(results, output_dir, dpi):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    plot_panel(
        axes[0, 0],
        results["racetrack_position"],
        "Racetrack Position MSE",
    )
    plot_panel(
        axes[0, 1],
        results["racetrack_velocity"],
        "Racetrack Velocity MSE",
    )
    plot_panel(
        axes[1, 0],
        results["edgecase_position"],
        "Edge Case Position MSE",
    )
    plot_panel(
        axes[1, 1],
        results["edgecase_velocity"],
        "Edge Case Velocity MSE",
    )

    fig.suptitle(
        "Sensitivity to Randomized Evaluation Physics",
        fontsize=14,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = os.path.join(
        output_dir,
        "domain_randomization_relative_mse_change.png",
    )
    pdf_path = os.path.join(
        output_dir,
        "domain_randomization_relative_mse_change.pdf",
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


# --------------------------------------------------
# CSV OUTPUT
# --------------------------------------------------

def save_summary_csv(results, output_dir):
    rows = []

    panel_info = [
        ("Racetrack", "Position", results["racetrack_position"]),
        ("Racetrack", "Velocity", results["racetrack_velocity"]),
        ("Edge Case", "Position", results["edgecase_position"]),
        ("Edge Case", "Velocity", results["edgecase_velocity"]),
    ]

    for benchmark, evaluation, changes in panel_info:
        for controller in CONTROLLERS:
            rows.append(
                {
                    "benchmark": benchmark,
                    "evaluation": evaluation,
                    "controller": CONTROLLER_LABELS[controller].replace("\n", " "),
                    "relative_mse_change_percent": changes[controller],
                }
            )

    summary_df = pd.DataFrame(rows)

    csv_path = os.path.join(
        output_dir,
        "domain_randomization_relative_mse_change.csv",
    )

    summary_df.to_csv(csv_path, index=False)

    return csv_path


# --------------------------------------------------
# PRINTING
# --------------------------------------------------

def print_changes(results):
    print()
    print("=" * 90)
    print("RELATIVE MSE CHANGE UNDER RANDOMIZED EVALUATION PHYSICS")
    print("Positive = error increased | Negative = error decreased")
    print("=" * 90)

    for name, values in [
        ("Racetrack Position", results["racetrack_position"]),
        ("Racetrack Velocity", results["racetrack_velocity"]),
        ("Edge Case Position", results["edgecase_position"]),
        ("Edge Case Velocity", results["edgecase_velocity"]),
    ]:
        print()
        print(name)
        print("-" * len(name))

        for controller in CONTROLLERS:
            label = CONTROLLER_LABELS[controller].replace("\n", " ")
            value = values[controller]

            print(
                f"{label:25s}: "
                f"{value:+8.2f} %"
            )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    args = parse_args()

    evaluation_root = os.path.abspath(args.evaluation_root)

    racetrack_nominal_csv, racetrack_randomized_csv = resolve_csv_pair(
        evaluation_root=evaluation_root,
        filename=RACETRACK_CSV,
        nominal_override=args.racetrack_nominal_csv,
        randomized_override=args.racetrack_randomized_csv,
        label="racetrack",
    )

    edgecase_nominal_csv, edgecase_randomized_csv = resolve_csv_pair(
        evaluation_root=evaluation_root,
        filename=EDGECASE_CSV,
        nominal_override=args.edgecase_nominal_csv,
        randomized_override=args.edgecase_randomized_csv,
        label="edgecase",
    )

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir is not None
        else os.path.join(
            evaluation_root,
            "domain_randomization_analysis",
        )
    )

    print("\nEvaluation root:")
    print(f"  {evaluation_root}")
    print("\nResolved CSV files:")

    racetrack_nominal_df = load_combined_csv(
        racetrack_nominal_csv,
        "racetrack nominal",
    )
    racetrack_randomized_df = load_combined_csv(
        racetrack_randomized_csv,
        "racetrack randomized",
    )
    edgecase_nominal_df = load_combined_csv(
        edgecase_nominal_csv,
        "Edge Case nominal",
    )
    edgecase_randomized_df = load_combined_csv(
        edgecase_randomized_csv,
        "Edge Case randomized",
    )

    racetrack_position_nominal = get_controller_values(
        racetrack_nominal_df,
        benchmark_type="racetrack",
        evaluation="absolute_position",
    )
    racetrack_position_randomized = get_controller_values(
        racetrack_randomized_df,
        benchmark_type="racetrack",
        evaluation="absolute_position",
    )
    racetrack_velocity_nominal = get_controller_values(
        racetrack_nominal_df,
        benchmark_type="racetrack",
        evaluation="velocity",
    )
    racetrack_velocity_randomized = get_controller_values(
        racetrack_randomized_df,
        benchmark_type="racetrack",
        evaluation="velocity",
    )

    edgecase_position_nominal = get_controller_values(
        edgecase_nominal_df,
        benchmark_type="edgecase",
        evaluation="absolute_position",
    )
    edgecase_position_randomized = get_controller_values(
        edgecase_randomized_df,
        benchmark_type="edgecase",
        evaluation="absolute_position",
    )
    edgecase_velocity_nominal = get_controller_values(
        edgecase_nominal_df,
        benchmark_type="edgecase",
        evaluation="velocity",
    )
    edgecase_velocity_randomized = get_controller_values(
        edgecase_randomized_df,
        benchmark_type="edgecase",
        evaluation="velocity",
    )

    results = {
        "racetrack_position": compute_relative_change(
            racetrack_position_nominal,
            racetrack_position_randomized,
        ),
        "racetrack_velocity": compute_relative_change(
            racetrack_velocity_nominal,
            racetrack_velocity_randomized,
        ),
        "edgecase_position": compute_relative_change(
            edgecase_position_nominal,
            edgecase_position_randomized,
        ),
        "edgecase_velocity": compute_relative_change(
            edgecase_velocity_nominal,
            edgecase_velocity_randomized,
        ),
    }

    print_changes(results)

    png_path, pdf_path = make_figure(
        results=results,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    csv_path = save_summary_csv(
        results=results,
        output_dir=output_dir,
    )

    print()
    print("Saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
