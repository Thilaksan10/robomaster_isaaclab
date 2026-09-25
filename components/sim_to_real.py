import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import stats



MAX_STEPS = 100
DPI = 300
CONFIDENCE_LEVEL = 0.95


SIM_IK_REPETITIONS = 15
SIM_LEARNED_REPETITIONS_PER_SEED = 5

TRAJECTORIES = [
    "changing_velocities",
    "circle_small_radius",
    "connected_u_curves_5",
    "connected_u_curves_8",
]

TRAJECTORY_LABELS = {
    "changing_velocities":
        "Changing Velocities",

    "circle_small_radius":
        "Circle Small Radius",

    "connected_u_curves_5":
        "Connected U-Curves (5)",

    "connected_u_curves_8":
        "Connected U-Curves (8)",
}

CONTROLLERS = [
    "IK",
    "Hybrid Offset",
    "Hybrid Offset (DR)",
    "Fully Learned",
    "Fully Learned (DR)",
]

LEARNED_CONTROLLERS = [
    "Hybrid Offset",
    "Hybrid Offset (DR)",
    "Fully Learned",
    "Fully Learned (DR)",
]

ARCHITECTURES = [
    (
        "Hybrid Offset",
        "Hybrid Offset (DR)",
        "Hybrid Offset",
    ),
    (
        "Fully Learned",
        "Fully Learned (DR)",
        "Fully Learned",
    ),
]

CONTROLLER_COLORS = {
    "IK":
        "tab:blue",

    "Hybrid Offset":
        "tab:orange",

    "Hybrid Offset (DR)":
        "tab:red",

    "Fully Learned":
        "tab:green",

    "Fully Learned (DR)":
        "tab:purple",
}

REAL_REQUIRED_FILES = {
    "actual_pos":
        "actual_positions.csv",

    "reference_pos":
        "reference_positions.csv",

    "actual_vel":
        "actual_velocities.csv",

    "reference_vel":
        "reference_velocities.csv",
}

SIM_RAW_PATTERN = (
    "benchmark_*_raw_per_edgecase.csv"
)

SIM_REQUIRED_COLUMNS = {
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
}



def detect_project_root():
    """
    Try to locate the project root automatically.

    Normal expected placement:
        <project>/components/sim_to_real.py
    """
    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    cwd = os.getcwd()

    candidates = [
        script_dir,
        os.path.dirname(script_dir),
        cwd,
        os.path.dirname(cwd),
    ]

    seen = set()

    for candidate in candidates:
        candidate = os.path.abspath(
            candidate
        )

        if candidate in seen:
            continue

        seen.add(
            candidate
        )

        if os.path.isdir(
            os.path.join(
                candidate,
                "evaluation",
            )
        ):
            return candidate

    # Fallback for components/
    return os.path.dirname(
        script_dir
    )


def parse_args():
    project_root = (
        detect_project_root()
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate matched Sim-to-Real thesis figures."
        )
    )

    parser.add_argument(
        "--evaluation-root",
        default=os.path.join(
            project_root,
            "evaluation",
        ),
        help=(
            "Directory containing real_non_opt, real_opt, "
            "non_optimized, and optimized."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Default: <evaluation-root>/sim_to_real"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=(
            "Number of physical-robot control steps used."
        ),
    )

    parser.add_argument(
        "--sim-evaluation-condition",
        choices=[
            "no_rand",
            "rand",
        ],
        default="no_rand",
        help=(
            "Which simulation evaluation environment to use. "
            "For direct comparison with the physical robot, "
            "normally use no_rand."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Show figures interactively in addition to saving them."
        ),
    )

    return parser.parse_args()


def normalize_text(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_edgecase_name(
    value,
):
    """
    Convert any supported physical/simulation spelling to one canonical key.
    """
    text = normalize_text(
        value
    )

    text = (
        text
        .replace("(", "")
        .replace(")", "")
    )

    if (
        "changing_velocities" in text
        or "changing_velocity" in text
        or (
            "changing" in text
            and "veloc" in text
        )
    ):
        return "changing_velocities"

    if (
        "circle_small_radius" in text
        or "small_radius_circle" in text
        or (
            "circle" in text
            and "small" in text
            and "radius" in text
        )
    ):
        return "circle_small_radius"

    if (
        "connected_u_curves_5" in text
        or "connected_u_curve_5" in text
        or (
            "connected" in text
            and "curve" in text
            and re.search(
                r"(?:^|_)5(?:_|$)",
                text,
            )
        )
    ):
        return "connected_u_curves_5"

    if (
        "connected_u_curves_8" in text
        or "connected_u_curve_8" in text
        or (
            "connected" in text
            and "curve" in text
            and re.search(
                r"(?:^|_)8(?:_|$)",
                text,
            )
        )
    ):
        return "connected_u_curves_8"

    return None


def controller_tick_labels(
    controllers,
):
    mapping = {
        "IK":
            "IK",

        "Hybrid Offset":
            "Hybrid\nOffset",

        "Hybrid Offset (DR)":
            "Hybrid Offset\n(DR)",

        "Fully Learned":
            "Fully\nLearned",

        "Fully Learned (DR)":
            "Fully Learned\n(DR)",
    }

    return [
        mapping[
            controller
        ]
        for controller
        in controllers
    ]


def metric_ylabel(
    metric,
):
    if metric == "position":
        return "Position MSE"

    if metric == "velocity":
        return "Velocity MSE"

    raise ValueError(
        f"Unknown metric: {metric}"
    )


def relative_change(
    old,
    new,
):
    if (
        not np.isfinite(old)
        or not np.isfinite(new)
        or np.isclose(
            old,
            0.0,
        )
    ):
        return np.nan

    return (
        (new - old)
        / old
        * 100.0
    )


def mean_and_t_ci(
    values,
    confidence=CONFIDENCE_LEVEL,
):
    """
    Mean and two-sided Student-t confidence interval.

    Values are the execution-level MSE values used in the matched comparison.

    Physical robot:
        the actual repeated executions are used directly.

    Deterministic simulation:
        IK is repeated 15 times and each learned seed is repeated 5 times
        to mirror the physical execution structure.

    The returned interval is:
        mean +/- t_critical * SEM
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    n = len(
        values
    )

    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean = float(
        np.mean(
            values
        )
    )

    if n < 2:
        return (
            mean,
            mean,
            mean,
        )

    sem = stats.sem(
        values,
        ddof=1,
    )

    alpha = (
        1.0
        - confidence
    )

    t_crit = stats.t.ppf(
        1.0
        - alpha / 2.0,
        df=n - 1,
    )

    half_width = float(
        t_crit
        * sem
    )

    return (
        mean,
        mean - half_width,
        mean + half_width,
    )


def ci_yerr(
    stat,
):
    if stat is None:
        return np.array(
            [
                [0.0],
                [0.0],
            ]
        )

    mean = stat.get(
        "mean",
        np.nan,
    )

    lower = stat.get(
        "lower",
        np.nan,
    )

    upper = stat.get(
        "upper",
        np.nan,
    )

    if not (
        np.isfinite(mean)
        and np.isfinite(lower)
        and np.isfinite(upper)
    ):
        return np.array(
            [
                [0.0],
                [0.0],
            ]
        )

    return np.array(
        [
            [
                max(
                    0.0,
                    mean - lower,
                )
            ],
            [
                max(
                    0.0,
                    upper - mean,
                )
            ],
        ]
    )


def infer_real_controller(
    relative_path,
):
    """
    Infer physical controller from path.

    Supports names such as:
        ik
        offset
        offset_rand
        policy
        policy_rand
        *_true
        *_false
        hybrid...
        fully...
    """
    text = normalize_text(
        os.path.normpath(
            relative_path
        )
    )

    parts = [
        part
        for part
        in re.split(
            r"[/\\]+",
            text,
        )
        if part
    ]

    # IK
    if any(
        part == "ik"
        or part.startswith(
            "ik_"
        )
        or part.endswith(
            "_ik"
        )
        for part
        in parts
    ):
        return "IK"

    # Architecture
    if (
        "offset" in text
        or "hybrid" in text
    ):
        base = (
            "Hybrid Offset"
        )

    elif (
        "full" in text
        or "policy" in text
        or "fully_learned" in text
        or "fullylearned" in text
        or "full_policy" in text
    ):
        base = (
            "Fully Learned"
        )

    else:
        return None

    # non-DR indicators
    if (
        "no_rand" in text
        or "non_rand" in text
        or "nonrandom" in text
        or "false" in text
    ):
        return base

    # DR indicators.
    if (
        "true" in text
        or "randomized" in text
        or "_rand" in text
        or "/rand/" in text
        or text.endswith(
            "/rand"
        )
        or "_dr" in text
    ):
        return (
            f"{base} (DR)"
        )

    return base


def discover_real_runs(
    root,
):
    root = os.path.abspath(
        root
    )

    if not os.path.isdir(
        root
    ):
        raise FileNotFoundError(
            "Physical-robot directory does not exist:\n"
            f"  {root}"
        )

    required_files = set(
        REAL_REQUIRED_FILES.values()
    )

    runs = []

    for (
        current_root,
        dirs,
        files,
    ) in os.walk(
        root
    ):
        dirs[:] = [
            directory
            for directory
            in dirs
            if directory.lower()
            not in {
                "plots",
                "plot",
                "sim_to_real",
            }
        ]

        if not required_files.issubset(
            set(files)
        ):
            continue

        relative = os.path.relpath(
            current_root,
            root,
        )

        trajectory = (
            normalize_edgecase_name(
                relative
            )
        )

        controller = (
            infer_real_controller(
                relative
            )
        )

        if trajectory is None:
            print(
                "[WARNING] Could not infer physical trajectory from:"
            )

            print(
                f"          {relative}"
            )

            continue

        if controller is None:
            print(
                "[WARNING] Could not infer physical controller from:"
            )

            print(
                f"          {relative}"
            )

            continue

        runs.append(
            {
                "path":
                    current_root,

                "relative_path":
                    relative,

                "trajectory":
                    trajectory,

                "controller":
                    controller,
            }
        )

    return runs


def load_real_run(
    meta,
    max_steps,
):
    paths = {
        key:
            os.path.join(
                meta[
                    "path"
                ],
                filename,
            )
        for (
            key,
            filename,
        )
        in REAL_REQUIRED_FILES.items()
    }

    try:
        actual_pos = np.atleast_2d(
            np.loadtxt(
                paths[
                    "actual_pos"
                ],
                delimiter=",",
            )
        )

        reference_pos = np.atleast_2d(
            np.loadtxt(
                paths[
                    "reference_pos"
                ],
                delimiter=",",
            )
        )

        actual_vel = np.atleast_2d(
            np.loadtxt(
                paths[
                    "actual_vel"
                ],
                delimiter=",",
            )
        )

        reference_vel = np.atleast_2d(
            np.loadtxt(
                paths[
                    "reference_vel"
                ],
                delimiter=",",
            )
        )

    except Exception as exc:
        print(
            "[WARNING] Could not load physical run:"
        )

        print(
            f"          {meta['path']}"
        )

        print(
            f"          {exc}"
        )

        return None

    if (
        actual_pos.shape[1] < 2
        or reference_pos.shape[1] < 2
    ):
        print(
            "[WARNING] Physical position data needs at least 2 columns:"
        )

        print(
            f"          {meta['path']}"
        )

        return None

    if (
        actual_vel.shape[1] < 3
        or reference_vel.shape[1] < 3
    ):
        print(
            "[WARNING] Physical velocity data needs at least 3 columns:"
        )

        print(
            f"          {meta['path']}"
        )

        return None

    n_pos = min(
        max_steps,
        len(
            actual_pos
        ),
        len(
            reference_pos
        ),
    )

    n_vel = min(
        max_steps,
        len(
            actual_vel
        ),
        len(
            reference_vel
        ),
    )

    if (
        n_pos == 0
        or n_vel == 0
    ):
        return None


    pos_diff = (
        reference_pos[
            :n_pos,
            :2,
        ]
        - actual_pos[
            :n_pos,
            :2,
        ]
    )

    vel_diff = (
        reference_vel[
            :n_vel,
            :3,
        ]
        - actual_vel[
            :n_vel,
            :3,
        ]
    )

    position_mse = float(
        np.mean(
            np.sum(
                pos_diff ** 2,
                axis=1,
            )
        )
    )

    velocity_mse = float(
        np.mean(
            np.sum(
                vel_diff ** 2,
                axis=1,
            )
        )
    )

    result = dict(
        meta
    )

    result.update(
        {
            "position":
                position_mse,

            "velocity":
                velocity_mse,

            "n_position_steps":
                n_pos,

            "n_velocity_steps":
                n_vel,

            "source":
                "real",
        }
    )

    return result


def load_real_dataset(
    root,
    label,
    max_steps,
):
    print(
        "\n"
        + "=" * 80
    )

    print(
        label
    )

    print(
        os.path.abspath(
            root
        )
    )

    print(
        "=" * 80
    )

    discovered = (
        discover_real_runs(
            root
        )
    )

    if len(
        discovered
    ) == 0:
        raise RuntimeError(
            "No physical-robot runs were found below:\n"
            f"  {root}\n\n"
            "Expected directories containing:\n"
            "  actual_positions.csv\n"
            "  reference_positions.csv\n"
            "  actual_velocities.csv\n"
            "  reference_velocities.csv"
        )

    runs = []

    for meta in discovered:
        loaded = (
            load_real_run(
                meta,
                max_steps=max_steps,
            )
        )

        if loaded is not None:
            runs.append(
                loaded
            )

    print(
        f"Loaded physical runs: {len(runs)}"
    )

    print_record_counts(
        runs
    )

    return runs


def simulation_evaluation_folder(
    condition,
):
    if condition == "no_rand":
        return (
            "seeded_benchmark_no_rand"
        )

    return (
        "seeded_benchmark_rand"
    )


def simulation_controller_label(
    drive_mode,
    training_condition,
):
    if drive_mode == "offset":
        base = (
            "Hybrid Offset"
        )

    elif drive_mode == "policy":
        base = (
            "Fully Learned"
        )

    else:
        raise ValueError(
            f"Unknown drive mode: {drive_mode}"
        )

    if training_condition == "rand":
        return (
            f"{base} (DR)"
        )

    return base


def find_raw_per_edgecase_csvs(
    branch,
):
    pattern = os.path.join(
        branch,
        "**",
        SIM_RAW_PATTERN,
    )

    return sorted(
        set(
            glob.glob(
                pattern,
                recursive=True,
            )
        )
    )


def validate_sim_csv(
    df,
    path,
):
    missing = sorted(
        SIM_REQUIRED_COLUMNS
        - set(
            df.columns
        )
    )

    if len(
        missing
    ) > 0:
        raise ValueError(
            "Simulation raw-per-edgecase CSV is missing required columns:\n"
            f"  {path}\n"
            f"Missing: {missing}"
        )


def prepare_sim_dataframe(
    df,
):
    df = df.copy()

    df[
        "metric"
    ] = (
        df[
            "metric"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.upper()
    )

    df[
        "evaluation"
    ] = (
        df[
            "evaluation"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.lower()
    )

    df[
        "controller"
    ] = (
        df[
            "controller"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.upper()
    )

    df[
        "benchmark_type"
    ] = (
        df[
            "benchmark_type"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.lower()
    )

    df[
        "trajectory_key"
    ] = (
        df[
            "edgecase_name"
        ]
        .apply(
            normalize_edgecase_name
        )
    )

    return df


def simulation_rows_to_records(
    df,
    source_file,
    learned_controller_label,
    raw_learned_controller,
    include_ik,
):
    """
    Convert the simulation benchmark rows to the same scalar-MSE record
    structure used for the physical runs.

    For simulation MSE:
        stored x = mean(dx^2)
        stored y = mean(dy^2)
        stored z = mean(dz^2) / mean(domega^2), depending on evaluation

    Therefore:
        physical-compatible position MSE = x + y
        physical-compatible velocity MSE = x + y + z
    """
    df = (
        prepare_sim_dataframe(
            df
        )
    )

    df = df[
        (
            df[
                "benchmark_type"
            ]
            == "edgecase"
        )
        & (
            df[
                "metric"
            ]
            == "MSE"
        )
        & (
            df[
                "trajectory_key"
            ]
            .isin(
                TRAJECTORIES
            )
        )
    ].copy()

    records = []

    controllers_to_load = [
        (
            raw_learned_controller,
            learned_controller_label,
        )
    ]

    if include_ik:
        controllers_to_load.append(
            (
                "IK",
                "IK",
            )
        )

    for (
        raw_controller,
        output_controller,
    ) in controllers_to_load:

        controller_df = df[
            df[
                "controller"
            ]
            == raw_controller.upper()
        ].copy()

        if len(
            controller_df
        ) == 0:
            continue

        group_columns = [
            "seed",
            "eval_index",
            "edgecase_id",
            "trajectory_key",
        ]

        for (
            group_key,
            group,
        ) in controller_df.groupby(
            group_columns,
            dropna=False,
        ):
            (
                seed,
                eval_index,
                edgecase_id,
                trajectory,
            ) = group_key

            position_rows = group[
                group[
                    "evaluation"
                ]
                == "absolute_position"
            ]

            velocity_rows = group[
                group[
                    "evaluation"
                ]
                == "velocity"
            ]

            if (
                len(
                    position_rows
                )
                == 0
                or len(
                    velocity_rows
                )
                == 0
            ):
                continue

            pos_x = float(
                position_rows[
                    "x"
                ]
                .astype(
                    float
                )
                .mean()
            )

            pos_y = float(
                position_rows[
                    "y"
                ]
                .astype(
                    float
                )
                .mean()
            )

            vel_x = float(
                velocity_rows[
                    "x"
                ]
                .astype(
                    float
                )
                .mean()
            )

            vel_y = float(
                velocity_rows[
                    "y"
                ]
                .astype(
                    float
                )
                .mean()
            )

            vel_z = float(
                velocity_rows[
                    "z"
                ]
                .astype(
                    float
                )
                .mean()
            )

            position_mse = (
                pos_x
                + pos_y
            )

            velocity_mse = (
                vel_x
                + vel_y
                + vel_z
            )

            records.append(
                {
                    "trajectory":
                        trajectory,

                    "controller":
                        output_controller,

                    "position":
                        position_mse,

                    "velocity":
                        velocity_mse,

                    "seed":
                        seed,

                    "eval_index":
                        eval_index,

                    "edgecase_id":
                        edgecase_id,

                    "source_file":
                        source_file,

                    "source":
                        "simulation",
                }
            )

    return records


def load_sim_branch(
    branch,
    drive_mode,
    training_condition,
    include_ik,
):
    learned_label = (
        simulation_controller_label(
            drive_mode,
            training_condition,
        )
    )

    raw_learned_controller = (
        "OFFSET"
        if drive_mode == "offset"
        else "POLICY"
    )

    csv_files = (
        find_raw_per_edgecase_csvs(
            branch
        )
    )

    print(
        f"  {os.path.relpath(branch)}"
    )

    print(
        f"    raw_per_edgecase files: "
        f"{len(csv_files)}"
    )

    if len(
        csv_files
    ) == 0:
        return []

    records = []

    for csv_file in csv_files:
        df = pd.read_csv(
            csv_file
        )

        validate_sim_csv(
            df,
            csv_file,
        )

        records.extend(
            simulation_rows_to_records(
                df=df,
                source_file=csv_file,
                learned_controller_label=learned_label,
                raw_learned_controller=raw_learned_controller,
                include_ik=False,
            )
        )


    if include_ik:
        ik_file = (
            csv_files[0]
        )

        ik_df = pd.read_csv(
            ik_file
        )

        validate_sim_csv(
            ik_df,
            ik_file,
        )

        records.extend(
            simulation_rows_to_records(
                df=ik_df,
                source_file=ik_file,
                learned_controller_label=learned_label,
                raw_learned_controller=raw_learned_controller,
                include_ik=True,
            )
        )

        records = [
            record
            for record
            in records
            if not (
                record[
                    "source_file"
                ]
                == ik_file
                and record[
                    "controller"
                ]
                == learned_label
                and sum(
                    1
                    for candidate
                    in records
                    if (
                        candidate[
                            "source_file"
                        ]
                        == ik_file
                        and candidate[
                            "controller"
                        ]
                        == learned_label
                        and candidate[
                            "trajectory"
                        ]
                        == record[
                            "trajectory"
                        ]
                        and candidate[
                            "seed"
                        ]
                        == record[
                            "seed"
                        ]
                        and candidate[
                            "eval_index"
                        ]
                        == record[
                            "eval_index"
                        ]
                    )
                )
                > 1
            )
        ]

    return records


def load_sim_branch_clean(
    branch,
    drive_mode,
    training_condition,
    include_ik,
):
    """
    Cleaner wrapper around the raw-per-edgecase branch loading.

    Learned controller:
        all raw-per-edgecase seed files

    IK:
        only the first raw-per-edgecase file from offset/no_rand
    """
    learned_label = (
        simulation_controller_label(
            drive_mode,
            training_condition,
        )
    )

    raw_learned_controller = (
        "OFFSET"
        if drive_mode == "offset"
        else "POLICY"
    )

    csv_files = (
        find_raw_per_edgecase_csvs(
            branch
        )
    )

    print(
        f"  {os.path.relpath(branch)}"
    )

    print(
        f"    raw_per_edgecase files: "
        f"{len(csv_files)}"
    )

    if len(
        csv_files
    ) == 0:
        return []

    records = []

    for csv_file in csv_files:
        df = pd.read_csv(
            csv_file
        )

        validate_sim_csv(
            df,
            csv_file,
        )

        branch_records = (
            simulation_rows_to_records(
                df=df,
                source_file=csv_file,
                learned_controller_label=learned_label,
                raw_learned_controller=raw_learned_controller,
                include_ik=False,
            )
        )

        records.extend(
            branch_records
        )

    if include_ik:
        ik_file = (
            csv_files[0]
        )

        df = pd.read_csv(
            ik_file
        )

        validate_sim_csv(
            df,
            ik_file,
        )

        prepared = (
            prepare_sim_dataframe(
                df
            )
        )

        prepared = prepared[
            (
                prepared[
                    "benchmark_type"
                ]
                == "edgecase"
            )
            & (
                prepared[
                    "metric"
                ]
                == "MSE"
            )
            & (
                prepared[
                    "controller"
                ]
                == "IK"
            )
            & (
                prepared[
                    "trajectory_key"
                ]
                .isin(
                    TRAJECTORIES
                )
            )
        ].copy()

        group_columns = [
            "seed",
            "eval_index",
            "edgecase_id",
            "trajectory_key",
        ]

        for (
            group_key,
            group,
        ) in prepared.groupby(
            group_columns,
            dropna=False,
        ):
            (
                seed,
                eval_index,
                edgecase_id,
                trajectory,
            ) = group_key

            position_rows = group[
                group[
                    "evaluation"
                ]
                == "absolute_position"
            ]

            velocity_rows = group[
                group[
                    "evaluation"
                ]
                == "velocity"
            ]

            if (
                len(
                    position_rows
                )
                == 0
                or len(
                    velocity_rows
                )
                == 0
            ):
                continue

            pos_x = float(
                position_rows[
                    "x"
                ]
                .astype(
                    float
                )
                .mean()
            )

            pos_y = float(
                position_rows[
                    "y"
                ]
                .astype(
                    float
                )
                .mean()
            )

            vel_x = float(
                velocity_rows[
                    "x"
                ]
                .astype(
                    float
                )
                .mean()
            )

            vel_y = float(
                velocity_rows[
                    "y"
                ]
                .astype(
                    float
                )
                .mean()
            )

            vel_z = float(
                velocity_rows[
                    "z"
                ]
                .astype(
                    float
                )
                .mean()
            )

            records.append(
                {
                    "trajectory":
                        trajectory,

                    "controller":
                        "IK",

                    "position":
                        pos_x
                        + pos_y,

                    "velocity":
                        vel_x
                        + vel_y
                        + vel_z,

                    "seed":
                        seed,

                    "eval_index":
                        eval_index,

                    "edgecase_id":
                        edgecase_id,

                    "source_file":
                        ik_file,

                    "source":
                        "simulation",
                }
            )

    return records



def replicate_simulation_executions(
    records,
):
    """
    Expand deterministic simulation values to the same execution counts used
    during the physical evaluation.

    IK:
        one deterministic simulation result per trajectory is repeated
        SIM_IK_REPETITIONS times.

    Learned controllers:
        every seed/trajectory result is repeated
        SIM_LEARNED_REPETITIONS_PER_SEED times.

    The numerical mean is unchanged. The purpose is to represent the same
    execution structure as the physical evaluation when plotting descriptive
    standard deviations.
    """
    expanded = []

    for record in records:
        if record[
            "controller"
        ] == "IK":
            repetitions = (
                SIM_IK_REPETITIONS
            )
        else:
            repetitions = (
                SIM_LEARNED_REPETITIONS_PER_SEED
            )

        for execution_index in range(
            repetitions
        ):
            copied = dict(
                record
            )

            copied[
                "execution_index"
            ] = execution_index

            copied[
                "simulation_repetition_factor"
            ] = repetitions

            expanded.append(
                copied
            )

    return expanded


def load_sim_dataset(
    model_root,
    label,
    evaluation_condition,
):
    print(
        "\n"
        + "=" * 80
    )

    print(
        label
    )

    print(
        os.path.abspath(
            model_root
        )
    )

    print(
        "=" * 80
    )

    eval_folder = (
        simulation_evaluation_folder(
            evaluation_condition
        )
    )

    base = os.path.join(
        os.path.abspath(
            model_root
        ),
        "edgecase_100",
        eval_folder,
    )

    if not os.path.isdir(
        base
    ):
        raise RuntimeError(
            "Could not find simulation edge-case benchmark root:\n"
            f"  {base}\n\n"
            "Expected:\n"
            "  edgecase_100/\n"
            f"    {eval_folder}/\n"
            "      offset/no_rand/\n"
            "      offset/rand/\n"
            "      policy/no_rand/\n"
            "      policy/rand/"
        )

    print(
        "Simulation source:"
    )

    print(
        "  benchmark_*_raw_per_edgecase.csv"
    )

    print(
        "Simulation matched metric:"
    )

    print(
        "  position = x + y"
    )

    print(
        "  velocity = x + y + z"
    )

    branches = [
        (
            "offset",
            "no_rand",
        ),
        (
            "offset",
            "rand",
        ),
        (
            "policy",
            "no_rand",
        ),
        (
            "policy",
            "rand",
        ),
    ]

    records = []

    for (
        drive_mode,
        training_condition,
    ) in branches:

        branch = os.path.join(
            base,
            drive_mode,
            training_condition,
        )

        if not os.path.isdir(
            branch
        ):
            print(
                "[WARNING] Missing simulation branch:"
            )

            print(
                f"          {branch}"
            )

            continue

        include_ik = (
            drive_mode
            == "offset"
            and training_condition
            == "no_rand"
        )

        branch_records = (
            load_sim_branch_clean(
                branch=branch,
                drive_mode=drive_mode,
                training_condition=training_condition,
                include_ik=include_ik,
            )
        )

        records.extend(
            branch_records
        )

    if len(
        records
    ) == 0:
        raise RuntimeError(
            "No simulation benchmark values were loaded below:\n"
            f"  {base}\n\n"
            "The script expects files named:\n"
            "  benchmark_*_raw_per_edgecase.csv\n"
            "inside the offset/policy and rand/no_rand branches."
        )

    print(
        f"Loaded original simulation seed/evaluation records: "
        f"{len(records)}"
    )

    print(
        "Expanding deterministic simulation executions:"
    )

    print(
        f"  IK: {SIM_IK_REPETITIONS} repetitions per trajectory"
    )

    print(
        "  Learned controllers: "
        f"{SIM_LEARNED_REPETITIONS_PER_SEED} repetitions per seed/trajectory"
    )

    records = (
        replicate_simulation_executions(
            records
        )
    )

    print(
        f"Expanded simulation execution records: "
        f"{len(records)}"
    )

    print_record_counts(
        records
    )

    return records



def print_record_counts(
    records,
):
    counts = defaultdict(
        int
    )

    for record in records:
        counts[
            (
                record[
                    "trajectory"
                ],
                record[
                    "controller"
                ],
            )
        ] += 1

    print(
        "Counts:"
    )

    for trajectory in TRAJECTORIES:
        print(
            "  "
            + TRAJECTORY_LABELS[
                trajectory
            ]
        )

        for controller in CONTROLLERS:
            print(
                f"    {controller:<23} "
                f"{counts[(trajectory, controller)]}"
            )


def aggregate_dataset(
    records,
):
    """
    Aggregate scalar run/evaluation MSEs.

    Output keys:
        (trajectory, controller)
        ("all", controller)

    Metrics:
        position
        velocity
    """
    grouped = defaultdict(
        lambda: {
            "position": [],
            "velocity": [],
        }
    )

    for record in records:
        key = (
            record[
                "trajectory"
            ],
            record[
                "controller"
            ],
        )

        grouped[
            key
        ][
            "position"
        ].append(
            float(
                record[
                    "position"
                ]
            )
        )

        grouped[
            key
        ][
            "velocity"
        ].append(
            float(
                record[
                    "velocity"
                ]
            )
        )

    summary = {}

    for trajectory in TRAJECTORIES:
        for controller in CONTROLLERS:
            key = (
                trajectory,
                controller,
            )

            if key not in grouped:
                continue

            summary[
                key
            ] = {}

            for metric in [
                "position",
                "velocity",
            ]:
                values = grouped[
                    key
                ][
                    metric
                ]

                (
                    mean,
                    lower,
                    upper,
                ) = mean_and_t_ci(
                    values
                )

                summary[
                    key
                ][
                    metric
                ] = {
                    "mean":
                        mean,

                    "lower":
                        lower,

                    "upper":
                        upper,

                    "n":
                        len(
                            values
                        ),

                    "values":
                        np.asarray(
                            values,
                            dtype=float,
                        ),
                }

    for controller in CONTROLLERS:
        overall_key = (
            "all",
            controller,
        )

        summary[
            overall_key
        ] = {}

        for metric in [
            "position",
            "velocity",
        ]:
            values = []

            for trajectory in TRAJECTORIES:
                key = (
                    trajectory,
                    controller,
                )

                if key in grouped:
                    values.extend(
                        grouped[
                            key
                        ][
                            metric
                        ]
                    )

            if len(
                values
            ) == 0:
                continue

            (
                mean,
                lower,
                upper,
            ) = mean_and_t_ci(
                values
            )

            summary[
                overall_key
            ][
                metric
            ] = {
                "mean":
                    mean,

                "lower":
                    lower,

                "upper":
                    upper,

                "n":
                    len(
                        values
                    ),

                "values":
                    np.asarray(
                        values,
                        dtype=float,
                    ),
            }

    return summary


def get_stat(
    summary,
    trajectory,
    controller,
    metric,
):
    return (
        summary
        .get(
            (
                trajectory,
                controller,
            ),
            {},
        )
        .get(
            metric,
            None,
        )
    )


def stat_mean(
    stat,
):
    if stat is None:
        return np.nan

    return stat[
        "mean"
    ]


def report_missing(
    summary,
    label,
):
    missing = []

    for trajectory in TRAJECTORIES:
        for controller in CONTROLLERS:
            for metric in [
                "position",
                "velocity",
            ]:
                stat = get_stat(
                    summary,
                    trajectory,
                    controller,
                    metric,
                )

                if stat is None:
                    missing.append(
                        (
                            trajectory,
                            controller,
                            metric,
                        )
                    )

    if len(
        missing
    ) == 0:
        print(
            f"[OK] Complete dataset: {label}"
        )

        return

    print(
        f"[WARNING] Missing combinations in {label}:"
    )

    for (
        trajectory,
        controller,
        metric,
    ) in missing:
        print(
            "  "
            + TRAJECTORY_LABELS[
                trajectory
            ]
            + " | "
            + controller
            + " | "
            + metric
        )


def finish_figure(
    fig,
    path,
    dpi,
    show,
):
    os.makedirs(
        os.path.dirname(
            path
        ),
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    print(
        f"[SAVED] {path}"
    )

    if show:
        plt.show()

    plt.close(
        fig
    )


def add_change_annotation(
    ax,
    x_left,
    x_right,
    y_left,
    y_right,
    color,
):
    """
    Connect the tops of two comparison bars and annotate the relative change.

    The connecting line and percentage label use the same color as the
    corresponding controller bars.
    """
    change = relative_change(
        y_left,
        y_right,
    )

    if not np.isfinite(
        change
    ):
        return

    top = max(
        y_left,
        y_right,
    )

    if (
        not np.isfinite(
            top
        )
        or top <= 0
    ):
        return

    ax.plot(
        [
            x_left,
            x_right,
        ],
        [
            y_left,
            y_right,
        ],
        color="black",
        linewidth=1.4,
        marker="o",
        markersize=3.5,
        zorder=5,
    )

    text_y = (
        top
        * 1.08
    )

    ax.text(
        (
            x_left
            + x_right
        )
        / 2.0,
        text_y,
        f"{change:+.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
        fontweight="bold",
    )


def plot_matched_overall(
    summaries,
    metric,
    output_dir,
    dpi,
    show,
):
    """
    Two panels:
        left  -> non-optimized simulation model
        right -> optimized simulation model

    For each controller:
        solid bar   -> simulation
        hatched bar -> physical robot
    """
    panels = [
        (
            "Non-Optimized Model",
            summaries[
                "sim_non_opt"
            ],
            summaries[
                "real_non_opt"
            ],
        ),
        (
            "Optimized Model",
            summaries[
                "sim_opt"
            ],
            summaries[
                "real_opt"
            ],
        ),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            14,
            5.6,
        ),
    )

    x = np.arange(
        len(
            CONTROLLERS
        )
    )

    width = 0.34

    for (
        ax,
        (
            panel_title,
            sim_summary,
            real_summary,
        ),
    ) in zip(
        axes,
        panels,
    ):
        for (
            index,
            controller,
        ) in enumerate(
            CONTROLLERS
        ):
            sim_stat = get_stat(
                sim_summary,
                "all",
                controller,
                metric,
            )

            real_stat = get_stat(
                real_summary,
                "all",
                controller,
                metric,
            )

            color = (
                CONTROLLER_COLORS[
                    controller
                ]
            )

            ax.bar(
                index
                - width / 2.0,
                stat_mean(
                    sim_stat
                ),
                width,
                yerr=ci_yerr(
                    sim_stat
                ),
                capsize=4,
                color=color,
                alpha=0.85,
            )

            ax.bar(
                index
                + width / 2.0,
                stat_mean(
                    real_stat
                ),
                width,
                yerr=ci_yerr(
                    real_stat
                ),
                capsize=4,
                facecolor="none",
                edgecolor=color,
                linewidth=1.4,
                hatch="//",
            )

        ax.set_title(
            panel_title
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            controller_tick_labels(
                CONTROLLERS
            ),
            fontsize=9,
        )

        ax.set_ylabel(
            metric_ylabel(
                metric
            )
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    fig.legend(
        handles=[
            Patch(
                facecolor="0.6",
                alpha=0.85,
                label="Simulation",
            ),
            Patch(
                facecolor="none",
                edgecolor="black",
                hatch="//",
                label="Physical Robot",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.94,
        )
    )

    finish_figure(
        fig,
        os.path.join(
            output_dir,
            f"matched_{metric}_mse.png",
        ),
        dpi,
        show,
    )


def plot_matched_by_trajectory(
    sim_summary,
    real_summary,
    model_slug,
    model_title,
    metric,
    output_dir,
    dpi,
    show,
):
    """
    2x2 panels for the four matched edge cases.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            14,
            10,
        ),
    )

    axes = axes.ravel()

    x = np.arange(
        len(
            CONTROLLERS
        )
    )

    width = 0.34

    for (
        ax,
        trajectory,
    ) in zip(
        axes,
        TRAJECTORIES,
    ):
        for (
            index,
            controller,
        ) in enumerate(
            CONTROLLERS
        ):
            sim_stat = get_stat(
                sim_summary,
                trajectory,
                controller,
                metric,
            )

            real_stat = get_stat(
                real_summary,
                trajectory,
                controller,
                metric,
            )

            color = (
                CONTROLLER_COLORS[
                    controller
                ]
            )

            ax.bar(
                index
                - width / 2.0,
                stat_mean(
                    sim_stat
                ),
                width,
                yerr=ci_yerr(
                    sim_stat
                ),
                capsize=3,
                color=color,
                alpha=0.85,
            )

            ax.bar(
                index
                + width / 2.0,
                stat_mean(
                    real_stat
                ),
                width,
                yerr=ci_yerr(
                    real_stat
                ),
                capsize=3,
                facecolor="none",
                edgecolor=color,
                linewidth=1.4,
                hatch="//",
            )

        ax.set_title(
            TRAJECTORY_LABELS[
                trajectory
            ]
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            controller_tick_labels(
                CONTROLLERS
            ),
            fontsize=8,
        )

        ax.set_ylabel(
            metric_ylabel(
                metric
            )
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    fig.suptitle(
        (
            "Matched Simulation vs Physical Robot"
            f" — {model_title}"
        ),
        y=0.99,
    )

    fig.legend(
        handles=[
            Patch(
                facecolor="0.6",
                alpha=0.85,
                label="Simulation",
            ),
            Patch(
                facecolor="none",
                edgecolor="black",
                hatch="//",
                label="Physical Robot",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.955,
        ),
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.92,
        )
    )

    finish_figure(
        fig,
        os.path.join(
            output_dir,
            (
                f"matched_{model_slug}_"
                f"by_trajectory_{metric}_mse.png"
            ),
        ),
        dpi,
        show,
    )


def plot_model_influence(
    summaries,
    metric,
    output_dir,
    dpi,
    show,
):
    """
    Two panels:
        Simulation
        Physical Robot

    Learned controllers only.

    Each controller:
        solid   -> non-optimized training model
        hatched -> optimized training model
    """
    panels = [
        (
            "Simulation",
            summaries[
                "sim_non_opt"
            ],
            summaries[
                "sim_opt"
            ],
        ),
        (
            "Physical Robot",
            summaries[
                "real_non_opt"
            ],
            summaries[
                "real_opt"
            ],
        ),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.5,
            5.6,
        ),
    )

    x = np.arange(
        len(
            LEARNED_CONTROLLERS
        )
    )

    width = 0.34

    for (
        ax,
        (
            panel_title,
            non_opt_summary,
            opt_summary,
        ),
    ) in zip(
        axes,
        panels,
    ):
        annotation_top = 0.0

        for (
            index,
            controller,
        ) in enumerate(
            LEARNED_CONTROLLERS
        ):
            non_opt_stat = get_stat(
                non_opt_summary,
                "all",
                controller,
                metric,
            )

            opt_stat = get_stat(
                opt_summary,
                "all",
                controller,
                metric,
            )

            y_left = stat_mean(
                non_opt_stat
            )

            y_right = stat_mean(
                opt_stat
            )

            x_left = (
                index
                - width / 2.0
            )

            x_right = (
                index
                + width / 2.0
            )

            color = (
                CONTROLLER_COLORS[
                    controller
                ]
            )

            ax.bar(
                x_left,
                y_left,
                width,
                yerr=ci_yerr(
                    non_opt_stat
                ),
                capsize=4,
                color=color,
                alpha=0.85,
            )

            ax.bar(
                x_right,
                y_right,
                width,
                yerr=ci_yerr(
                    opt_stat
                ),
                capsize=4,
                facecolor="none",
                edgecolor=color,
                linewidth=1.4,
                hatch="//",
            )

            if (
                np.isfinite(
                    y_left
                )
                and np.isfinite(
                    y_right
                )
            ):
                add_change_annotation(
                    ax,
                    x_left,
                    x_right,
                    y_left,
                    y_right,
                    color=color,
                )

                annotation_top = max(
                    annotation_top,
                    max(
                        y_left,
                        y_right,
                    )
                    * 1.24,
                )

        ax.set_title(
            panel_title
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            controller_tick_labels(
                LEARNED_CONTROLLERS
            ),
            fontsize=9,
        )

        ax.set_ylabel(
            metric_ylabel(
                metric
            )
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        if annotation_top > 0:
            current_top = (
                ax.get_ylim()[1]
            )

            ax.set_ylim(
                0,
                max(
                    current_top,
                    annotation_top,
                ),
            )

    fig.legend(
        handles=[
            Patch(
                facecolor="0.6",
                alpha=0.85,
                label="Non-Optimized Model",
            ),
            Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=1.4,
                hatch="//",
                label="Optimized Model",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.94,
        )
    )

    finish_figure(
        fig,
        os.path.join(
            output_dir,
            (
                f"model_influence_"
                f"{metric}_mse.png"
            ),
        ),
        dpi,
        show,
    )


def plot_domain_randomization(
    summaries,
    metric,
    output_dir,
    dpi,
    show,
):
    """
    Four panels:

        Non-Optimized Model — Simulation
        Non-Optimized Model — Physical Robot
        Optimized Model — Simulation
        Optimized Model — Physical Robot

    x-axis:
        Hybrid Offset
        Fully Learned

    bars:
        without DR
        with DR
    """
    panels = [
        (
            "Non-Optimized Model — Simulation",
            summaries[
                "sim_non_opt"
            ],
        ),
        (
            "Non-Optimized Model — Physical Robot",
            summaries[
                "real_non_opt"
            ],
        ),
        (
            "Optimized Model — Simulation",
            summaries[
                "sim_opt"
            ],
        ),
        (
            "Optimized Model — Physical Robot",
            summaries[
                "real_opt"
            ],
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.5,
            9.5,
        ),
    )

    axes = axes.ravel()

    x = np.arange(
        len(
            ARCHITECTURES
        )
    )

    width = 0.34

    for (
        ax,
        (
            panel_title,
            summary,
        ),
    ) in zip(
        axes,
        panels,
    ):
        annotation_top = 0.0

        for (
            index,
            (
                non_dr_controller,
                dr_controller,
                architecture_label,
            ),
        ) in enumerate(
            ARCHITECTURES
        ):
            non_dr_stat = get_stat(
                summary,
                "all",
                non_dr_controller,
                metric,
            )

            dr_stat = get_stat(
                summary,
                "all",
                dr_controller,
                metric,
            )

            y_left = stat_mean(
                non_dr_stat
            )

            y_right = stat_mean(
                dr_stat
            )

            x_left = (
                index
                - width / 2.0
            )

            x_right = (
                index
                + width / 2.0
            )

            color = (
                CONTROLLER_COLORS[
                    non_dr_controller
                ]
            )

            ax.bar(
                x_left,
                y_left,
                width,
                yerr=ci_yerr(
                    non_dr_stat
                ),
                capsize=4,
                color=color,
                alpha=0.85,
            )

            ax.bar(
                x_right,
                y_right,
                width,
                yerr=ci_yerr(
                    dr_stat
                ),
                capsize=4,
                facecolor="none",
                edgecolor=color,
                linewidth=1.4,
                hatch="//",
            )

            if (
                np.isfinite(
                    y_left
                )
                and np.isfinite(
                    y_right
                )
            ):
                add_change_annotation(
                    ax,
                    x_left,
                    x_right,
                    y_left,
                    y_right,
                    color=color,
                )

                annotation_top = max(
                    annotation_top,
                    max(
                        y_left,
                        y_right,
                    )
                    * 1.24,
                )

        ax.set_title(
            panel_title
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                "Hybrid Offset",
                "Fully Learned",
            ]
        )

        ax.set_ylabel(
            metric_ylabel(
                metric
            )
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        if annotation_top > 0:
            current_top = (
                ax.get_ylim()[1]
            )

            ax.set_ylim(
                0,
                max(
                    current_top,
                    annotation_top,
                ),
            )

    fig.legend(
        handles=[
            Patch(
                facecolor="0.6",
                alpha=0.85,
                label="Without DR",
            ),
            Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=1.4,
                hatch="//",
                label="With DR",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            1.005,
        ),
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.965,
        )
    )

    finish_figure(
        fig,
        os.path.join(
            output_dir,
            (
                "domain_randomization_"
                f"{metric}_mse.png"
            ),
        ),
        dpi,
        show,
    )


def save_summary_csv(
    summaries,
    output_path,
):
    metadata = {
        "sim_non_opt":
            (
                "Simulation",
                "Non-Optimized",
            ),

        "sim_opt":
            (
                "Simulation",
                "Optimized",
            ),

        "real_non_opt":
            (
                "Physical Robot",
                "Non-Optimized",
            ),

        "real_opt":
            (
                "Physical Robot",
                "Optimized",
            ),
    }

    rows = []

    for (
        dataset_key,
        summary,
    ) in summaries.items():
        (
            domain,
            model,
        ) = metadata[
            dataset_key
        ]

        for trajectory in (
            TRAJECTORIES
            + [
                "all",
            ]
        ):
            for controller in CONTROLLERS:
                for metric in [
                    "position",
                    "velocity",
                ]:
                    stat = get_stat(
                        summary,
                        trajectory,
                        controller,
                        metric,
                    )

                    if stat is None:
                        continue

                    rows.append(
                        {
                            "domain":
                                domain,

                            "simulation_model":
                                model,

                            "trajectory":
                                (
                                    "Overall"
                                    if trajectory == "all"
                                    else TRAJECTORY_LABELS[
                                        trajectory
                                    ]
                                ),

                            "controller":
                                controller,

                            "metric":
                                (
                                    "Position MSE"
                                    if metric == "position"
                                    else "Velocity MSE"
                                ),

                            "mean":
                                stat[
                                    "mean"
                                ],

                            "ci_lower":
                                stat[
                                    "lower"
                                ],

                            "ci_upper":
                                stat[
                                    "upper"
                                ],

                            "n_values":
                                stat[
                                    "n"
                                ],
                        }
                    )

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        output_path,
        index=False,
    )

    print(
        f"[SAVED] {output_path}"
    )


def main():
    args = parse_args()

    evaluation_root = os.path.abspath(
        args.evaluation_root
    )

    output_root = (
        os.path.abspath(
            args.output_dir
        )
        if args.output_dir
        is not None
        else os.path.join(
            evaluation_root,
            "sim_to_real",
        )
    )

    roots = {
        "real_non_opt":
            os.path.join(
                evaluation_root,
                "real_non_opt",
            ),

        "real_opt":
            os.path.join(
                evaluation_root,
                "real_opt",
            ),

        "sim_non_opt":
            os.path.join(
                evaluation_root,
                "non_optimized",
            ),

        "sim_opt":
            os.path.join(
                evaluation_root,
                "optimized",
            ),
    }

    print(
        "\nEvaluation root:"
    )

    print(
        f"  {evaluation_root}"
    )

    print(
        "\nInput roots:"
    )

    for (
        key,
        path,
    ) in roots.items():
        print(
            f"  {key:<13} "
            f"{path}"
        )

    print(
        "\nMatched metric definition:"
    )

    print(
        "  Position MSE = mean(dx^2 + dy^2)"
    )

    print(
        "  Velocity MSE = mean(dvx^2 + dvy^2 + domega^2)"
    )

    print(
        "\nSimulation conversion from raw_per_edgecase:"
    )

    print(
        "  position = x + y"
    )

    print(
        "  velocity = x + y + z"
    )

    print(
        "\nMatched repeated-execution structure:"
    )

    print(
        f"  Simulation IK: each deterministic result x {SIM_IK_REPETITIONS}"
    )

    print(
        "  Simulation learned: each deterministic seed result x "
        f"{SIM_LEARNED_REPETITIONS_PER_SEED}"
    )

    print(
        "  Error bars: 95% Student-t confidence interval"
    )

    print(
        "\nSimulation evaluation environment:"
    )

    print(
        "  "
        + simulation_evaluation_folder(
            args.sim_evaluation_condition
        )
    )

    real_non_opt = (
        load_real_dataset(
            roots[
                "real_non_opt"
            ],
            (
                "Physical Robot / "
                "Policies Trained With Non-Optimized Model"
            ),
            max_steps=args.max_steps,
        )
    )

    real_opt = (
        load_real_dataset(
            roots[
                "real_opt"
            ],
            (
                "Physical Robot / "
                "Policies Trained With Optimized Model"
            ),
            max_steps=args.max_steps,
        )
    )


    sim_non_opt = (
        load_sim_dataset(
            roots[
                "sim_non_opt"
            ],
            (
                "Simulation / "
                "Non-Optimized Model"
            ),
            evaluation_condition=(
                args.sim_evaluation_condition
            ),
        )
    )

    sim_opt = (
        load_sim_dataset(
            roots[
                "sim_opt"
            ],
            (
                "Simulation / "
                "Optimized Model"
            ),
            evaluation_condition=(
                args.sim_evaluation_condition
            ),
        )
    )

    datasets = {
        "real_non_opt":
            real_non_opt,

        "real_opt":
            real_opt,

        "sim_non_opt":
            sim_non_opt,

        "sim_opt":
            sim_opt,
    }


    summaries = {
        key:
            aggregate_dataset(
                records
            )
        for (
            key,
            records,
        )
        in datasets.items()
    }

    print(
        "\nDataset completeness:"
    )

    for (
        key,
        summary,
    ) in summaries.items():
        report_missing(
            summary,
            key,
        )


    save_summary_csv(
        summaries,
        os.path.join(
            output_root,
            "sim_to_real_summary.csv",
        ),
    )

    matched_dir = os.path.join(
        output_root,
        "matched",
    )

    for metric in [
        "position",
        "velocity",
    ]:
        plot_matched_overall(
            summaries=summaries,
            metric=metric,
            output_dir=matched_dir,
            dpi=args.dpi,
            show=args.show,
        )

        plot_matched_by_trajectory(
            sim_summary=summaries[
                "sim_non_opt"
            ],
            real_summary=summaries[
                "real_non_opt"
            ],
            model_slug=(
                "non_optimized"
            ),
            model_title=(
                "Non-Optimized Model"
            ),
            metric=metric,
            output_dir=matched_dir,
            dpi=args.dpi,
            show=args.show,
        )

        plot_matched_by_trajectory(
            sim_summary=summaries[
                "sim_opt"
            ],
            real_summary=summaries[
                "real_opt"
            ],
            model_slug=(
                "optimized"
            ),
            model_title=(
                "Optimized Model"
            ),
            metric=metric,
            output_dir=matched_dir,
            dpi=args.dpi,
            show=args.show,
        )


    model_dir = os.path.join(
        output_root,
        "model_influence",
    )

    for metric in [
        "position",
        "velocity",
    ]:
        plot_model_influence(
            summaries=summaries,
            metric=metric,
            output_dir=model_dir,
            dpi=args.dpi,
            show=args.show,
        )

    dr_dir = os.path.join(
        output_root,
        "domain_randomization",
    )

    for metric in [
        "position",
        "velocity",
    ]:
        plot_domain_randomization(
            summaries=summaries,
            metric=metric,
            output_dir=dr_dir,
            dpi=args.dpi,
            show=args.show,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SIM-TO-REAL ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Outputs:"
    )

    print(
        f"  {output_root}"
    )


if __name__ == "__main__":
    main()
