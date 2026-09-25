import os
import sys
import json
import argparse
import subprocess

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "train.py"
)

LOG_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "rl_games"
    / "Racetrack"
)


TASK = "Racetrack"

DEFAULT_POLICY_MODES = [
    "policy",
    "offset",
]

DEFAULT_SEEDS = [
    0,
    24,
    42,
]

DEFAULT_RANDOMIZATION_MODES = [
    "no_rand",
    "rand",
]

# Change these defaults if necessary.
DEFAULT_NUM_ENVS = 2048

# None means:
# use max_epochs from your RL-Games config.
DEFAULT_MAX_ITERATIONS = 3000



@dataclass
class TrainingConfig:
    policy_mode: str
    seed: int
    randomization_mode: str

    @property
    def offset(self):
        return self.policy_mode == "offset"

    @property
    def randomized(self):
        return self.randomization_mode == "rand"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run Racetrack training combinations "
            "sequentially."
        )
    )

    parser.add_argument(
        "--policy-modes",
        nargs="+",
        choices=[
            "policy",
            "offset",
        ],
        default=DEFAULT_POLICY_MODES,
        help=(
            "Controller modes to train. "
            "Default: policy offset"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=(
            "Training seeds. "
            "Default: 0 24 42"
        ),
    )

    parser.add_argument(
        "--randomization-modes",
        nargs="+",
        choices=[
            "no_rand",
            "rand",
        ],
        default=DEFAULT_RANDOMIZATION_MODES,
        help=(
            "Training randomization modes. "
            "Default: no_rand rand"
        ),
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help=(
            "Number of parallel training environments."
        ),
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=(
            "Override RL-Games max training iterations. "
            "If omitted, use agent config."
        ),
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Optional maximum number of combinations "
            "to execute."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue with the next training if one "
            "training fails."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print all training combinations without "
            "starting Isaac Lab."
        ),
    )

    parser.add_argument(
        "--optimized",
        action="store_true",
        help=(
            "Use the optimized simulation model for training."
        ),
    )

    return parser.parse_args()



def build_training_configs(args):

    configs = []

    for policy_mode in args.policy_modes:

        for randomization_mode in args.randomization_modes:

            for seed in args.seeds:

                configs.append(
                    TrainingConfig(
                        policy_mode=policy_mode,
                        seed=seed,
                        randomization_mode=(
                            randomization_mode
                        ),
                    )
                )

    if args.max_runs is not None:
        configs = configs[:args.max_runs]

    return configs


def make_experiment_name(config, optimized):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    model_name = (
        "optimized"
        if optimized
        else "non_optimized"
    )

    return (
        f"{timestamp}_"
        f"{config.policy_mode}_"
        f"{config.randomization_mode}_"
        f"{model_name}_"
        f"seed_{config.seed}"
    )


def save_manifest(
    manifest_path,
    batch_info,
):

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        manifest_path,
        "w",
    ) as f:

        json.dump(
            batch_info,
            f,
            indent=4,
        )


def print_training_header(
    index,
    total,
    config,
    experiment_name,
    optimized,
):

    print()
    print("#" * 100)
    print(
        f"TRAINING {index}/{total}"
    )
    print("#" * 100)

    print(
        f"Optimized model: "
        f"{optimized}"
    )

    print(
        f"Policy mode        : "
        f"{config.policy_mode.upper()}"
    )

    print(
        f"Seed               : "
        f"{config.seed}"
    )

    print(
        f"Randomization      : "
        f"{config.randomization_mode.upper()}"
    )

    print(
        f"Offset             : "
        f"{config.offset}"
    )

    print(
        f"Domain randomization: "
        f"{config.randomized}"
    )

    print(
        f"Experiment name    : "
        f"{experiment_name}"
    )

    print("#" * 100)
    print()


def run_training(
    config,
    experiment_name,
    num_envs,
    max_iterations,
    optimized,
):

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),

        "--task",
        TASK,

        "--seed",
        str(config.seed),

        "--num_envs",
        str(num_envs),

        "--policy_mode",
        config.policy_mode,

        "--randomization_mode",
        config.randomization_mode,

        "--experiment_name",
        experiment_name,

        "--headless",
    ]

    if optimized:
        command.append("--optimized")

    if max_iterations is not None:
        command.extend([
            "--max_iterations",
            str(max_iterations),
        ])

    print("Command:")
    print()
    print(" ".join(str(part) for part in command))
    print()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    return result.returncode


def print_final_summary(batch_info):

    runs = batch_info["runs"]

    successful = [
        run
        for run in runs
        if run["status"] == "success"
    ]

    failed = [
        run
        for run in runs
        if run["status"] == "failed"
    ]


    print()
    print("=" * 100)
    print("TRAINING BATCH FINISHED")
    print("=" * 100)

    print(
        f"Requested trainings: "
        f"{len(runs)}"
    )

    print(
        f"Successful:          "
        f"{len(successful)}"
    )

    print(
        f"Failed:              "
        f"{len(failed)}"
    )

    print()


    for run in runs:

        status = run["status"].upper()

        print(
            f"{status:8s} | "
            f"{run['policy_mode']:6s} | "
            f"seed {run['seed']:2d} | "
            f"{run['randomization_mode']:7s} | "
            f"{run['experiment_name']}"
        )


    print("=" * 100)



def main():

    args = parse_args()


    if not TRAIN_SCRIPT.is_file():

        raise FileNotFoundError(
            f"Could not find train.py:\n"
            f"{TRAIN_SCRIPT}"
        )


    configs = build_training_configs(
        args
    )


    if len(configs) == 0:

        raise RuntimeError(
            "No training combinations selected."
        )


    print()
    print("=" * 100)
    print("RACETRACK TRAINING BATCH")
    print("=" * 100)

    print(
        f"Number of trainings: "
        f"{len(configs)}"
    )

    print(
        f"Policy modes: "
        f"{args.policy_modes}"
    )

    print(
        f"Seeds: "
        f"{args.seeds}"
    )

    print(
        f"Randomization modes: "
        f"{args.randomization_modes}"
    )

    print(
        f"Number of environments: "
        f"{args.num_envs}"
    )

    print(
        f"Max iterations: "
        f"{args.max_iterations}"
    )

    print("=" * 100)
    print()



    if args.dry_run:

        print(
            "DRY RUN - no training will be started."
        )

        for index, config in enumerate(
            configs,
            start=1,
        ):

            print(
                f"{index:2d}. "
                f"{config.policy_mode:6s} | "
                f"seed={config.seed:2d} | "
                f"{config.randomization_mode}"
            )

        return


    batch_timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    manifest_path = (
        LOG_ROOT
        / f"training_batch_{batch_timestamp}.json"
    )

    batch_info = {
        "batch_started": batch_timestamp,
        "task": TASK,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "optimized": args.optimized,
        "runs": [],
    }


    save_manifest(
        manifest_path,
        batch_info,
    )


    for index, config in enumerate(
        configs,
        start=1,
    ):

        experiment_name = (
            make_experiment_name(
                config,
                args.optimized
            )
        )


        print_training_header(
            index=index,
            total=len(configs),
            config=config,
            experiment_name=(
                experiment_name
            ),
            optimized=args.optimized
        )


        expected_run_dir = (
            LOG_ROOT
            / experiment_name
        )

        expected_checkpoint = (
            expected_run_dir
            / "nn"
            / "Racetrack.pth"
        )


        run_entry = {
            **asdict(config),

            "experiment_name": (
                experiment_name
            ),

            "run_directory": str(
                expected_run_dir
            ),

            "expected_checkpoint": str(
                expected_checkpoint
            ),

            "status": "running",

            "return_code": None,
        }


        batch_info["runs"].append(
            run_entry
        )

        save_manifest(
            manifest_path,
            batch_info,
        )


        return_code = run_training(
            config=config,
            experiment_name=(
                experiment_name
            ),
            num_envs=args.num_envs,
            max_iterations=(
                args.max_iterations
            ),
            optimized=args.optimized,
        )


        run_entry["return_code"] = (
            return_code
        )


        if return_code == 0:

            run_entry["status"] = (
                "success"
            )

            print()
            print(
                f"[SUCCESS] "
                f"{experiment_name}"
            )

        else:

            run_entry["status"] = (
                "failed"
            )

            print()
            print(
                f"[FAILED] "
                f"{experiment_name}"
            )

            print(
                f"Return code: "
                f"{return_code}"
            )


        run_entry["checkpoint_exists"] = (
            expected_checkpoint.is_file()
        )


        save_manifest(
            manifest_path,
            batch_info,
        )



        if (
            return_code != 0
            and not args.continue_on_error
        ):

            print()
            print(
                "Stopping batch because a "
                "training failed."
            )

            print(
                "Use --continue-on-error "
                "if you want remaining "
                "trainings to continue."
            )

            break


    batch_info["batch_finished"] = (
        datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    save_manifest(
        manifest_path,
        batch_info,
    )


    print_final_summary(
        batch_info
    )


    print()
    print(
        f"Training manifest:\n"
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()