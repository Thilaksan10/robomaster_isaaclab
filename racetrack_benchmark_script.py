import os
import sys
import json
import argparse
import subprocess

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PLAY_SCRIPT = (
    PROJECT_ROOT
    / "play.py"
)

BENCHMARK_SCRIPT = (
    PROJECT_ROOT
    / "components"
    / "benchmark.py"
)

SEEDED_SCRIPT = (
    PROJECT_ROOT
    / "components"
    / "seeded_benchmark.py"
)

LOG_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "rl_games"
    / "Racetrack"
)



NUM_ENVS = 4000
NUM_RACETRACKS = 4

# Keep the task from your current runner.
TASK = "Robomaster"

# IMPORTANT:
# Raw recordings stay OUTSIDE evaluation/.
RECORD_FOLDER = (
    PROJECT_ROOT
    / f"records_multiple_{NUM_ENVS}_small_new"
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run Racetrack benchmarks using either the "
            "optimized or non-optimized robot model."
        )
    )

    parser.add_argument(
        "--optimized",
        action="store_true",
        help=(
            "Use the optimized simulation model and "
            "the policies trained with the optimized robot."
        ),
    )

    return parser.parse_args()


@dataclass
class Model:
    controller: str
    run_id: str
    seed: int
    training_randomization: bool


NON_OPTIMIZED_MODELS = [

    # --------------------------------------------------------
    # FULL POLICY - NON RANDOMIZED TRAINING
    # --------------------------------------------------------

    Model(
        controller="policy",
        run_id="2026-07-29_20-48-06",
        seed=0,
        training_randomization=False,
    ),

    Model(
        controller="policy",
        run_id="2026-07-30_02-26-03",
        seed=24,
        training_randomization=False,
    ),

    Model(
        controller="policy",
        run_id="2026-07-29_14-25-19",
        seed=42,
        training_randomization=False,
    ),


    # --------------------------------------------------------
    # FULL POLICY - RANDOMIZED TRAINING
    # --------------------------------------------------------

    Model(
        controller="policy",
        run_id="2026-09-03_18-02-53_policy_rand_seed_0",
        seed=0,
        training_randomization=True,
    ),

    Model(
        controller="policy",
        run_id="2026-09-03_23-51-25_policy_rand_seed_24",
        seed=24,
        training_randomization=True,
    ),

    Model(
        controller="policy",
        run_id="2026-09-04_05-43-48_policy_rand_seed_42",
        seed=42,
        training_randomization=True,
    ),


    # --------------------------------------------------------
    # OFFSET - NON RANDOMIZED TRAINING
    # --------------------------------------------------------

    Model(
        controller="offset",
        run_id="2026-08-02_11-09-37",
        seed=0,
        training_randomization=False,
    ),

    Model(
        controller="offset",
        run_id="2026-08-02_17-10-30",
        seed=24,
        training_randomization=False,
    ),

    Model(
        controller="offset",
        run_id="2026-07-25_14-09-06",
        seed=42,
        training_randomization=False,
    ),


    # --------------------------------------------------------
    # OFFSET - RANDOMIZED TRAINING
    # --------------------------------------------------------

    Model(
        controller="offset",
        run_id="2026-09-04_11-35-17_offset_rand_seed_0",
        seed=0,
        training_randomization=True,
    ),

    Model(
        controller="offset",
        run_id="2026-09-04_16-24-16_offset_rand_seed_24",
        seed=24,
        training_randomization=True,
    ),

    Model(
        controller="offset",
        run_id="2026-09-04_21-30-07_offset_rand_seed_42",
        seed=42,
        training_randomization=True,
    ),
]


OPTIMIZED_MODELS = [

    # --------------------------------------------------------
    # FULL POLICY - NON RANDOMIZED TRAINING
    # --------------------------------------------------------

    Model(
        controller="policy",
        run_id="2026-09-07_01-37-21_policy_no_rand_seed_0",
        seed=0,
        training_randomization=False,
    ),

    Model(
        controller="policy",
        run_id="2026-09-07_06-34-14_policy_no_rand_seed_24",
        seed=24,
        training_randomization=False,
    ),

    Model(
        controller="policy",
        run_id="2026-09-07_11-30-29_policy_no_rand_seed_42",
        seed=42,
        training_randomization=False,
    ),


    # --------------------------------------------------------
    # FULL POLICY - RANDOMIZED TRAINING
    # --------------------------------------------------------

    Model(
        controller="policy",
        run_id="2026-09-07_16-33-13_policy_rand_seed_0",
        seed=0,
        training_randomization=True,
    ),

    Model(
        controller="policy",
        run_id="2026-09-07_22-16-58_policy_rand_seed_24",
        seed=24,
        training_randomization=True,
    ),

    Model(
        controller="policy",
        run_id="2026-09-08_04-02-38_policy_rand_seed_42",
        seed=42,
        training_randomization=True,
    ),


    # --------------------------------------------------------
    # OFFSET - NON RANDOMIZED TRAINING
    # --------------------------------------------------------

    Model(
        controller="offset",
        run_id="2026-09-08_09-37-30_offset_no_rand_seed_0",
        seed=0,
        training_randomization=False,
    ),

    Model(
        controller="offset",
        run_id="2026-09-08_14-35-34_offset_no_rand_seed_24",
        seed=24,
        training_randomization=False,
    ),

    Model(
        controller="offset",
        run_id="2026-09-08_19-41-13_offset_no_rand_seed_42",
        seed=42,
        training_randomization=False,
    ),


    # --------------------------------------------------------
    # OFFSET - RANDOMIZED TRAINING
    # --------------------------------------------------------

    Model(
        controller="offset",
        run_id="2026-09-09_00-40-39_offset_rand_seed_0",
        seed=0,
        training_randomization=True,
    ),

    Model(
        controller="offset",
        run_id="2026-09-09_06-10-22_offset_rand_seed_24",
        seed=24,
        training_randomization=True,
    ),

    Model(
        controller="offset",
        run_id="2026-09-09_11-35-40_offset_rand_seed_42",
        seed=42,
        training_randomization=True,
    ),
]


# OPTIMIZED_MODELS = [

#     # --------------------------------------------------------
#     # POLICY - NON RANDOMIZED
#     # --------------------------------------------------------

#     Model(
#         controller="policy",
#         run_id="2026-09-10_17-44-22_policy_no_rand_seed_0",
#         seed=0,
#         training_randomization=False,
#     ),

#     Model(
#         controller="policy",
#         run_id="2026-09-10_22-49-33_policy_no_rand_seed_24",
#         seed=24,
#         training_randomization=False,
#     ),

#     Model(
#         controller="policy",
#         run_id="2026-09-11_03-57-06_policy_no_rand_seed_42",
#         seed=42,
#         training_randomization=False,
#     ),


#     # --------------------------------------------------------
#     # POLICY - RANDOMIZED
#     # --------------------------------------------------------

#     Model(
#         controller="policy",
#         run_id="2026-09-11_08-58-15_policy_rand_seed_0",
#         seed=0,
#         training_randomization=True,
#     ),

#     Model(
#         controller="policy",
#         run_id="2026-09-11_14-25-15_policy_rand_seed_24",
#         seed=24,
#         training_randomization=True,
#     ),

#     Model(
#         controller="policy",
#         run_id="2026-09-11_20-09-59_policy_rand_seed_42",
#         seed=42,
#         training_randomization=True,
#     ),


#     # --------------------------------------------------------
#     # OFFSET - NON RANDOMIZED
#     # --------------------------------------------------------

#     Model(
#         controller="offset",
#         run_id="2026-09-12_01-51-31_offset_no_rand_seed_0",
#         seed=0,
#         training_randomization=False,
#     ),

#     Model(
#         controller="offset",
#         run_id="2026-09-12_06-58-48_offset_no_rand_seed_24",
#         seed=24,
#         training_randomization=False,
#     ),

#     Model(
#         controller="offset",
#         run_id="2026-09-12_11-52-54_offset_no_rand_seed_42",
#         seed=42,
#         training_randomization=False,
#     ),


#     # --------------------------------------------------------
#     # OFFSET - RANDOMIZED
#     # --------------------------------------------------------

#     Model(
#         controller="offset",
#         run_id="2026-09-12_16-49-25_offset_rand_seed_0",
#         seed=0,
#         training_randomization=True,
#     ),

#     Model(
#         controller="offset",
#         run_id="2026-09-12_21-54-08_offset_rand_seed_24",
#         seed=24,
#         training_randomization=True,
#     ),

#     Model(
#         controller="offset",
#         run_id="2026-09-13_03-05-26_offset_rand_seed_42",
#         seed=42,
#         training_randomization=True,
#     ),
# ]


def get_evaluation_paths(optimized):

    simulation_name = (
        "optimized"
        if optimized
        else "non_optimized"
    )

    evaluation_root = (
        PROJECT_ROOT
        / "evaluation"
        / simulation_name
        / "racetrack"
    )

    reset_stats_dir = (
        evaluation_root
        / "racetrack_reset_statistics"
    )

    no_rand_benchmark_root = (
        evaluation_root
        / "seeded_benchmark_no_rand"
    )

    rand_benchmark_root = (
        evaluation_root
        / "seeded_benchmark_rand"
    )

    return {
        "simulation_name": simulation_name,
        "evaluation_root": evaluation_root,
        "reset_stats_dir": reset_stats_dir,
        "no_rand_benchmark_root": no_rand_benchmark_root,
        "rand_benchmark_root": rand_benchmark_root,
    }


def checkpoint_path(model):

    return (
        LOG_ROOT
        / model.run_id
        / "nn"
        / "Racetrack.pth"
    )


def run_command(command, env=None):

    print()
    print("=" * 100)
    print("RUNNING")
    print(" ".join(str(x) for x in command))
    print("=" * 100)
    print()

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def check_files(models, paths):

    required = [
        PLAY_SCRIPT,
        BENCHMARK_SCRIPT,
        SEEDED_SCRIPT,
    ]

    for path in required:

        if not path.is_file():

            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )


    missing_checkpoints = []

    for model in models:

        checkpoint = checkpoint_path(model)

        if not checkpoint.is_file():

            missing_checkpoints.append(
                (
                    model,
                    checkpoint,
                )
            )


    if missing_checkpoints:

        print()
        print("=" * 100)
        print("MISSING CHECKPOINTS")
        print("=" * 100)

        for model, checkpoint in missing_checkpoints:

            training_name = (
                "rand"
                if model.training_randomization
                else "no_rand"
            )

            print(
                f"{model.controller.upper():6s} | "
                f"seed {model.seed:2d} | "
                f"train {training_name:7s}"
            )

            print(
                f"    {checkpoint}"
            )

        print("=" * 100)

        raise FileNotFoundError(
            "One or more model checkpoints are missing."
        )


    paths["evaluation_root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["reset_stats_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["no_rand_benchmark_root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["rand_benchmark_root"].mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print(
        "Raw simulation recordings will be read from:"
    )
    print(RECORD_FOLDER)
    print()


def run_racetracks(
    checkpoint,
    use_ik,
    offset,
    evaluation_randomization,
    optimized,
    reset_stats_dir,
    stats_name,
):

    env = os.environ.copy()


    env["RACETRACK_USE_IK"] = (
        "1"
        if use_ik
        else "0"
    )

    env["RACETRACK_OFFSET"] = (
        "1"
        if offset
        else "0"
    )


    env["RACETRACK_DOMAIN_RAND"] = (
        "1"
        if evaluation_randomization
        else "0"
    )


    env["RACETRACK_OPTIMIZED"] = (
        "1"
        if optimized
        else "0"
    )


    stats_file = (
        reset_stats_dir
        / f"{stats_name}.json"
    )

    if stats_file.exists():
        stats_file.unlink()

    env["RACETRACK_RESET_STATS_FILE"] = str(
        stats_file
    )


    command = [
        sys.executable,
        str(PLAY_SCRIPT),

        "--task",
        TASK,

        "--num_envs",
        str(NUM_ENVS),

        "--checkpoint",
        str(checkpoint),

        "--headless",
    ]

    run_command(
        command,
        env=env,
    )


    if not stats_file.exists():

        raise RuntimeError(
            "Racetrack run finished without "
            "writing reset statistics:\n"
            f"{stats_file}"
        )


    with open(stats_file, "r") as f:
        stats = json.load(f)


    reset_count = int(
        stats["total"]
    )


    print()
    print(
        f"Unexpected resets THIS run: "
        f"{reset_count}"
    )
    print()


    return reset_count


def run_benchmark(
    model,
    output_root,
):

    training_condition = (
        "rand"
        if model.training_randomization
        else "no_rand"
    )


    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),

        "--controller",
        model.controller,

        "--seed",
        str(model.seed),

        "--training-randomization",
        training_condition,

        "--output-root",
        str(output_root),

        # Raw records remain outside evaluation/
        "--folder-path",
        str(RECORD_FOLDER),

        "--num-racetracks",
        str(NUM_RACETRACKS),
    ]


    run_command(
        command
    )


def run_seeded_benchmark(
    benchmark_root,
):

    command = [
        sys.executable,
        str(SEEDED_SCRIPT),

        "--base-dir",
        str(benchmark_root),
    ]

    run_command(
        command
    )

def run_evaluation_environment(
    evaluation_randomization,
    optimized,
    models,
    paths,
):

    run_results = []



    if evaluation_randomization:

        evaluation_name = "RAND"

        benchmark_root = (
            paths["rand_benchmark_root"]
        )

    else:

        evaluation_name = "NO_RAND"

        benchmark_root = (
            paths["no_rand_benchmark_root"]
        )


    benchmark_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("#" * 100)
    print(
        f"EVALUATION ENVIRONMENT: "
        f"{evaluation_name}"
    )
    print(
        f"ROBOT MODEL: "
        f"{'OPTIMIZED' if optimized else 'NON_OPTIMIZED'}"
    )
    print(
        f"Benchmark output: "
        f"{benchmark_root}"
    )
    print(
        f"Raw recordings: "
        f"{RECORD_FOLDER}"
    )
    print("#" * 100)
    print()


    ik_checkpoint = checkpoint_path(
        models[0]
    )


    print()
    print("=" * 100)
    print(
        f"IK BASELINE - "
        f"{evaluation_name} - "
        f"{'OPTIMIZED' if optimized else 'NON_OPTIMIZED'}"
    )
    print("=" * 100)


    ik_resets = run_racetracks(
        checkpoint=ik_checkpoint,
        use_ik=True,
        offset=False,
        evaluation_randomization=(
            evaluation_randomization
        ),
        optimized=optimized,
        reset_stats_dir=(
            paths["reset_stats_dir"]
        ),
        stats_name=(
            f"{evaluation_name.lower()}_ik"
        ),
    )


    run_results.append(
        {
            "controller": "IK",
            "seed": None,
            "training_randomization": None,
            "run_id": "IK",
            "resets": ik_resets,
        }
    )


    for index, model in enumerate(
        models,
        start=1,
    ):

        checkpoint = checkpoint_path(
            model
        )

        training_name = (
            "rand"
            if model.training_randomization
            else "no_rand"
        )


        print()
        print("#" * 100)
        print(
            f"MODEL {index}/{len(models)}"
        )
        print(
            f"Simulation     : "
            f"{'OPTIMIZED' if optimized else 'NON_OPTIMIZED'}"
        )
        print(
            f"Evaluation env : "
            f"{evaluation_name}"
        )
        print(
            f"Controller     : "
            f"{model.controller.upper()}"
        )
        print(
            f"Training rand  : "
            f"{training_name}"
        )
        print(
            f"Seed           : "
            f"{model.seed}"
        )
        print(
            f"Checkpoint     : "
            f"{model.run_id}"
        )
        print("#" * 100)


        model_resets = run_racetracks(
            checkpoint=checkpoint,
            use_ik=False,
            offset=(
                model.controller
                == "offset"
            ),
            evaluation_randomization=(
                evaluation_randomization
            ),
            optimized=optimized,
            reset_stats_dir=(
                paths["reset_stats_dir"]
            ),
            stats_name=(
                f"{evaluation_name.lower()}_"
                f"{model.controller}_"
                f"{training_name}_"
                f"seed_{model.seed}"
            ),
        )


        run_results.append(
            {
                "controller": (
                    model.controller.upper()
                ),
                "seed": model.seed,
                "training_randomization": (
                    model.training_randomization
                ),
                "run_id": model.run_id,
                "resets": model_resets,
            }
        )


        run_benchmark(
            model=model,
            output_root=benchmark_root,
        )


        accumulated = sum(
            result["resets"]
            for result in run_results
        )


        print()
        print(
            f"Unexpected resets THIS run: "
            f"{model_resets}"
        )
        print(
            f"ACCUMULATED {evaluation_name} "
            f"resets: {accumulated}"
        )
        print()



    total = sum(
        result["resets"]
        for result in run_results
    )


    print()
    print("=" * 100)
    print(
        f"{evaluation_name} RESET REPORT"
    )
    print("=" * 100)


    for result in run_results:

        if result["controller"] == "IK":

            print(
                f"IK       | "
                f"{result['resets']:6d} resets"
            )

            continue


        training_name = (
            "RAND"
            if result[
                "training_randomization"
            ]
            else "NO_RAND"
        )


        print(
            f"{result['controller']:6s} | "
            f"seed {result['seed']:2d} | "
            f"train {training_name:7s} | "
            f"{result['resets']:6d} resets | "
            f"{result['run_id']}"
        )


    print("-" * 100)

    print(
        f"TOTAL {evaluation_name} "
        f"UNEXPECTED RESETS: {total}"
    )

    print("=" * 100)
    print()


    return {
        "evaluation": evaluation_name,
        "total": total,
        "runs": run_results,
    }



def print_final_reset_report(
    no_rand_results,
    rand_results,
):

    no_rand_total = (
        no_rand_results["total"]
    )

    rand_total = (
        rand_results["total"]
    )

    grand_total = (
        no_rand_total
        + rand_total
    )


    print()
    print("=" * 100)
    print("FINAL RACETRACK RESET REPORT")
    print("=" * 100)

    print(
        f"Non-randomized evaluation resets: "
        f"{no_rand_total}"
    )

    print(
        f"Randomized evaluation resets:     "
        f"{rand_total}"
    )

    print(
        f"Total unexpected resets:          "
        f"{grand_total}"
    )

    print("=" * 100)
    print()



def main():

    args = parse_args()

    optimized = args.optimized

    if optimized:

        models = OPTIMIZED_MODELS

    else:

        models = NON_OPTIMIZED_MODELS


    paths = get_evaluation_paths(
        optimized
    )


    print()
    print("=" * 100)
    print("RACETRACK BENCHMARK")
    print("=" * 100)

    print(
        f"Simulation model : "
        f"{'OPTIMIZED' if optimized else 'NON_OPTIMIZED'}"
    )

    print(
        f"Model set        : "
        f"{'OPTIMIZED' if optimized else 'NON_OPTIMIZED'}"
    )

    print(
        f"Evaluation root  : "
        f"{paths['evaluation_root']}"
    )

    print(
        f"Raw recordings   : "
        f"{RECORD_FOLDER}"
    )

    print("=" * 100)
    print()


    check_files(
        models=models,
        paths=paths,
    )


    no_rand_results = (
        run_evaluation_environment(
            evaluation_randomization=False,
            optimized=optimized,
            models=models,
            paths=paths,
        )
    )


    rand_results = (
        run_evaluation_environment(
            evaluation_randomization=True,
            optimized=optimized,
            models=models,
            paths=paths,
        )
    )

    no_rand_root = (
        paths["no_rand_benchmark_root"]
    )

    rand_root = (
        paths["rand_benchmark_root"]
    )


    print()
    print("#" * 100)
    print(
        "SEEDED ANALYSIS: "
        "NON-RANDOMIZED ENVIRONMENT"
    )
    print("#" * 100)

    run_seeded_benchmark(
        no_rand_root
    )


    print()
    print("#" * 100)
    print(
        "SEEDED ANALYSIS: "
        "RANDOMIZED ENVIRONMENT"
    )
    print("#" * 100)

    run_seeded_benchmark(
        rand_root
    )


    print_final_reset_report(
        no_rand_results,
        rand_results,
    )

    print()
    print("=" * 100)
    print(
        "ALL RACETRACK BENCHMARKS FINISHED"
    )
    print("=" * 100)

    print(
        f"\nSimulation model:\n"
        f"{'OPTIMIZED' if optimized else 'NON_OPTIMIZED'}"
    )

    print(
        f"\nEvaluation root:\n"
        f"{paths['evaluation_root']}"
    )

    print(
        f"\nNon-randomized results:\n"
        f"{no_rand_root}"
    )

    print(
        f"\nRandomized results:\n"
        f"{rand_root}"
    )

    print(
        f"\nRaw recordings:\n"
        f"{RECORD_FOLDER}"
    )


if __name__ == "__main__":
    main()