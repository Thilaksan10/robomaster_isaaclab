import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

benchmark_modes = ["policy", "offset"]
mode = 1

DT = 0.02

trajectory_size = 4000
seeds = [0, 24, 42]
domain_randomization_modes = [True, False]

base_path = "."


def make_folder_name(trajectory_size, seed_no, domain_randomized):
    suffix = "rand" if domain_randomized else "no_rand"
    return f"records_multiple_{trajectory_size}_small_seed_{seed_no}_{suffix}"


def get_errors(reference, policy, metric="mae"):
    diff = reference - policy

    if metric == "mae":
        err = np.abs(diff)
    elif metric == "mse":
        err = diff ** 2
    else:
        raise ValueError("metric must be 'mae' or 'mse'")

    return (
        np.mean(err[:, 0]),
        np.mean(err[:, 1]),
        np.mean(err[:, 2]),
        np.mean(err[:, :2]),
        np.mean(err),
    )


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

    return (
        np.mean(err[:, 0]),
        np.mean(err[:, 1]),
        np.mean(err[:, 2]),
        np.mean(err[:, :2]),
        np.mean(err),
    )


def compute_mean_and_ci(data, confidence=0.95):
    data = np.asarray(data)

    n = data.shape[0]
    mean = np.mean(data, axis=0)

    if n < 2:
        return mean, np.zeros_like(mean)

    sem = stats.sem(data, axis=0, ddof=1)
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    ci = sem * t_crit

    return mean, ci


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


def plot_group_bar(rand_mean, rand_ci, no_rand_mean, no_rand_ci, title, ylabel, filename):
    labels = ["Rand", "No Rand"]
    values = [rand_mean[4], no_rand_mean[4]]
    cis = [rand_ci[4], no_rand_ci[4]]

    x = np.arange(len(labels))

    plt.figure(figsize=(6, 5))
    plt.bar(x, values, yerr=cis, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


metrics = ["mae", "mse"]

results = {}

for group_name in ["rand", "no_rand"]:
    results[group_name] = {}

    for metric in metrics:
        results[group_name][metric] = {
            "ik_abs_pos_errors": [],
            "offset_abs_pos_errors": [],
            "ik_perstep_pos_errors": [],
            "offset_perstep_pos_errors": [],
            "ik_vel_errors": [],
            "offset_vel_errors": [],
        }


folders = []

for domain_randomized in domain_randomization_modes:
    group_name = "rand" if domain_randomized else "no_rand"

    for seed_no in seeds:
        folder_name = make_folder_name(
            trajectory_size=trajectory_size,
            seed_no=seed_no,
            domain_randomized=domain_randomized,
        )

        folder_path = os.path.join(base_path, folder_name)

        folders.append((folder_path, group_name, seed_no))


with tqdm(total=len(folders)) as pbar:
    for folder_path, group_name, seed_no in folders:
        if not os.path.isdir(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            pbar.update(1)
            continue

        directories = [
            name for name in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, name))
        ]

        directories.sort(key=lambda x: int(x.split("_")[1]))

        for directory_name in directories:
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
                results[group_name][metric]["ik_abs_pos_errors"].append(
                    get_errors(reference_omni, ik_omni, metric=metric)
                )

                results[group_name][metric]["offset_abs_pos_errors"].append(
                    get_errors(reference_omni, offset_omni, metric=metric)
                )

                results[group_name][metric]["ik_perstep_pos_errors"].append(
                    get_perstep_position_errors(
                        ik_omni,
                        reference_omni_vel,
                        DT,
                        metric=metric,
                    )
                )

                results[group_name][metric]["offset_perstep_pos_errors"].append(
                    get_perstep_position_errors(
                        offset_omni,
                        reference_omni_vel,
                        DT,
                        metric=metric,
                    )
                )

                results[group_name][metric]["ik_vel_errors"].append(
                    get_errors(reference_omni_vel, ik_omni_vel, metric=metric)
                )

                results[group_name][metric]["offset_vel_errors"].append(
                    get_errors(reference_omni_vel, offset_omni_vel, metric=metric)
                )

        pbar.update(1)


print(f"\nUsing DT = {DT}")
print(f"Benchmark mode = {benchmark_modes[mode].upper()}")
print(f"Trajectory size = {trajectory_size}")
print(f"Seeds = {seeds}\n")


summary = {}

for group_name in ["rand", "no_rand"]:
    summary[group_name] = {}

    print(f"\n{'=' * 20} DOMAIN RANDOMIZATION: {group_name.upper()} {'=' * 20}\n")

    for metric in metrics:
        eval_method = metric.upper()
        summary[group_name][metric] = {}

        r = results[group_name][metric]

        for key in r:
            r[key] = np.asarray(r[key])

        metric_pairs = [
            (
                "OMNI Absolute Position Error",
                "ik_abs_pos_errors",
                "offset_abs_pos_errors",
                "abs",
            ),
            (
                "OMNI Per-Step Position Error",
                "ik_perstep_pos_errors",
                "offset_perstep_pos_errors",
                "step",
            ),
            (
                "OMNI Velocity Error",
                "ik_vel_errors",
                "offset_vel_errors",
                "vel",
            ),
        ]

        for title, ik_key, off_key, short_name in metric_pairs:
            ik_mean, ik_ci = compute_mean_and_ci(r[ik_key])
            off_mean, off_ci = compute_mean_and_ci(r[off_key])

            summary[group_name][metric][f"ik_{short_name}_mean"] = ik_mean
            summary[group_name][metric][f"ik_{short_name}_ci"] = ik_ci
            summary[group_name][metric][f"offset_{short_name}_mean"] = off_mean
            summary[group_name][metric][f"offset_{short_name}_ci"] = off_ci

            print_metric_block(
                f"{title} ({eval_method})",
                eval_method,
                ik_mean,
                ik_ci,
                off_mean,
                off_ci,
            )


# Compare rand vs no_rand directly for IK and benchmark mode

for metric in metrics:
    eval_method = metric.upper()

    for method_prefix, method_name in [
        ("ik", "IK"),
        ("offset", benchmark_modes[mode].upper()),
    ]:
        for short_name, plot_title in [
            ("abs", "Omni Absolute Position Error"),
            ("step", "Omni Per-Step Position Error"),
            ("vel", "Omni Velocity Error"),
        ]:
            rand_mean = summary["rand"][metric][f"{method_prefix}_{short_name}_mean"]
            rand_ci = summary["rand"][metric][f"{method_prefix}_{short_name}_ci"]

            no_rand_mean = summary["no_rand"][metric][f"{method_prefix}_{short_name}_mean"]
            no_rand_ci = summary["no_rand"][metric][f"{method_prefix}_{short_name}_ci"]

            plot_group_bar(
                rand_mean,
                rand_ci,
                no_rand_mean,
                no_rand_ci,
                title=f"{method_name} {plot_title} 3D {eval_method}",
                ylabel=eval_method,
                filename=f"bar_compare_{method_prefix}_{short_name}_{metric}_rand_vs_no_rand.png",
            )