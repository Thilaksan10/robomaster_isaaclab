import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Directory
# ============================================================

BASE_DIR = Path("preliminary")


# ============================================================
# Files
#
# 1 + 2:
#   activate_contact_sensors=False
#   enabled_self_collisions=False
#
# 3 + 4:
#   activate_contact_sensors=True
#   enabled_self_collisions=True
# ============================================================

files = {
    "Hybrid Offset Policy Training - Sensors/Collisions Disabled":
        BASE_DIR / "rl_games_Racetrack_2026-09-13_21-01-22_summaries.json",

    "Fully Learned Policy Training - Sensors/Collisions Disabled":
        BASE_DIR / "rl_games_Racetrack_2026-09-13_21-43-21_summaries.json",

    "Hybrid Offset Policy Training - Sensors/Collisions Enabled":
        BASE_DIR / "rl_games_Racetrack_2026-09-13_22-22-33_summaries.json",

    "Fully Learned Policy Training - Sensors/Collisions Enabled":
        BASE_DIR / "rl_games_Racetrack_2026-09-13_22-34-03_summaries.json",
}

# ============================================================
# Load JSON
#
# Format:
# [
#     [timestamp, step, value],
#     [timestamp, step, value],
#     ...
# ]
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    data = np.asarray(data, dtype=float)

    timestamps = data[:, 0]
    steps = data[:, 1]
    values = data[:, 2]

    return timestamps, steps, values


data = {}

for name, path in files.items():

    timestamps, steps, values = load_json(path)

    data[name] = {
        "timestamps": timestamps,
        "steps": steps,
        "values": values,
    }


# ============================================================
# Plot all four
# ============================================================

plt.figure(figsize=(12, 6))

for name, run in data.items():
    plt.plot(
        run["steps"],
        run["values"],
        linewidth=2,
        label=name,
    )

plt.xlabel("Training Steps")
plt.ylabel("FPS")
plt.title("Training FPS Comparison")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

save_path = BASE_DIR / "fps_all_runs.png"
plt.savefig(save_path, dpi=300)

plt.show()

print(f"Saved: {save_path}")


# ============================================================
# Offset comparison
#
# activate_contact_sensors=False / self collisions=False
# vs
# activate_contact_sensors=True / self collisions=True
# ============================================================

plt.figure(figsize=(12, 6))

offset_disabled = (
    "Hybrid Offset Policy Training - Sensors/Collisions Disabled"
)
offset_enabled = (
    "Hybrid Offset Policy Training - Sensors/Collisions Enabled"
)

plt.plot(
    data[offset_disabled]["steps"],
    data[offset_disabled]["values"],
    linewidth=2,
    label="Contact Sensors OFF, Self Collisions OFF",
)

plt.plot(
    data[offset_enabled]["steps"],
    data[offset_enabled]["values"],
    linewidth=2,
    label="Contact Sensors ON, Self Collisions ON",
)

plt.xlabel("Training Steps")
plt.ylabel("FPS")
plt.title("Hybrid Offset Policy Training - FPS Comparison")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

save_path = BASE_DIR / "fps_offset_comparison.png"
plt.savefig(save_path, dpi=300)

plt.show()

print(f"Saved: {save_path}")


# ============================================================
# Full policy comparison
#
# activate_contact_sensors=False / self collisions=False
# vs
# activate_contact_sensors=True / self collisions=True
# ============================================================

plt.figure(figsize=(12, 6))

policy_disabled = (
    "Fully Learned Policy Training - Sensors/Collisions Disabled"
)
policy_enabled = (
    "Fully Learned Policy Training - Sensors/Collisions Enabled"
)

plt.plot(
    data[policy_disabled]["steps"],
    data[policy_disabled]["values"],
    linewidth=2,
    label="Contact Sensors OFF, Self Collisions OFF",
)

plt.plot(
    data[policy_enabled]["steps"],
    data[policy_enabled]["values"],
    linewidth=2,
    label="Contact Sensors ON, Self Collisions ON",
)

plt.xlabel("Training Steps")
plt.ylabel("FPS")
plt.title("Fully Learned Policy Training - FPS Comparison")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

save_path = BASE_DIR / "fps_policy_comparison.png"
plt.savefig(save_path, dpi=300)

plt.show()

print(f"Saved: {save_path}")


# ============================================================
# Statistics
# ============================================================

print("\nAverage FPS")
print("=" * 80)

for name, run in data.items():

    values = run["values"]

    print(
        f"{name:45s}: "
        f"mean = {np.mean(values):8.2f}, "
        f"min = {np.min(values):8.2f}, "
        f"max = {np.max(values):8.2f}"
    )


# ============================================================
# Configuration comparison
# ============================================================

offset_disabled_mean = np.mean(data[offset_disabled]["values"])
offset_enabled_mean = np.mean(data[offset_enabled]["values"])

policy_disabled_mean = np.mean(data[policy_disabled]["values"])
policy_enabled_mean = np.mean(data[policy_enabled]["values"])


offset_change = (
    offset_enabled_mean / offset_disabled_mean - 1
) * 100

policy_change = (
    policy_enabled_mean / policy_disabled_mean - 1
) * 100


print("\nConfiguration Comparison")
print("=" * 80)

print("\nHYBRID OFFSET POLICY TRAINING")
print(
    f"Sensors/Collisions Disabled : {offset_disabled_mean:.2f} FPS"
)
print(
    f"Sensors/Collisions Enabled  : {offset_enabled_mean:.2f} FPS"
)
print(
    f"Change                      : {offset_change:+.2f} %"
)


print("\nFULLY LEARNED POLICY TRAINING")
print(
    f"Sensors/Collisions Disabled : {policy_disabled_mean:.2f} FPS"
)
print(
    f"Sensors/Collisions Enabled  : {policy_enabled_mean:.2f} FPS"
)
print(
    f"Change                      : {policy_change:+.2f} %"
)