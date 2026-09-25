import numpy as np
import os


# ============================================================
# Paths
# ============================================================

REAL_FILE = "debug_real/policy_outputs.csv"

SIM_FILE = (
    "components/record/debug_real/"
    "policy_outputs_sim_from_real_obs.csv"
)

OUTPUT_FILE = (
    "components/record/debug_real/"
    "policy_output_comparison.csv"
)


# ============================================================
# Load
# ============================================================

real = np.loadtxt(
    REAL_FILE,
    delimiter=","
)

sim = np.loadtxt(
    SIM_FILE,
    delimiter=","
)


# Make sure a single row still has shape (1, 4)
real = np.atleast_2d(real)
sim = np.atleast_2d(sim)


print("Real shape:", real.shape)
print("Sim shape: ", sim.shape)


# ============================================================
# Check length
# ============================================================

if len(real) != len(sim):
    print()
    print(
        "WARNING: Different number of rows!"
    )
    print(f"Real: {len(real)}")
    print(f"Sim:  {len(sim)}")

    n = min(
        len(real),
        len(sim)
    )

    print(
        f"Comparing first {n} samples."
    )

    real = real[:n]
    sim = sim[:n]


# ============================================================
# Compare
# ============================================================

difference = sim - real

abs_difference = np.abs(
    difference
)


wheel_names = [
    "FL",
    "RL",
    "RR",
    "FR"
]


print()
print("========================================")
print("POLICY OUTPUT COMPARISON")
print("========================================")

for i, wheel in enumerate(wheel_names):

    mae = np.mean(
        abs_difference[:, i]
    )

    max_error = np.max(
        abs_difference[:, i]
    )

    print(
        f"{wheel}: "
        f"MAE = {mae:.10f}, "
        f"Max = {max_error:.10f}"
    )


overall_mae = np.mean(
    abs_difference
)

overall_max = np.max(
    abs_difference
)


print()
print(
    f"Overall MAE:       "
    f"{overall_mae:.10f}"
)

print(
    f"Largest difference: "
    f"{overall_max:.10f}"
)


# ============================================================
# Check whether they are effectively identical
# ============================================================

same = np.allclose(
    real,
    sim,
    rtol=1e-5,
    atol=1e-6
)

print()
print(
    "Outputs effectively identical:",
    same
)


# ============================================================
# Find location of largest difference
# ============================================================

flat_index = np.argmax(
    abs_difference
)

row, col = np.unravel_index(
    flat_index,
    abs_difference.shape
)

print()
print("Largest error:")
print(f"Timestep: {row}")
print(f"Wheel:    {wheel_names[col]}")
print(f"Real:     {real[row, col]:.10f}")
print(f"Sim:      {sim[row, col]:.10f}")
print(
    f"Difference: "
    f"{difference[row, col]:.10f}"
)


# ============================================================
# Print first few rows
# ============================================================

print()
print("First 5 comparisons:")

for i in range(
    min(5, len(real))
):

    print()
    print(f"Timestep {i}")
    print(
        "Real:",
        real[i]
    )
    print(
        "Sim: ",
        sim[i]
    )
    print(
        "Diff:",
        difference[i]
    )


# ============================================================
# Save full comparison
# ============================================================

comparison = np.hstack([
    real,
    sim,
    difference,
    abs_difference
])


header = (
    "real_FL,real_RL,real_RR,real_FR,"
    "sim_FL,sim_RL,sim_RR,sim_FR,"
    "diff_FL,diff_RL,diff_RR,diff_FR,"
    "abs_diff_FL,abs_diff_RL,"
    "abs_diff_RR,abs_diff_FR"
)


np.savetxt(
    OUTPUT_FILE,
    comparison,
    delimiter=",",
    header=header,
    comments=""
)


print()
print(
    f"Comparison saved to: "
    f"{OUTPUT_FILE}"
)
print("========================================")