import os

filename = "policies_table_real.txt"

columns = [
    "policy_name",
    "lin_scale",        # <-- FIXED (comma added)
    "ang_scale",
    "bal_scale"
    "k_lin",
    "k_ang",
    "mae_vel",
    "mae_pos",
    "mse_vel",
    "mse_pos",
]

# Example data (adjust as needed)
new_rows = [
    ["-", "IK", "-", 0, 0, 0, 0.3922635045853298, 0.014353330112151874, 0.2752538425044906, 9.04499977006462],
    ["2026-04-23_19-16-35", 0.5, 1.0, 0.5, 5.5, 0.1, 0.3434147793061141, 0.013567870772515454, 0.21286672868566078, 25.155744689321576],
    ["2026-04-23_22-20-09", 0.7, 1.0, 0.25, 5.5, 0.1, 0.33265964708250095, 0.012898786472310672, 0.19308018519751843, 20.837774780344517],
    ["2026-04-24_10-54-14", 0.8, 1.0, 0.25, 5.5, 0.1, 0.2801832327678997, 0.01195304573662012, 0.1449107045475943, 22.952603814138094],
    ["2026-04-24_15-53-50", 0.9, 1.0, 0.25, 5.5, 0.1, 0.36419031906487365, 0.013723544245157576, 0.2513249458396646, 21.53197704712538],
    ["2026-04-25_02-14-25", 1.0, 1.0, 0.25, 5.5, 0.1, 0.3034609900924599, 0.01256668609093987, 0.1697476460156725, 0.1697476460156725],
]
# Experiments with k_ang 0.1 and new k_head
# (k_x & k_y 5.5) 0.5, (k_z 0.1) 1, (k_bal 0.5) 1
# 2026-04-17_14-48-07  0.26577562408169314, 14.242374830534331 --> best positional

# 2026-04-21_16-36-28  0.15590500946357183, 26.20873325716808
# 2026-04-22_13-35-59  0.16374633293731833  28.69861010112499

def to_str(value):
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)

def format_row(row, widths):
    return " | ".join(to_str(v).ljust(w) for v, w in zip(row, widths))

# Load existing rows (if file exists)
existing_rows = []
if os.path.exists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
        if len(lines) >= 2:
            for line in lines[2:]:
                if line.strip():
                    existing_rows.append([part.strip() for part in line.split("|")])

# Compute column widths
all_rows = [columns] + existing_rows + [[to_str(v) for v in row] for row in new_rows]
widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(columns))]

# Write file
if not os.path.exists(filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(format_row(columns, widths) + "\n")
        f.write("-+-".join("-" * w for w in widths) + "\n")
        for row in new_rows:
            f.write(format_row(row, widths) + "\n")
else:
    with open(filename, "a", encoding="utf-8") as f:
        for row in new_rows:
            f.write(format_row(row, widths) + "\n")

print(f"Table updated in {filename}")