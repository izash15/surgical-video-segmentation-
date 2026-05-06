import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "./experiments/plots/"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------
# Extra-large, bold poster-style font settings
# ----------------------------------------------------
plt.rcParams.update({
    "font.size": 28,
    "axes.titlesize": 34,
    "axes.labelsize": 32,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 24,
    "figure.titlesize": 36
})

# ----------------------------------------------------
# Full per-organ data — QuadPath vs TriPath only
# ----------------------------------------------------
data = pd.DataFrame({
    "Organ": [
        "abdominal_wall", "colon", "inferior_mesenter", "intestinal_veins",
        "liver", "pancreas", "small_intestine", "spleen",
        "stomach", "ureter", "vesicular_glands"
    ],
    "QuadPath_mIoU":  [0.8232, 0.7026, 0.4048, 0.4183, 0.6568, 0.2644, 0.8078, 0.7443, 0.6526, 0.3997, 0.3854],
    "QuadPath_mDice": [0.8941, 0.7979, 0.5408, 0.5275, 0.7374, 0.3509, 0.8802, 0.8121, 0.7428, 0.5128, 0.5023],
    "QuadPath_MASD":  [4.3284, 10.4963, 12.8809, 14.8925, 12.0273, 28.0022, 6.5622, 5.3002, 12.0547, 14.9958, 17.6602],
    "QuadPath_NSD":   [0.7650, 0.6390, 0.4267, 0.5106, 0.6059, 0.3033, 0.7824, 0.7803, 0.6331, 0.4933, 0.4059],
    "TriPath_mIoU":   [0.7886, 0.6282, 0.3604, 0.4126, 0.5765, 0.2665, 0.7672, 0.6667, 0.5739, 0.2501, 0.3142],
    "TriPath_mDice":  [0.8688, 0.7374, 0.4835, 0.5183, 0.6690, 0.3597, 0.8520, 0.7441, 0.6663, 0.3351, 0.4242],
    "TriPath_MASD":   [5.7115, 13.5144, 14.0131, 15.5295, 16.7844, 32.7120, 8.0613, 11.4087, 13.8403, 26.1186, 21.7211],
    "TriPath_NSD":    [0.7285, 0.5603, 0.3762, 0.4974, 0.5231, 0.2870, 0.7320, 0.7037, 0.5495, 0.3170, 0.3124],
})

x = np.arange(len(data["Organ"]))
width = 0.38

# ============================================================
# PLOT 1: Boxplot — mIoU distribution
# ============================================================
fig, ax = plt.subplots(figsize=(16, 10))

box_data = [data["TriPath_mIoU"], data["QuadPath_mIoU"]]

box = ax.boxplot(
    box_data,
    tick_labels=["TriPathEB-WF\nOriginal", "QuadPath"],
    patch_artist=True,
    widths=0.55,
    showmeans=True
)

for element in ["boxes", "whiskers", "caps", "medians", "means"]:
    for item in box[element]:
        item.set_linewidth(3)

for i, model in enumerate(["TriPath_mIoU", "QuadPath_mIoU"], start=1):
    jitter = np.linspace(-0.08, 0.08, len(data))
    ax.scatter(
        np.full(len(data), i) + jitter,
        data[model],
        s=140,
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
        label="_nolegend_"
    )

tri_mean = data["TriPath_mIoU"].mean()
quad_mean = data["QuadPath_mIoU"].mean()

ax.text(1, tri_mean + 0.055, f"Mean = {tri_mean:.3f}", ha="center", fontsize=26, fontweight="bold")
ax.text(2, quad_mean + 0.055, f"Mean = {quad_mean:.3f}", ha="center", fontsize=26, fontweight="bold")

ax.set_title("Distribution of mIoU Across Organs", fontweight="bold", pad=24)
ax.set_ylabel("mIoU", fontweight="bold", labelpad=20)
ax.set_ylim(0, 1.0)
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(True, axis="y", alpha=0.35, linewidth=1.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_paper_miou_distribution_boxplot.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 01_paper_miou_distribution_boxplot")

# ============================================================
# PLOT 2: Grouped bar — mIoU per organ
# ============================================================
fig, ax = plt.subplots(figsize=(24, 12))

bars1 = ax.bar(x - width/2, data["TriPath_mIoU"],  width, label="TriPathEB-WF Original")
bars2 = ax.bar(x + width/2, data["QuadPath_mIoU"], width, label="QuadPath")

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center", va="bottom",
            fontsize=20, fontweight="bold", rotation=90
        )

ax.set_title("Per-Organ mIoU: QuadPath vs TriPathEB-WF Original", fontweight="bold", pad=24)
ax.set_xlabel("Organ", fontweight="bold", labelpad=20)
ax.set_ylabel("mIoU", fontweight="bold", labelpad=20)
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(data["Organ"], rotation=45, ha="right", fontweight="bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(True, axis="y", alpha=0.35, linewidth=1.5)
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_paper_per_organ_miou_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 02_paper_per_organ_miou_comparison")

# ============================================================
# PLOT 3: mIoU improvement % per organ
# ============================================================
data["mIoU_improvement_pct"] = (
    (data["QuadPath_mIoU"] - data["TriPath_mIoU"]) / data["TriPath_mIoU"] * 100
)

fig, ax = plt.subplots(figsize=(24, 12))

bars = ax.bar(data["Organ"], data["mIoU_improvement_pct"], width=0.65)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 4,
        f"{height:.1f}%",
        ha="center", va="bottom",
        fontsize=22, fontweight="bold"
    )

ax.axhline(0, linewidth=3)
ax.set_title("Per-Organ mIoU Improvement: QuadPath Over TriPathEB-WF Original", fontweight="bold", pad=24)
ax.set_xlabel("Organ", fontweight="bold", labelpad=20)
ax.set_ylabel("mIoU Improvement (%)", fontweight="bold", labelpad=20)
plt.xticks(rotation=45, ha="right", fontweight="bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(True, axis="y", alpha=0.35, linewidth=1.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_paper_per_organ_improvement.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 03_paper_per_organ_improvement")

# ============================================================
# PLOT 4: Grouped bar — mDice per organ
# ============================================================
fig, ax = plt.subplots(figsize=(24, 12))

bars1 = ax.bar(x - width/2, data["TriPath_mDice"],  width, label="TriPathEB-WF Original")
bars2 = ax.bar(x + width/2, data["QuadPath_mDice"], width, label="QuadPath")

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center", va="bottom",
            fontsize=20, fontweight="bold", rotation=90
        )

ax.set_title("Per-Organ mDice: QuadPath vs TriPathEB-WF Original", fontweight="bold", pad=24)
ax.set_xlabel("Organ", fontweight="bold", labelpad=20)
ax.set_ylabel("mDice", fontweight="bold", labelpad=20)
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(data["Organ"], rotation=45, ha="right", fontweight="bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(True, axis="y", alpha=0.35, linewidth=1.5)
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_paper_per_organ_mdice_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 04_paper_per_organ_mdice_comparison")

# ============================================================
# PLOT 5: Grouped bar — MASD per organ (lower is better)
# ============================================================
fig, ax = plt.subplots(figsize=(24, 12))

bars1 = ax.bar(x - width/2, data["TriPath_MASD"],  width, label="TriPathEB-WF Original")
bars2 = ax.bar(x + width/2, data["QuadPath_MASD"], width, label="QuadPath")

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            f"{height:.2f}",
            ha="center", va="bottom",
            fontsize=20, fontweight="bold", rotation=90
        )

ax.set_title("Per-Organ MASD: QuadPath vs TriPathEB-WF Original (lower is better)", fontweight="bold", pad=24)
ax.set_xlabel("Organ", fontweight="bold", labelpad=20)
ax.set_ylabel("MASD (pixels)", fontweight="bold", labelpad=20)
ax.set_xticks(x)
ax.set_xticklabels(data["Organ"], rotation=45, ha="right", fontweight="bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(True, axis="y", alpha=0.35, linewidth=1.5)
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_paper_per_organ_masd_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 05_paper_per_organ_masd_comparison")

# ============================================================
# PLOT 6: Grouped bar — NSD per organ
# ============================================================
fig, ax = plt.subplots(figsize=(24, 12))

bars1 = ax.bar(x - width/2, data["TriPath_NSD"],  width, label="TriPathEB-WF Original")
bars2 = ax.bar(x + width/2, data["QuadPath_NSD"], width, label="QuadPath")

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center", va="bottom",
            fontsize=20, fontweight="bold", rotation=90
        )

ax.set_title("Per-Organ NSD: QuadPath vs TriPathEB-WF Original", fontweight="bold", pad=24)
ax.set_xlabel("Organ", fontweight="bold", labelpad=20)
ax.set_ylabel("NSD", fontweight="bold", labelpad=20)
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(data["Organ"], rotation=45, ha="right", fontweight="bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(True, axis="y", alpha=0.35, linewidth=1.5)
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "06_paper_per_organ_nsd_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 06_paper_per_organ_nsd_comparison")

# ============================================================
# Print summary averages
# ============================================================
print("\n--- Average metrics across all 11 organs ---")
for metric in ["mIoU", "mDice", "MASD", "NSD"]:
    q = data[f"QuadPath_{metric}"].mean()
    t = data[f"TriPath_{metric}"].mean()
    print(f"{metric:6s} | QuadPath: {q:.4f} | TriPath: {t:.4f} | Delta: {q-t:+.4f}")

# ============================================================
# TABLE IMAGE: Per-organ raw scores + averages row
# ============================================================

# Pull font sizes from rcParams to match rest of script
BASE_FONT     = plt.rcParams["font.size"]          # 28
TITLE_FONT    = plt.rcParams["axes.titlesize"]     # 34
HEADER_FONT   = BASE_FONT                          # 28
ORGAN_FONT    = BASE_FONT                          # 28
VALUE_FONT    = BASE_FONT - 2                      # 26
LEGEND_FONT   = plt.rcParams["legend.fontsize"]    # 24

# Build display organ names
organ_display = [o.replace("_", " ").title() for o in data["Organ"]]

# Compute averages
avg_row = {
    "Organ": "Average",
    "QuadPath_mIoU":  data["QuadPath_mIoU"].mean(),
    "QuadPath_mDice": data["QuadPath_mDice"].mean(),
    "QuadPath_MASD":  data["QuadPath_MASD"].mean(),
    "QuadPath_NSD":   data["QuadPath_NSD"].mean(),
    "TriPath_mIoU":   data["TriPath_mIoU"].mean(),
    "TriPath_mDice":  data["TriPath_mDice"].mean(),
    "TriPath_MASD":   data["TriPath_MASD"].mean(),
    "TriPath_NSD":    data["TriPath_NSD"].mean(),
}

# Column headers
col_labels = [
    "Organ",
    "Tri mIoU", "Tri mDice", "Tri MASD", "Tri NSD",
    "Quad mIoU", "Quad mDice", "Quad MASD", "Quad NSD",
]

# Build rows
table_rows = []
for i, organ in enumerate(organ_display):
    row = [
        organ,
        f"{data['TriPath_mIoU'].iloc[i]:.4f}",
        f"{data['TriPath_mDice'].iloc[i]:.4f}",
        f"{data['TriPath_MASD'].iloc[i]:.4f}",
        f"{data['TriPath_NSD'].iloc[i]:.4f}",
        f"{data['QuadPath_mIoU'].iloc[i]:.4f}",
        f"{data['QuadPath_mDice'].iloc[i]:.4f}",
        f"{data['QuadPath_MASD'].iloc[i]:.4f}",
        f"{data['QuadPath_NSD'].iloc[i]:.4f}",
    ]
    table_rows.append(row)

n_rows = len(table_rows) + 1  # +1 for avg
n_cols = len(col_labels)

fig, ax = plt.subplots(figsize=(36, 14))
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows + 1)
ax.axis("off")

ax.set_title(
    "Per-Organ Segmentation Metrics: QuadPath vs TriPathEB-WF",
    fontweight="bold", pad=24, fontsize=TITLE_FONT
)

col_widths = [2.8] + [1.15] * 8
col_positions = [sum(col_widths[:i]) for i in range(n_cols)]
total_width = sum(col_widths)

col_positions = [p * n_cols / total_width for p in col_positions]
col_centers   = [col_positions[i] + col_widths[i] * n_cols / total_width / 2
                 for i in range(n_cols)]

HEADER_COLOR = "#2c3e50"
AVG_COLOR    = "#fff2cc"
ALT_COLOR    = "#f8f8f8"
WHITE        = "#ffffff"
BEST_COLOR   = "#27ae60"

header_y = n_rows + 0.5

# Header background
ax.barh(header_y, n_cols, left=0, height=0.85, color=HEADER_COLOR, zorder=2)

# Divider line between model groups

mid_x = (col_centers[4] + col_centers[5]) / 2
ax.plot([mid_x, mid_x], [0, n_rows + 1], color="#aaaaaa", linewidth=2.0, zorder=3)

# Header text
for j, label in enumerate(col_labels):
    ax.text(
        col_centers[j], header_y, label,
        ha="center", va="center",
        fontsize=HEADER_FONT, fontweight="bold", color="white", zorder=3
    )

# Data rows
metrics_pairs = [
    ("TriPath_mIoU",  "QuadPath_mIoU",  False),
    ("TriPath_mDice", "QuadPath_mDice", False),
    ("TriPath_MASD",  "QuadPath_MASD",  True),
    ("TriPath_NSD",   "QuadPath_NSD",   False),
]

for i, row in enumerate(table_rows):
    y = n_rows - i - 0.5
    bg = ALT_COLOR if i % 2 == 0 else WHITE
    ax.barh(y, n_cols, left=0, height=0.9, color=bg, zorder=1)

    ax.text(col_centers[0], y, row[0], ha="center", va="center",
            fontsize=ORGAN_FONT, fontweight="bold", color="#2c3e50")

    for m_idx, (tri_col, quad_col, lower_better) in enumerate(metrics_pairs):
        tri_val  = data[tri_col].iloc[i]
        quad_val = data[quad_col].iloc[i]
        tri_better  = (tri_val < quad_val) if lower_better else (tri_val > quad_val)
        quad_better = (quad_val < tri_val) if lower_better else (quad_val > tri_val)

        tri_j  = 1 + m_idx
        quad_j = 5 + m_idx

        ax.text(col_centers[tri_j], y, f"{tri_val:.4f}", ha="center", va="center",
                fontsize=VALUE_FONT,
                fontweight="bold" if tri_better else "normal",
                color=BEST_COLOR if tri_better else "#2c3e50")

        ax.text(col_centers[quad_j], y, f"{quad_val:.4f}", ha="center", va="center",
                fontsize=VALUE_FONT,
                fontweight="bold" if quad_better else "normal",
                color=BEST_COLOR if quad_better else "#2c3e50")

# Averages row
avg_y = 0.5
ax.barh(avg_y, n_cols, left=0, height=0.9, color=AVG_COLOR, zorder=1)
ax.text(col_centers[0], avg_y, "Average", ha="center", va="center",
        fontsize=ORGAN_FONT, fontweight="bold", color="#2c3e50")

avg_vals = [
    (avg_row["TriPath_mIoU"],  avg_row["QuadPath_mIoU"],  False),
    (avg_row["TriPath_mDice"], avg_row["QuadPath_mDice"], False),
    (avg_row["TriPath_MASD"],  avg_row["QuadPath_MASD"],  True),
    (avg_row["TriPath_NSD"],   avg_row["QuadPath_NSD"],   False),
]

for m_idx, (tri_val, quad_val, lower_better) in enumerate(avg_vals):
    tri_better  = (tri_val < quad_val) if lower_better else (tri_val > quad_val)
    quad_better = (quad_val < tri_val) if lower_better else (quad_val > tri_val)
    tri_j  = 1 + m_idx
    quad_j = 5 + m_idx

    ax.text(col_centers[tri_j], avg_y, f"{tri_val:.4f}", ha="center", va="center",
            fontsize=VALUE_FONT,
            fontweight="bold" if tri_better else "normal",
            color=BEST_COLOR if tri_better else "#2c3e50")

    ax.text(col_centers[quad_j], avg_y, f"{quad_val:.4f}", ha="center", va="center",
            fontsize=VALUE_FONT,
            fontweight="bold" if quad_better else "normal",
            color=BEST_COLOR if quad_better else "#2c3e50")

# Outer border
for spine in ["top", "bottom", "left", "right"]:
    ax.spines[spine].set_visible(False)
ax.add_patch(plt.Rectangle((0, 0), n_cols, n_rows + 1,
             fill=False, edgecolor="#2c3e50", linewidth=2, zorder=4))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "07_paper_results_table.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 07_paper_results_table")