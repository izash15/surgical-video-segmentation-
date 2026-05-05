import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "/home/rmosad/surgical-video-segmentation-/experiments/plots/"
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
# Per-organ mIoU results
# ----------------------------------------------------
data = pd.DataFrame({
    "Organ": [
        "abdominal_wall", "colon", "inferior_mesenter", "intestinal_veins",
        "liver", "pancreas", "small_intestine", "spleen",
        "stomach", "ureter", "vesicular_glands"
    ],
    "QuadPath": [
        0.8232, 0.7026, 0.4048, 0.4183, 0.6568, 0.2644,
        0.8078, 0.7443, 0.6526, 0.3997, 0.3854
    ],
    "TriPathEB-WF Original": [
        0.760, 0.568, 0.269, 0.315, 0.603, 0.192,
        0.719, 0.661, 0.542, 0.145, 0.215
    ]
})

# ----------------------------------------------------
# Plot 1: Boxplot comparing mIoU distribution across organs
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 10))

box_data = [
    data["TriPathEB-WF Original"],
    data["QuadPath"]
]

box = ax.boxplot(
    box_data,
    tick_labels=["TriPathEB-WF\nOriginal", "QuadPath"],
    patch_artist=True,
    widths=0.55,
    showmeans=True
)

# Make boxplot lines thicker
for element in ["boxes", "whiskers", "caps", "medians", "means"]:
    for item in box[element]:
        item.set_linewidth(3)

# Add jittered points so the viewer can see each organ result
for i, model in enumerate(["TriPathEB-WF Original", "QuadPath"], start=1):
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

# Add mean labels
tri_mean = data["TriPathEB-WF Original"].mean()
quad_mean = data["QuadPath"].mean()

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
plt.savefig(os.path.join(output_dir, "01_miou_distribution_boxplot.png"), dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# Plot 2: Per-organ mIoU comparison using grouped bars
# This avoids implying a continuous trend across organs.
# ----------------------------------------------------
x = np.arange(len(data["Organ"]))
width = 0.38

fig, ax = plt.subplots(figsize=(24, 12))

bars1 = ax.bar(
    x - width/2,
    data["TriPathEB-WF Original"],
    width,
    label="TriPathEB-WF Original"
)

bars2 = ax.bar(
    x + width/2,
    data["QuadPath"],
    width,
    label="QuadPath"
)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            rotation=90
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
plt.savefig(os.path.join(output_dir, "02_per_organ_miou_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------
# Plot 3: Per-organ improvement as a bar chart
# This replaces the previous line graph for improvement.
# ----------------------------------------------------
data["mIoU Improvement (%)"] = (
    (data["QuadPath"] - data["TriPathEB-WF Original"])
    / data["TriPathEB-WF Original"]
    * 100
)

fig, ax = plt.subplots(figsize=(24, 12))

bars = ax.bar(
    data["Organ"],
    data["mIoU Improvement (%)"],
    width=0.65
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 4,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=22,
        fontweight="bold"
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
plt.savefig(os.path.join(output_dir, "03_per_organ_improvement.png"), dpi=300, bbox_inches='tight')
plt.show()
