import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("cnr_efficiency_table.csv")

# Filter flip angles and remove readouts
df = df[
    (df["flip_angle"] >= 10) &
    (df["flip_angle"] <= 90) &
    (~df["n_readout"].isin([15, 20, 30]))
]

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(14, 5),
    sharex=True,
    sharey=False
)

for readout, sub in df.groupby("n_readout"):

    sub = sub.sort_values("flip_angle")

    # -------------------
    # Myelin panel
    # -------------------
    ax1.plot(
        sub["flip_angle"],
        sub["snr_myelin"],
        color="tab:blue",
        marker="o",
        markersize=2,
        linewidth=2,
        alpha=0.8
    )

    idx_max_myelin = sub["snr_myelin"].idxmax()

    ax1.scatter(
        sub.loc[idx_max_myelin, "flip_angle"],
        sub.loc[idx_max_myelin, "snr_myelin"],
        color="red",
        edgecolor="black",
        linewidth=0.8,
        s=35,
        zorder=10,
    )

    # # Myelin: label readout number to the right of the last point
    # ax1.text(
    #     sub["flip_angle"].iloc[-1] + 1,
    #     sub["snr_myelin"].iloc[-1],
    #     str(readout),
    #     fontsize=10,
    #     color="tab:blue",
    #     va="center"
    # )

    # -------------------
    # White Matter panel
    # -------------------
    ax2.plot(
        sub["flip_angle"],
        sub["snr_wm"],
        color="tab:orange",
        marker="s",
        markersize=2,
        linewidth=2,
        alpha=0.8
    )

    idx_max_wm = sub["snr_wm"].idxmax()

    ax2.scatter(
        sub.loc[idx_max_wm, "flip_angle"],
        sub.loc[idx_max_wm, "snr_wm"],
        color="red",
        edgecolor="black",
        linewidth=0.8,
        s=35,
        zorder=10,
    )

    # White Matter: label readout number above the first point
    # ax2.annotate(
    #     str(readout),
    #     xy=(
    #         sub["flip_angle"].iloc[0],
    #         sub["snr_wm"].iloc[0]
    #     ),
    #     xytext=(4, 3),
    #     textcoords="offset points",
    #     fontsize=10,
    #     color="tab:orange",
    #     ha="center",
    #     va="bottom"
    # )

# Formatting
ax1.set_title("Myelin")
ax2.set_title("White Matter")

ax1.set_xlabel("Flip Angle (°)")
ax2.set_xlabel("Flip Angle (°)")

ax1.set_ylabel("SNR")
ax2.set_ylabel("SNR")

for ax in [ax1, ax2]:
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df["flip_angle"].unique()))

# Give space for Myelin right-side labels
ax1.set_xlim(10, 95)

# WM labels are at first point, so no need for large right margin
ax2.set_xlim(8, 92)

plt.suptitle(
    "SNR vs Flip Angle for Different Readouts",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "SNR_vs_FlipAngle_Myelin_WM_side_by_side.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()