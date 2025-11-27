# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks

# -------------------------------------------------------
# Load sample XRD files
# -------------------------------------------------------
sample_path = "xrd"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root

DATA_DIR = os.path.join(BASE_DIR, "data")
XRD_DIR = os.path.join(DATA_DIR, "xrd")

sample_files = [f for f in os.listdir(XRD_DIR) if "sample" in f]
sample_data = [(f, np.loadtxt(os.path.join(XRD_DIR, f))) for f in sample_files]

# Mapping Sample-# → formula
sample_map = {
    "1": "Y_3Al_{5}O_{12}",
    "2": "Y_{2.25}Lu_{0.75}Al_{5}O_{12}",
    "3": "Y_{1.5}Lu_{1.5}Al_{5}O_{12}",
    "4": "Y_{0.75}Lu_{2.25}Al_{5}O_{12}",
    "5": "Lu_{3}Al_{5}O_{12}",
}


# -------------------------------------------------------
# Load reference XRD PDF (No, 2θ, d, I, h, k, l)
# -------------------------------------------------------
def load_reference_pdf(path):
    """
    Load reference XRD PDF from Excel.
    Returns:
        two_theta: array of 2θ values
        intensity: array of intensities
        numbers: array of peak numbers from 'No' column
    """
    df = pd.read_excel(os.path.join(XRD_DIR, path))

    # find column names automatically
    col_2theta = [c for c in df.columns if "2Theta" in c or "2theta" in c][0]
    col_I = [c for c in df.columns if "Int" in c][0]
    col_no = [c for c in df.columns if "No" in c or "no" in c][0]  # the enumerator

    two_theta = df[col_2theta].values
    intensity = df[col_I].values
    numbers = df[col_no].values

    return two_theta, intensity, numbers


def get_sample_name(filename):
    """Return formula based on detecting sample number inside filename."""
    for num in sample_map:
        if f"sample_{num}" in filename.lower():
            return sample_map[num]
    return "Unknown sample"


ref_yag_path = "cubic-Y3Al5O12-PDF 01-075-6655.xlsx"
ref_lug_path = "cubic-Lu3Al5O12-PDF 04-027-8643.xlsx"

ref_yag = load_reference_pdf(ref_yag_path)
ref_lug = load_reference_pdf(ref_lug_path)


# -------------------------------------------------------
# Plot each sample with peaks & references
# -------------------------------------------------------
for fname, data in sample_data:
    x = data[:, 0]
    y = data[:, 1]

    # find peaks in sample
    peaks, _ = find_peaks(y, prominence=0.01 * np.max(y))

    # --------------------------
    plt.figure(figsize=(10, 4), dpi=200)
    # --------------------------

    sample_name = get_sample_name(fname)
    plt.plot(x, y, label=rf"${sample_name}$", color="blue")

    # label sample peaks
    for p in peaks:
        px = x[p]
        py = y[p]
        plt.axvline(px, color="blue", alpha=0.4, linewidth=0.5)
        plt.text(
            px + 0.1,
            py,
            f"{px:.2f}°",
            rotation=90,
            va="bottom",
            fontsize=6,
            color="blue",
        )

    # plot reference YAG
    ry_x, ry_y, ry_no = ref_yag
    for xx, yy, n in zip(ry_x, ry_y, ry_no):
        # vertical line for reference peak
        plt.vlines(
            xx,
            0,
            np.max(y) * (yy / np.max(ry_y)),
            color="green",
            alpha=0.6,
        )

        plt.text(
            xx,  # use the reference peak position
            np.max(y) * -0.02,  # just below the line
            f"{n}",  # label
            rotation=90,
            fontsize=3,
            color="green",
            ha="center",
        )
    # label for ref
    plt.plot([], [], color="green", alpha=0.6, label="YAG reference")

    rl_x, rl_y, rl_no = ref_lug
    for xx, yy, n in zip(rl_x, rl_y, rl_no):

        # vertical reference line
        plt.vlines(
            xx,
            0,
            np.max(y) * (yy / np.max(rl_y)),
            color="red",
            alpha=0.6,
        )

        plt.text(
            xx,
            np.max(y) * -0.02,  # just above the line
            f"{n}",
            rotation=90,
            fontsize=3,
            color="red",
            ha="center",
        )
    # label for ref
    plt.plot([], [], color="red", alpha=0.6, label="LuAG reference")

    figfilename = f"{fname[:8]}_fig.png"

    plt.title(rf"XRD: ${sample_name}$")
    plt.xlabel("2θ (deg)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figures", "xrd", figfilename))
    plt.show()


# calculation example for one peak
# need to get theta from graph and find the peak from the reference.
# the miller indices are only in the .xslx files for now

two_theta = 18.16  # degrees
h, k, l = 1, 1, 2
lam = 1.5406  # Å

theta = np.deg2rad(two_theta / 2)
a = lam * np.sqrt(h**2 + k**2 + l**2) / (2 * np.sin(theta))
d_hkl = a / np.sqrt(h**2 + k**2 + l**2)

print(f"a = {a:.5f} Å,    d_hkl = {d_hkl:.5f} Å")
