# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.signal import find_peaks


def read_emission_file(path):
    metadata = {}
    data = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2 and not is_float(parts[0]):
                key, value = parts
                metadata[key] = value
                continue
            if len(parts) >= 2 and is_float(parts[0]) and is_float(parts[1]):
                data.append((float(parts[0]), float(parts[1])))

    # Replace label with formula if possible
    label = metadata.get("Labels", "")
    for sample_key in sample_map:
        if sample_key in label:
            metadata["Labels"] = sample_map[sample_key]
            break

    return metadata, np.array(data)


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root

DATA_DIR = os.path.join(BASE_DIR, "data", "luminiscence")

em_file_names = os.listdir(DATA_DIR)

# Mapping Sample-# → formula
sample_map = {
    "Sample-1": "Y_3Al_{5}O_{12}",
    "Sample-2": "Y_{2.25}Lu_{0.75}Al_{5}O_{12}",
    "Sample-3": "Y_{1.5}Lu_{1.5}Al_{5}O_{12}",
    "Sample-4": "Y_{0.75}Lu_{2.25}Al_{5}O_{12}",
    "Sample-5": "Lu_{3}Al_{5}O_{12}",
}


# %%
# [func(elem) for elem in iterable if cond(elem)]
em_files = [read_emission_file(os.path.join(DATA_DIR, f)) for f in em_file_names]

excitation_data = [f for f in em_files if "Excitation" in f[0]["Type"]]
emission_data = [f for f in em_files if "Emission" in f[0]["Type"]]


# --- Emission data plot + peaks ---
plt.figure(dpi=200)
for f in emission_data:
    metadata = f[0]
    data = f[1]

    x, y = data[:, 0], data[:, 1]
    y_norm = y / np.max(data[:, 1])
    # smooth spectrum to find peaks
    denoised = savgol_filter(y_norm, window_length=20, polyorder=3)
    peaks, _ = find_peaks(denoised)

    # main peak (highest)
    main_peak_idx = peaks[np.argmax(denoised[peaks])]
    peak_x = x[main_peak_idx]
    peak_y = denoised[main_peak_idx]

    # plot data
    plt.plot(x, y_norm, label=rf"${metadata['Labels']}$")
    # plt.plot(x, denoised, label=rf"${metadata['Labels']}$")
    # mark peak
    plt.plot(peak_x, peak_y, "o", c="black")

    print(f"{metadata['Labels']}: Emission peak at {peak_x:.1f} nm")

plt.legend()
plt.title("Emission data")
plt.savefig(os.path.join(BASE_DIR, "figures", "luminiscence", "emission.png"))
plt.show()

# Not normalized spectrum, not useful
# plt.figure(dpi=200)
# for f in excitation_data:
#     metadata = f[0]
#     data = f[1]
#     plt.plot(data[:, 0], data[:, 1], label=rf"${metadata['Labels']}$")
# plt.title("Excitation data")
# plt.legend()
# plt.show()


plt.figure(dpi=200)
for f in excitation_data:
    metadata = f[0]
    data = f[1]

    mask = data[:, 0] > 250
    x, y = data[mask, 0], data[mask, 1]
    y_norm = y / np.max(np.abs(y))

    # smooth for peak detection
    denoised = savgol_filter(y_norm, window_length=10, polyorder=2)

    # find all peaks
    peaks, _ = find_peaks(denoised)

    # split peaks into two regions
    peaks_low = peaks[x[peaks] < 380]
    peaks_high = peaks[x[peaks] >= 380]

    # pick the highest peak in each region
    if len(peaks_low) > 0:
        idx_low = peaks_low[np.argmax(denoised[peaks_low])]
        peak_x_low = x[idx_low]
        peak_y_low = denoised[idx_low]
    else:
        peak_x_low, peak_y_low = None, None

    if len(peaks_high) > 0:
        idx_high = peaks_high[np.argmax(denoised[peaks_high])]
        peak_x_high = x[idx_high]
        peak_y_high = denoised[idx_high]
    else:
        peak_x_high, peak_y_high = None, None

    # plot normalized spectrum
    plt.plot(x, y_norm, label=rf"${metadata['Labels']}$")
    if peak_x_low is not None:
        plt.plot(peak_x_low, peak_y_low, "o", color="black", markersize=5)
    if peak_x_high is not None:
        plt.plot(peak_x_high, peak_y_high, "o", color="black", markersize=5)

    peaks_report = []
    if peak_x_low is not None:
        peaks_report.append(f"{peak_x_low:.1f}")
    if peak_x_high is not None:
        peaks_report.append(f"{peak_x_high:.1f}")
    print(f"{metadata['Labels']}: Excitation peaks at {peaks_report} nm")

plt.title("Excitation data (Normalized)")
plt.legend(loc="best")
plt.savefig(os.path.join(BASE_DIR, "figures", "luminiscence", "norm_excitation.png"))
plt.show()

"""
results:

| Sample Formula    | Emission Peak (nm) | Excitation Peak 1 (nm) | Excitation Peak 2 (nm) |
| ----------------- | ------------------ | ---------------------- | ---------------------- |
| Y₃Al₅O₁₂          | 553.0              | 341.0                  | 445.0                  |
| Y₂.₂₅Lu₀.₇₅Al₅O₁₂ | 550.0              | 344.0                  | 451.0                  |
| Y₁.₅Lu₁.₅Al₅O₁₂   | 543.0              | 344.0                  | 449.0                  |
| Y₀.₇₅Lu₂.₂₅Al₅O₁₂ | 540.0              | 346.0                  | 445.0                  |
| Lu₃Al₅O₁₂         | 522.0              | 349.0                  | 444.0                  |


"""
