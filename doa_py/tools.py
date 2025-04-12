import numpy as np


def find_angles_from_spectrum(
        spectrum,
        angle_grids,
        num_signal
):
    from scipy.signal import find_peaks

    """from spatial spectrum find angles

    Args:
        spectrum: Spatial spectrum estimated by the algorithm
        angle_grids: Angle grids corresponding to the spatial spectrum
        num_signal: Number of signals

    Return:
        angles: The angle that the spectrum finds is as it is
    """
    spectrum = spectrum / np.max(spectrum)
    # find peaks and peak heights
    peaks_idx, heights = find_peaks(spectrum, height=0)

    idx = heights["peak_heights"].argsort()[-num_signal:]
    peaks_idx = peaks_idx[idx]
    heights = heights["peak_heights"][idx]

    angles = angle_grids[peaks_idx]
    return angles
