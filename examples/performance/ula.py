import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from doa_py.algorithm import esprit, l1_svd, music, omp, root_music
from doa_py.tools import find_angles_from_spectrum
from doa_py.arrays import UniformLinearArray
from doa_py.signals import ComplexStochasticSignal
from doa_py.metrics import RMSPE, CRLB2D, CRLB_single_angle
from tqdm import tqdm

np.random.seed(42)
# signal parameters
num_snapshots = 1000
signal_fre = 3e8

# array parameters
num_antennas = 16
antenna_spacing = 0.5 * (
        3e8 / signal_fre
)  # set array spacing to half wavelength

# incident angles
angle_incidence = np.array([10.11, 13.3])
num_signal = len(angle_incidence)
# initialize signal instance
signal = ComplexStochasticSignal(fc=signal_fre)

# initialize array instance
array = UniformLinearArray(m=num_antennas, dd=antenna_spacing)

# search grids
search_grids = np.arange(-90, 90, 1)
snr_range = np.arange(-20, 31, 5)
result = pd.DataFrame(
    columns=["MUSIC", "ROOT-MUSIC", "ESPRIT", "OMP", "L1_SVD", "CRLB"],
    index=snr_range,
    data=np.zeros((len(snr_range), 6)),
)
# 蒙特卡罗次数
num_mc = 1000

for snr in tqdm(snr_range, desc="SNR", leave=True):
    for i in tqdm(range(num_mc), desc="MC", leave=False):
        # generate received data
        received_data = array.received_signal(
            signal=signal,
            snr=snr,
            nsamples=num_snapshots,
            angle_incidence=angle_incidence,
        )

        # MUSIC
        music_spectrum = music(
            received_data=received_data,
            num_signal=num_signal,
            array=array,
            signal_fre=signal_fre,
            angle_grids=search_grids,
        )
        music_estimates = find_angles_from_spectrum(
            spectrum=music_spectrum,
            angle_grids=search_grids,
            num_signal=num_signal,
        )

        # ROOT-MUSIC
        rmusic_estimates = root_music(
            received_data=received_data,
            num_signal=num_signal,
            array=array,
            signal_fre=signal_fre,
        )

        # ESPRIT
        esprit_estimates = esprit(
            received_data=received_data,
            num_signal=num_signal,
            array=array,
            signal_fre=signal_fre,
        )

        # OMP
        omp_estimates = omp(
            received_data=received_data,
            num_signal=num_signal,
            array=array,
            signal_fre=signal_fre,
            angle_grids=search_grids,
        )

        # L1-SVD
        l1_svd_spectrum = l1_svd(
            received_data=received_data,
            num_signal=num_signal,
            array=array,
            signal_fre=signal_fre,
            angle_grids=search_grids,
        )
        l1_svd_estimates = find_angles_from_spectrum(
            spectrum=l1_svd_spectrum,
            angle_grids=search_grids,
            num_signal=num_signal,
        )

        crlb = CRLB_single_angle(
            num_antennas=num_antennas,
            antenna_spacing=antenna_spacing,
            angle=np.deg2rad(angle_incidence),
            snr=snr,
            snapshots=num_snapshots,
        )

        result.loc[snr, "MUSIC"] += RMSPE(angle_incidence, music_estimates)
        result.loc[snr, "ROOT-MUSIC"] += RMSPE(angle_incidence, rmusic_estimates)
        result.loc[snr, "ESPRIT"] += RMSPE(angle_incidence, esprit_estimates)
        result.loc[snr, "OMP"] += RMSPE(angle_incidence, omp_estimates)
        result.loc[snr, "L1_SVD"] += RMSPE(angle_incidence, l1_svd_estimates)
        result.loc[snr, "CRLB"] += np.mean(crlb * 180 / np.pi)
    result.loc[snr] = result.loc[snr] / num_mc

import matplotlib.pyplot as plt

ax = result.plot(kind="line", marker="o", title="ULA")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("RMSPE (degree)")
ax.set_yscale("log")
ax.legend()
plt.show()
plt.savefig("ula.svg")
plt.close()