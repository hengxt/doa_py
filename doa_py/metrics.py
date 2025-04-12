import numpy as np
from itertools import permutations


def CRLB_single_angle(num_antennas, antenna_spacing, angle, snr, snapshots):
    m = num_antennas  # 阵列元素个数
    d = antenna_spacing  # 阵列元素间距
    wavelength = d * 2  # 波长
    snr = (10 ** (snr / 10))
    n = snapshots

    tmp1 = 3 * (wavelength ** 2)
    tmp2 = 2 * (np.pi ** 2) * (d ** 2) * (m ** 2 - 1)
    tmp3 = m * n * snr
    tmp4 = np.cos(angle) ** 2
    return np.sqrt(tmp1 / (tmp2 * tmp3 * tmp4))


def CRLB2D(num_antennas, antenna_spacing, angle, snr, snapshots):
    """Compute the DoA's Cramer-Rao Lower Bound.

    Args:
        num_antennas : the number of antennas in ULA
        antenna_spacing : the spacing of antennas in ULA
        angle : True DoA (radians)
        snr : Signal-to-noise ratio (dB)
        snapshots : Number of snapshots
    Return:
        crlb -- CRLB value
    """
    m = num_antennas  # 阵列元素个数
    d = antenna_spacing  # 阵列元素间距
    array = np.arange(0, m * d, d)

    wavelength = d * 2  # 波长
    k = len(angle)  # 信号源个数
    sigma2 = 1 / (10 ** (snr / 10))  # 噪声方差
    n = snapshots  # 快拍数

    # 构造方向矢量
    A = np.zeros((m, k), dtype=complex)
    for i in range(k):
        A[:, i] = np.exp(-1j * 2 * np.pi * array * np.sin(angle[i]) / wavelength)

    # 构造方向矢量对角度的导数 (D矩阵)
    D = np.zeros((m, k), dtype=complex)
    for i in range(k):
        D[:, i] = (-1j * 2 * np.pi * array * np.cos(angle[i]) / wavelength) * A[:, i]

    # 构造信号协方差矩阵R
    R = np.dot(A, A.conj().T) + sigma2 * np.eye(m)

    # 构造Fisher信息矩阵 (考虑信号相关性)
    FIM = np.zeros((k, k), dtype=complex)
    for i in range(k):
        for j in range(k):
            # 使用理论中的公式计算FIM
            temp1 = np.dot(np.linalg.inv(R), D[:, i])
            temp2 = np.dot(np.linalg.inv(R), D[:, j])
            FIM[i, j] = n * np.trace(np.outer(temp1, temp2.conj()))

    # 计算CRLB
    crlb = np.real(np.diag(np.linalg.inv(FIM)))

    return crlb


def RMSPE(
        theta_true,
        theta_pred
):
    """Compute the Root Mean Squared Periodic Error (RMSPE) between true and predicted DoA angles.

    Args:
        theta_true (np.ndarray): True DoA angles, shape (d,).
        theta_pred (np.ndarray): Predicted DoA angles, shape (d,).

    Returns:
        float: RMSPE value.
    """

    def mod_pi(angle):
        """
        Compute the periodic difference of an angle, wrapping it to [-pi, pi].

        Args:
            angle (float or np.ndarray): Angle or array of angles in radians.

        Returns:
            float or np.ndarray: Wrapped angle(s) in [-pi, pi].
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    # Ensure inputs are numpy arrays
    theta_true = np.array(theta_true)
    theta_pred = np.array(theta_pred)

    # Check if dimensions match
    d = len(theta_true)
    if len(theta_pred) != d:
        raise ValueError("The length of theta and theta_hat must be the same.")

    # Initialize minimum mean squared error
    min_mse = float('inf')

    # Generate all permutations of indices [0, 1, ..., d-1]
    for perm in permutations(range(d)):
        # Permute theta_hat according to the current permutation
        theta_hat_perm = theta_pred[list(perm)]

        # Compute the periodic difference
        diff = mod_pi(theta_true - theta_hat_perm)

        # Compute the mean squared error for this permutation
        mse = np.mean(diff ** 2)

        # Update the minimum MSE if this permutation yields a smaller error
        min_mse = min(min_mse, mse)

    # Return the RMSPE (square root of the minimum MSE)
    return np.sqrt(min_mse)
