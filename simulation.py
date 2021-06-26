import numpy as np


def iid(dim, var=1.0) -> np.ndarray:
    """
    return iid CN(0,var) with dimensions dim
    """
    return (1 / np.sqrt(2)) * np.random.normal(size=dim, scale=np.sqrt(var)) + 1j * (1 / np.sqrt(2)) * np.random.normal(
        size=dim, scale=np.sqrt(var))


def is_illcond(arr: np.ndarray):
    """
    Check if a matrix/arr is ill conditioned or not
    :param arr:
    :return:
    """
    print(1.0 / np.linalg.cond(arr))
    # A problem with a low condition number is said to be well-conditioned,
    # while a problem with a high condition number is said to be ill-conditioned.
    if np.linalg.cond(arr) < (1.0 / np.finfo(complex).eps):
        return False
    return True


def is_diag(arr):
    i, j = arr.shape
    assert i == j
    test = arr.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])


def prob_miss(arr, arr_est) -> float:
    num_active = np.sum(arr)
    # | A union A_hat | is equal to sum(arr AND arr_est), or only when both are 1 (active) -> 1
    num_correct = np.sum(arr_est[arr == 1])
    return 1.0 - (num_correct / num_active)


def prob_false(arr, arr_est) -> float:
    # num inactive devices arr=0 detected as active arr_est = 1
    num_active = np.sum(arr)
    num_users = arr.size
    p_false = np.sum(arr_est[arr == 0]) / (num_users - num_active)
    return p_false


def beta(d, model="oulu", sigma=0):
    d = d / 1000

    if model in "oulu":
        pl0 = 128.95
        n = 2.32
        sigma = 7.8
    elif model in "dortmund":
        pl0 = 132.25
        n = 2.65
    elif model in "three-slope":
        d = d * 1000  # here expressed in meters
        if d < 10:
            return 10 ** (-81.2 / 10)
        elif d < 50:
            return 10 ** ((-61.2 - 20 * np.log10(d)) / 10)
        else:
            return 10 ** ((-35.7 - 35 * np.log10(d) + np.random.normal(scale=8)) / 10)
    else:
        return ValueError

    pl_db = pl0 + 10 * n * np.log10(d) + np.random.normal(scale=sigma)
    return 10 ** (- pl_db / 10)


M = 64
T = 40
K = 400
eps_a = 0.1
rho_k = 1
noise_var = (10 ** (-109 / 10)) * 1000 #-109dBm

num_sim = 100
x_avg = []
y_avg = []

import tqdm

for n_sim in tqdm.trange(num_sim):

    a_k = np.random.choice([1, 0], p=[eps_a, 1 - eps_a], size=K)
    d_k = np.random.uniform(1, 250, size=K)

    beta_k = [beta(d_k[k], model="three-slope") for k in range(K)]
    S_TK = iid(dim=(T, K))

    gamma_K = a_k * np.sqrt(rho_k)
    gamma_K = np.array([gamma_K]).T  # to have a column vector

    # is_illcond(S_TK)

    G = iid((M, K)) * np.vstack([np.sqrt(beta_k) for m in range(M)])

    d_s_array = []
    for t in range(T):
        d_st = np.zeros((K, K), complex)
        np.fill_diagonal(d_st, S_TK[t, :])
        d_s_array.append(G @ d_st)

    Gamma = np.row_stack(d_s_array)

    gamma_corr = np.conj(Gamma.T) @ Gamma

    res = is_diag(gamma_corr)

    # is_illcond(Gamma)

    W = iid(M * T, var=noise_var)  # thermal noise of 125khz signal @25°C
    Y = Gamma @ gamma_K + np.array([W]).T

    gamma_hat = np.abs(np.linalg.pinv(Gamma) @ Y)

    # spgl1

    # Basic pursuit denoise problem
    # min ||x||_1 s.t. ||y- Gamma * gamma||^2 <= error_th
    #gamma_hat_2 = spgl1.spg_bpdn(A=Gamma, b=Y.flatten(), sigma=noise_var)

    a_k_estimate = np.zeros(K, int)

    # v_arr = (gamma_hat.flatten()/noise_var)

    x = []
    y = []

    for v in np.linspace(0.0001, 10, 1000):
        gamma_th = np.array([np.linalg.norm(G[:, k]) ** 2 / noise_var for k in range(K)])
        a_k_estimate = np.zeros(K, int)
        a_k_estimate[gamma_hat.flatten() > gamma_th] = 1

        x.append(prob_false(a_k, a_k_estimate))
        y.append(prob_miss(a_k, a_k_estimate))

    x_avg.append(x)
    y_avg.append(y)

import matplotlib.pyplot as plt

x = np.average(x_avg, axis=0)
y = np.average(y_avg, axis=0)

plt.scatter(x, y)

plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.xlabel("Probability of False Alarm")
plt.ylabel("Probability of Miss Detection")
plt.show()
