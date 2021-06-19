import numpy as np
from scipy.constants import c


def iid(dim, var=1) -> np.ndarray:
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


M = 32
T = 5
K = 100

eps_a = 0.1
a_K = np.random.choice([1, 0], p=[eps_a, 1 - eps_a], size=K)
d_k = np.random.uniform(10, 100, size=K)
rho_k = 0.01  # W = 10 dBm
lambda_c = 868e6 / c
beta_k = (lambda_c / (4 * np.pi * d_k)) ** 2  # Free space
S_TK = iid(dim=(T, K))

gamma_K = a_K * np.sqrt(rho_k)

gamma_K = np.array([gamma_K]).T  # to have a column vector

is_illcond(S_TK)

G = iid((M, K)) * np.sqrt(beta_k)

d_s_array = []
for t in range(T):
    d_st = np.zeros((K, K), complex)
    np.fill_diagonal(d_st, S_TK[t, :])
    d_s_array.append(G @ d_st)

Gamma = np.row_stack(d_s_array)
is_illcond(Gamma)

noise_var = 10 ** (-122 / 10)
W = iid(M * T, var=noise_var)  # thermal noise of 125khz signal @25°C
Y = Gamma @ gamma_K + np.array([W]).T

gamma_hat = np.abs(np.linalg.pinv(Gamma) @ Y)

a_k_found = np.zeros(K, int)

idx = gamma_hat.flatten() > noise_var + rho_k
a_k_found[idx] = 1

num_correct_found = (a_K.size - np.count_nonzero(a_K - a_k_found)) / a_K.size
print(f"{num_correct_found*100}%")

