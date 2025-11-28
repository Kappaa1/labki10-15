import numpy as np
import matplotlib.pyplot as plt

def k_fun(x):
    return np.ones_like(x, dtype=float)

def q_fun(x):
    return np.zeros_like(x, dtype=float)

def f_fun(x):
    return (np.pi ** 2) * np.sin(np.pi * x)

def u_exact(x):
    return np.sin(np.pi * x)

def solve_bvp(N):
    L = 1.0
    h = L / N
    x = np.linspace(0.0, L, N + 1)

    x_half_left = (np.arange(1, N) - 0.5) * h
    x_half_right = (np.arange(1, N) + 0.5) * h

    k_left = k_fun(x_half_left)
    k_right = k_fun(x_half_right)

    n = N - 1

    A = k_left / h**2
    B = k_right / h**2
    C = -(A + B) - q_fun(x[1:N])
    F = -f_fun(x[1:N])

    y0 = 0.0
    yN = 0.0

    F[0] -= A[0] * y0
    F[-1] -= B[-1] * yN

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    f = np.zeros(n)

    c[:] = C
    f[:] = F
    a[1:] = A[1:]
    b[:-1] = B[:-1]

    for i in range(1, n):
        m = a[i] / c[i - 1]
        c[i] -= m * b[i - 1]
        f[i] -= m * f[i - 1]

    y_inner = np.zeros(n)
    y_inner[-1] = f[-1] / c[-1]

    for i in range(n - 2, -1, -1):
        y_inner[i] = (f[i] - b[i] * y_inner[i + 1]) / c[i]

    y = np.zeros(N + 1)
    y[0] = y0
    y[N] = yN
    y[1:N] = y_inner

    return x, y

N = 50
x, y_num = solve_bvp(N)
y_true = u_exact(x)

max_err = np.max(np.abs(y_num - y_true))
print(f"N = {N}, max error = {max_err:.3e}")

plt.plot(x, y_num, "o-", label="численное")
plt.plot(x, y_true, "--", label="точное")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.title("ЛР 13")
plt.show()
