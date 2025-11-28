import numpy as np
import matplotlib.pyplot as plt

Lx = 1.0
Ly = 1.0

N = 50
M = 50

hx = Lx / N
hy = Ly / M

x = np.linspace(0.0, Lx, N + 1)
y = np.linspace(0.0, Ly, M + 1)

hx2 = hx * hx
hy2 = hy * hy
d = 1.0 / (2.0 * (hx2 + hy2))


def u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f_func(x, y):
    coef = (np.pi**2 / Lx**2 + np.pi**2 / Ly**2)
    return coef * u_exact(x, y)

def psi(x, y):
    return u_exact(x, y)


u = np.zeros((M + 1, N + 1))

for i in range(M + 1):
    u[i, 0] = psi(0.0, y[i])
    u[i, N] = psi(Lx, y[i])
for j in range(N + 1):
    u[0, j] = psi(x[j], 0.0)
    u[M, j] = psi(x[j], Ly)

f_grid = np.zeros((M + 1, N + 1))
for i in range(1, M):
    for j in range(1, N):
        f_grid[i, j] = f_func(x[j], y[i])

lambda_max = (hx2 * np.cos(np.pi / N) + hy2 * np.cos(np.pi / M)) / (hx2 + hy2)
omega = 2.0 / (1.0 + np.sqrt(1.0 - lambda_max**2))

print(f"lambda_max = {lambda_max:.6f}")
print(f"omega      = {omega:.6f}")

eps = 1e-6
max_iter = 100000
iter_num = 0

while True:
    iter_num += 1
    max_diff = 0.0

    for i in range(1, M):
        for j in range(1, N):
            u_old = u[i, j]

            rhs = d * (
                hx2 * (u[i, j + 1] + u[i, j - 1]) +
                hy2 * (u[i + 1, j] + u[i - 1, j]) +
                hx2 * hy2 * f_grid[i, j]
            )

            u_new = (1.0 - omega) * u_old + omega * rhs
            u[i, j] = u_new

            diff = abs(u_new - u_old)
            if diff > max_diff:
                max_diff = diff

    if max_diff < eps or iter_num >= max_iter:
        break

print(f"Итераций: {iter_num}")
print(f"max |Δu| = {max_diff:.3e}")

u_true = np.zeros_like(u)
for i in range(M + 1):
    for j in range(N + 1):
        u_true[i, j] = u_exact(x[j], y[i])

err = np.max(np.abs(u - u_true))
print(f"Максимальная погрешность: {err:.3e}")

X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(X, Y, u, rstride=1, cstride=1)
ax1.set_title("Численное решение")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, u_true, rstride=1, cstride=1)
ax2.set_title("Точное решение")

plt.tight_layout()
plt.show()
