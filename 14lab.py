import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 1.0

Nz = 50
Nt = 400

dz = L / Nz
dt = T / Nt

z = np.linspace(0.0, L, Nz + 1)
t = np.linspace(0.0, T, Nt + 1)

c = 1.0
lam = 1.0

courant = c * dt / dz
print("Courant =", courant)


def p_true_func(z):
    return 2.0 + np.sin(2.0 * np.pi * z)


p_true = p_true_func(z)


def solve_forward(p):
    v_prev = np.zeros(Nz + 1)
    v_cur = np.zeros(Nz + 1)
    v = np.zeros((Nt + 1, Nz + 1))

    v[0, :] = v_prev.copy()

    for j in range(1, Nz):
        vxx = (v_prev[j + 1] - 2.0 * v_prev[j] + v_prev[j - 1]) / dz**2
        v_cur[j] = v_prev[j] + 0.5 * dt**2 * (c**2 * vxx - p[j] * v_prev[j])

    v_cur[0] = 0.0
    v_cur[Nz] = 0.0

    v[1, :] = v_cur.copy()

    for n in range(1, Nt):
        v_next = np.zeros(Nz + 1)

        for j in range(1, Nz):
            vxx = (v_cur[j + 1] - 2.0 * v_cur[j] + v_cur[j - 1]) / dz**2
            v_next[j] = (
                2.0 * v_cur[j]
                - v_prev[j]
                + dt**2 * (c**2 * vxx - p[j] * v_cur[j])
            )

        v_next[0] = 0.0
        v_next[Nz] = 0.0

        v[n + 1, :] = v_next

        v_prev, v_cur = v_cur, v_next

    return v


v_true = solve_forward(p_true)
f_meas = v_true[:, 0].copy()


def functional_J(v, f):
    diff = v[:, 0] - f
    return np.trapz(diff * diff, t)


def solve_adjoint(p, v, f):
    phi_next = np.zeros(Nz + 1)
    phi_cur = np.zeros(Nz + 1)
    phi = np.zeros((Nt + 1, Nz + 1))

    phi[Nt, :] = phi_next.copy()

    for j in range(1, Nz):
        vxx = (phi_next[j + 1] - 2.0 * phi_next[j] + phi_next[j - 1]) / dz**2
        phi_cur[j] = phi_next[j] + 0.5 * dt**2 * (c**2 * vxx - p[j] * phi_next[j])

    phi_cur[0] = 2.0 * (v[Nt - 1, 0] - f[Nt - 1])
    phi_cur[Nz] = 0.0

    phi[Nt - 1, :] = phi_cur.copy()

    for n in range(Nt - 1, 0, -1):
        phi_prev = np.zeros(Nz + 1)

        for j in range(1, Nz):
            vxx = (phi_cur[j + 1] - 2.0 * phi_cur[j] + phi_cur[j - 1]) / dz**2
            phi_prev[j] = (
                2.0 * phi_cur[j]
                - phi_next[j]
                + dt**2 * (c**2 * vxx - p[j] * phi_cur[j])
            )

        phi_prev[0] = 2.0 * (v[n - 1, 0] - f[n - 1])
        phi_prev[Nz] = 0.0

        phi[n - 1, :] = phi_prev

        phi_next, phi_cur = phi_cur, phi_prev

    return phi


def compute_gradient(p, v, phi):
    grad = np.zeros_like(p)
    for j in range(1, Nz):
        integrand = v[:, j] * phi[:, j]
        grad[j] = np.trapz(integrand, t)
    grad[0] = 0.0
    grad[-1] = 0.0
    return grad


p = np.ones_like(z) * 1.0
alpha = 0.1
max_iter = 30

history_J = []

for it in range(max_iter):
    v = solve_forward(p)
    J_val = functional_J(v, f_meas)
    history_J.append(J_val)

    print(f"iter {it:02d}: J = {J_val:.6e}")

    phi = solve_adjoint(p, v, f_meas)
    grad = compute_gradient(p, v, phi)

    p = p - alpha * grad

plt.figure()
plt.plot(z, p_true, label="p_true")
plt.plot(z, p, "--", label="p_recovered")
plt.xlabel("z")
plt.ylabel("p(z)")
plt.legend()
plt.grid(True)
plt.title("Восстановление коэффициента p(z)")
plt.show()

plt.figure()
plt.semilogy(history_J, "-o")
plt.xlabel("итерация")
plt.ylabel("J")
plt.grid(True)
plt.title("Снижение функционала J(p)")
plt.show()
