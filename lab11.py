import numpy as np
import matplotlib.pyplot as plt

L = 1.0
a = 1.0

N = 50
h = L / N

T0 = L / a
tau0 = h / a

tau = 0.5 * tau0
T = 5.0 * T0
K = int(T / tau)

M = 20

c = a * tau / h
print(f"c = {c:.3f}  (для устойчивости нужно |c| < 1)")


def mu1(x):
    return 0.0 * x

def mu2(x):
    return (np.pi * a / L) * np.sin(np.pi * x / L)

def mu3(t):
    return 0.0

def mu4(t):
    return 0.0

def f_func(x, t):
    return 0.0 * x


def u_exact(x, t):
    return np.sin(np.pi * x / L) * np.sin(np.pi * a * t / L)


x = np.linspace(0.0, L, N + 1)

u_prev = np.zeros(N + 1)
u_cur = np.zeros(N + 1)
u_next = np.zeros(N + 1)

for j in range(N + 1):
    u_prev[j] = mu1(x[j])

u_prev[0] = mu3(0.0)
u_prev[-1] = mu4(0.0)

t = tau

u_cur[0] = mu3(t)
u_cur[-1] = mu4(t)

for j in range(1, N):
    uxx0 = (u_prev[j + 1] - 2.0 * u_prev[j] + u_prev[j - 1]) / h**2
    u_cur[j] = (
        u_prev[j]
        + tau * mu2(x[j])
        + 0.5 * tau**2 * (a**2 * uxx0 + f_func(x[j], 0.0))
    )

snapshots = []
times = []

snapshots.append(u_prev.copy())
times.append(0.0)
snapshots.append(u_cur.copy())
times.append(t)

for n in range(1, K):
    t_n = n * tau
    t_np1 = (n + 1) * tau

    for j in range(1, N):
        u_next[j] = (
            2.0 * (1.0 - c**2) * u_cur[j]
            - u_prev[j]
            + c**2 * (u_cur[j + 1] + u_cur[j - 1])
            + tau**2 * f_func(x[j], t_n)
        )

    u_next[0] = mu3(t_np1)
    u_next[-1] = mu4(t_np1)

    if n % M == 0:
        snapshots.append(u_next.copy())
        times.append(t_np1)

    u_prev, u_cur, u_next = u_cur, u_next, u_prev


t_final = K * tau
u_true_final = u_exact(x, t_final)
max_err = np.max(np.abs(u_cur - u_true_final))

print(f"\nФинальное время: t_final = {t_final:.4f}")
print(f"Максимальная погрешность (∞-норма): {max_err:.3e}")

plt.figure()
for u, tt in zip(snapshots, times):
    plt.plot(x, u, label=f"t = {tt:.2f}")

plt.plot(x, u_exact(x, times[-1]), '--', label=f"u_exact(t={times[-1]:.2f})")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Численное решение волнового уравнения (явная трёхслойная схема)")
plt.grid(True)
plt.legend()
plt.show()
