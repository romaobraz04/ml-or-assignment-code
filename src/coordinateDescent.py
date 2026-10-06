import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

print("Running coordinate descent comparison...")
data = pd.read_csv("data_2026.csv")

train = data.iloc[:200].copy()

outliers = [27, 31, 50, 68, 103, 127, 172, 175, 193]

train_cleaned = train.drop(train.index[outliers])

X_train_cleaned = train_cleaned.drop(columns="y")
y_train_cleaned = train_cleaned["y"].to_numpy()

scaler = StandardScaler()
X_train_cleaned = scaler.fit_transform(X_train_cleaned)


def lasso_cd(X, y, lam, tol, max_iter=10000):
    n, p = X.shape
    beta = np.zeros(p)

    def objective(beta):
        return 0.5 * np.sum((y - X @ beta) ** 2) + lam * np.sum(np.abs(beta))

    history = [objective(beta)]

    for _ in range(max_iter):
        old_obj = history[-1]

        for j in range(p):
            xj = X[:, j]
            r = y - X @ beta + xj * beta[j]
            c = xj @ r
            a = xj @ xj

            if c > lam:
                beta[j] = (c - lam) / a
            elif c < -lam:
                beta[j] = (c + lam) / a
            else:
                beta[j] = 0

        new_obj = objective(beta)
        history.append(new_obj)

        if old_obj - new_obj < tol:
            break

    return beta, history


def lasso_random_cd(X, y, lam, tol, max_iter=10000):
    n, p = X.shape
    beta = np.zeros(p)

    def objective(beta):
        return 0.5 * np.sum((y - X @ beta) ** 2) + lam * np.sum(np.abs(beta))

    history = [objective(beta)]

    for _ in range(max_iter):
        old_obj = history[-1]

        for _ in range(p):
            j = np.random.randint(p)

            xj = X[:, j]
            r = y - X @ beta + xj * beta[j]
            c = xj @ r
            a = xj @ xj

            if c > lam:
                beta[j] = (c - lam) / a
            elif c < -lam:
                beta[j] = (c + lam) / a
            else:
                beta[j] = 0

        new_obj = objective(beta)
        history.append(new_obj)

        if old_obj - new_obj < tol:
            break

    return beta, history


Xc = X_train_cleaned
yc = y_train_cleaned

lam = 1.6612205949999663
B = 30
sample_size = 70
rng = np.random.default_rng(42)

results = []
paths = {}


def obj(X, y, beta):
    return 0.5 * np.sum((y - X @ beta) ** 2) + lam * np.sum(np.abs(beta))


for b in range(B):
    idx = rng.choice(len(yc), sample_size, replace=True)

    Xb = Xc[idx]
    yb = yc[idx]

    sk = Lasso(alpha=lam / sample_size, fit_intercept=False, tol=1e-12, max_iter=100000)

    start = time.perf_counter()
    sk.fit(Xb, yb)
    sk_time = time.perf_counter() - start

    beta_sk = sk.coef_
    sk_obj = obj(Xb, yb, beta_sk)

    results.append(
        [
            b,
            "sklearn",
            sk_time,
            sk.n_iter_,
            True,
            sk_obj,
            0,
            0,
            np.sum(np.abs(beta_sk) > 1e-8),
        ]
    )

    methods = [
        ("cyclic 1e-4", lasso_cd, 1e-4),
        ("cyclic 1e-8", lasso_cd, 1e-8),
        ("random 1e-4", lasso_random_cd, 1e-4),
        ("random 1e-8", lasso_random_cd, 1e-8),
    ]

    for name, solver, tol in methods:
        if "random" in name:
            np.random.seed(b)

        start = time.perf_counter()
        beta, history = solver(Xb, yb, lam, tol)
        runtime = time.perf_counter() - start

        final_obj = history[-1]
        converged = history[-2] - history[-1] < tol

        results.append(
            [
                b,
                name,
                runtime,
                len(history) - 1,
                converged,
                final_obj,
                final_obj - sk_obj,
                np.linalg.norm(beta - beta_sk),
                np.sum(np.abs(beta) > 1e-8),
            ]
        )

        if b == 0:
            paths[name] = history

    if b == 0:
        sklearn_rep_obj = sk_obj


results = pd.DataFrame(
    results,
    columns=[
        "bootstrap",
        "solver",
        "runtime",
        "iterations",
        "converged",
        "objective",
        "objective_gap",
        "coef_difference",
        "nonzero",
    ],
)

summary = results.groupby("solver").agg(
    convergence_rate=("converged", "mean"),
    runtime_mean=("runtime", "mean"),
    runtime_sd=("runtime", "std"),
    iterations_mean=("iterations", "mean"),
    objective_gap_mean=("objective_gap", "mean"),
    coef_difference_mean=("coef_difference", "mean"),
    nonzero_mean=("nonzero", "mean"),
)

print("Cleaned training shape:", Xc.shape)
print(summary.to_string())

plt.figure()

for name, history in paths.items():
    plt.plot(history, label=name)

plt.axhline(sklearn_rep_obj, linestyle="--", label="sklearn final")
plt.xlabel("Cycle")
plt.ylabel("Objective")
plt.yscale("log")
plt.legend()
plt.show()
