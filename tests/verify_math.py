"""Independent numerical audit of the derivation; DOES NOT execute C++ code.

Uses quadrature of the matrix Gramian, rather than its analytic entry formula,
to check the scaled Gramian/Laurent-cost derivation used by pff.h.
"""
from math import factorial
from pathlib import Path
import json
import numpy as np


def check(dim, order, mode, duration):
    n = dim * order
    a = np.zeros((n, n))
    a[:-dim, dim:] = np.eye(n - dim)
    b = np.zeros((n, dim))
    b[-dim:, :] = np.eye(dim)
    r = np.eye(dim)
    r[0, 0] = 2
    r[0, 1] = r[1, 0] = .2
    inv_r = np.linalg.inv(r)

    def transition(t):
        return sum(np.linalg.matrix_power(a, k) * t**k / factorial(k) for k in range(order))

    def gramian(t):
        nodes, weights = np.polynomial.legendre.leggauss(24)
        g = np.zeros((n, n))
        for node, weight in zip(nodes, weights):
            e = transition((node + 1) * t / 2)
            g += weight * t / 2 * e @ b @ inv_r @ b.T @ e.T
        return g

    start = .1 * np.sin(np.arange(n) + .3)
    target = .4 * np.cos(np.arange(n) + .1)
    rows = list(range(dim))
    if mode == 1:
        rows = sorted(set(rows + [dim, n - 1]))
    elif mode == 2:
        rows = list(range(n))
    rows = np.array(rows)
    residual = (target - transition(duration) @ start)[rows]
    g = gramian(duration)[np.ix_(rows, rows)]
    expected_energy = residual @ np.linalg.solve(g, residual)
    g1 = gramian(1)[np.ix_(rows, rows)]
    coeff = np.zeros((len(rows), order))
    for i, index in enumerate(rows):
        derivative, axis = divmod(index, dim)
        coeff[i, order - derivative - 1] += target[index]
        for j in range(derivative, order):
            coeff[i, order - j - 1] -= start[j * dim + axis] / factorial(j - derivative)
    cross = coeff.T @ np.linalg.solve(g1, coeff)
    energy = sum(cross[k, l] / duration**(k + l + 1) for k in range(order) for l in range(order))
    error = abs(energy - expected_energy) / (1 + abs(expected_energy))
    assert error < 1e-7, (dim, order, mode, duration, error)
    scales = duration**(order - .5 - rows // dim)
    eta = np.zeros(n)
    eta[rows] = np.linalg.solve(g1, residual / scales) / scales
    endpoint = transition(duration) @ start + gramian(duration) @ eta
    endpoint_error = np.linalg.norm(endpoint[rows] - target[rows])
    assert endpoint_error < 1e-7, endpoint_error
    return error, endpoint_error


if __name__ == "__main__":
    errors = [check(dim, order, mode, t)
              for dim, order in [(2, 2), (2, 4), (4, 4)]
              for mode in range(3) for t in [.2, 1.7, 5]]
    report = {
        "scope": "Python derivation audit only; C++ compilation and tests NOT executed",
        "cases": len(errors),
        "max_relative_energy_error": max(e[0] for e in errors),
        "max_endpoint_error": max(e[1] for e in errors),
        "passed": True,
    }
    output = Path(__file__).resolve().parents[1] / "docs" / "math_check.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
