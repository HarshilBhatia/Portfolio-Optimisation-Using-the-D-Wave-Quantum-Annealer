import neal
import numpy
import pandas as pd
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from pyqubo import Sum, Model, Mul
import math
from dimod import AdjVectorBQM


def Excel():
    r = pd.read_excel("ret.xlsx")
    s = pd.read_excel("corr.xlsx")

    returns = r["return"]
    sigma = s.loc[:, s.columns != "STOCK"]
    sigma = sigma.to_numpy()
    returns = returns
    sigma = (sigma * 1000).astype(int)
    returns = (returns * 100).astype(int)
    return returns, sigma


def Create_Qubo(N, _lambda, P, A, returns, sigma):

    x = Array.create("vector", 2 * N, "BINARY")
    term1 = 0
    H = 0
    for i in range(N):
        for j in range(N):
            H += (
                _lambda
                * (sigma[i][j])
                * (x[2 * i] + x[2 * i + 1] - 1)
                * (x[2 * j] + x[2 * j + 1] - 1)
            )
    term1 += Constraint(H, label="Sigma")

    term2 = 0
    H = 0
    for i in range(N):
        H -= (1 - _lambda) * returns[i] * (x[2 * i] + x[2 * i + 1] - 1)
    term2 = Constraint(H, label="Returns")

    H = 0
    for i in range(N):
        H += x[2 * i] + x[2 * i + 1] - 1
    H -= A
    H = H ** 2
    H *= P
    select_n = Constraint(H, label="select_n_projects")

    model = H.compile()
    return model.to_qubo()


def final_objective(sample, _lambda, returns, sigma,N):
    final_sigx = 0

    for i in range(N):
        for j in range(N):
            final_sigx += _lambda * (
                (
                    sample["vector[{}]".format(2 * i)]
                    + sample["vector[{}]".format(2 * i + 1)]
                    - 1
                )
                * (
                    sample["vector[{}]".format(2 * j)]
                    + sample["vector[{}]".format(2 * j + 1)]
                    - 1
                )
                * sigma[i][j]
            )

    for i in range(N):
        final_sigx -= (
            (1 - _lambda)
            * (returns[i])
            * (
                sample["vector[{}]".format(2 * i)]
                + sample["vector[{}]".format(2 * i + 1)]
                - 1
            )
        )
    return final_sigx
