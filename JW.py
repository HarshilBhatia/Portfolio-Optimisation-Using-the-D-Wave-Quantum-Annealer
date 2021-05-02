import neal
import numpy
import pandas as pd
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from pyqubo import Sum, Model, Mul
import math
from dimod import AdjVectorBQM
import time

from source import Excel, Create_Qubo

returns, sigma = Excel()

N = 10
# n = 2
_lambda = 1
P = 100
A = 9

qubo, offset = Create_Qubo(N=N, _lambda=_lambda, P=P, A=A, returns=returns, sigma=sigma)

sampler = neal.SimulatedAnnealingSampler()
response = sampler.sample_qubo(qubo, num_sweeps=1000, num_reads=50)
la = response.first.sample

mn = 1e9

for sample in response.samples():
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
    # print(final_sigx)

    if final_sigx < mn:
        mn = final_sigx
        x = sample

print(mn)
print(x)
