import neal
import numpy
import pandas as pd
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from pyqubo import Sum, Model, Mul
import math
from dimod import AdjVectorBQM
import time

from source import Excel, Create_Qubo, final_objective

returns, sigma = Excel()

N = 10
_lambda = 1
P = 100
A = 9

qubo, offset = Create_Qubo(N=N, _lambda=_lambda, P=P, A=A, returns=returns, sigma=sigma)

sampler = neal.SimulatedAnnealingSampler()
response = sampler.sample_qubo(qubo, num_sweeps=1000, num_reads=50)

la = response.first.sample

mn = 1e9

for sample in response.samples():
    objective =  final_objective(sample, _lambda, returns, sigma,N)   
    if objective < mn:
        mn = objective
        x = sample

print(mn)
print(x)
