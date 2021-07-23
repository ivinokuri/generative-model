#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pandas
import os
cwd = os.getcwd()
print(cwd)

# pyro.set_rng_seed(101)
DATA = pandas.read_csv('../../data.csv')

count = DATA['Counter(/move_base/TebLocalPlannerROS/local_plan)']
meanTime = DATA['Mean_Consecutive(/move_base/TebLocalPlannerROS/local_plan)']

fix, axs = plt.subplots(2)

axs[0].plot(range(len(count)), count)
axs[0].set_title("Number of calls")
axs[1].plot(range(len(meanTime)), meanTime)
axs[1].set_title("Mean time between events")
plt.show()

pyro.clear_param_store()

def model(data):
    alpha0 = torch.tensor(2.0)
    beta0 = torch.tensor(5.0)

    for i in range(len(data)):
        pyro.sample("obs_{}".format(i), dist.Gamma(alpha0, beta0), obs=data[i])


def guide(data):
    alpha_q = pyro.param("alpha_q", torch.tensor(3.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(3.0),
                        constraint=constraints.positive)
    pyro.sample("latent_fairness", dist.Gamma(alpha_q, beta_q))


adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = pyro.optim.Adam(adam_params)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

for step in range(1000):
    svi.step(meanTime)
    if step % 100 == 0:
        print('.', end='')

alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

print(alpha_q)
print(beta_q)