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

fix, axs = plt.subplots(3)


pyro.clear_param_store()

def model(prior_time):
    beta0 = np.exp(10)
    time = pyro.sample("time", dist.Gamma(prior_time, beta0))
    return pyro.sample("observation", dist.Gamma(time, beta0))

# time | prior, obs ~ Gamma(a,b)
def cond_model(observation, prior_time):
    return pyro.condition(model, data={"observation": observation})(prior_time)

def guide(observation, prior_time):
    a = pyro.param("a", torch.tensor(prior_time))
    b = pyro.param("b", torch.tensor(np.exp(10)), constraint=constraints.positive)
    return pyro.sample("time", dist.Gamma(a, b))


adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = pyro.optim.Adam(adam_params)
svi = SVI(cond_model, guide, optimizer, loss=Trace_ELBO())

losses, a, b = [], [], []

for mt in meanTime:
    losses.append(svi.step(np.exp(mt), np.exp(0.5)))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())
    # if step % 100 == 0:
    #     print('.', end='')

alpha_q = np.log(pyro.param("a").item())
beta_q = np.log(pyro.param("b").item())

print(alpha_q)
print(beta_q)

gen_data = []
for i in range(len(meanTime)):
    gd = pyro.sample("time", dist.Gamma(alpha_q, beta_q)).item()
    gen_data.append(gd)

axs[0].plot(range(len(count)), count)
axs[0].set_title("Number of calls")
axs[1].plot(range(len(meanTime)), meanTime)
axs[1].set_title("Mean time between events")
axs[2].plot(range(len(gen_data)), gen_data)
axs[2].set_title("Generated Mean time between events")
plt.show()


