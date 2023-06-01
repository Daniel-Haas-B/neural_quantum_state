import os

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import matplotlib.pyplot as plt
import netket as nk
import jax.numpy as jnp

import flax
import flax.linen as nn

import plot_utils
import seaborn as sns

# set pallette to pastel

sns.set_palette('pastel')


# ### Define the Hamiltonian

N = 2


def v(x):

    return 0.5 * jnp.linalg.norm(x) ** 2


def minimum_distance(x, sdim):   # sdim is the spatial dimension
    """Computes distances between particles using mimimum image convention"""
    n_particles = x.shape[0] // sdim
    x = x.reshape(-1, sdim)

    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
        jnp.triu_indices(n_particles, 1)
    ]

    return jnp.linalg.norm(distances, axis=1)


def interaction(x, sdim):
    """Computes the potential energy of the system"""
    dis = minimum_distance(x, sdim)

    return jnp.sum(1 / dis)


# # Define the hilbert space

hilb = nk.hilbert.Particle(
    N=N, L=(jnp.inf, jnp.inf), pbc=False
)   # L = dimension, pcb is


ekin = nk.operator.KineticEnergy(hilb, mass=1.0)

pot = nk.operator.PotentialEnergy(hilb, v)

interact = nk.operator.PotentialEnergy(hilb, lambda x: interaction(x, 2))

ha_0 = ekin + pot   # + interact

ha_1 = ha_0 + interact


class FFN(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module. This is the ratio of neurons to input dofs
    alpha: int = 1

    @nn.compact
    def __call__(self, x):

        # here we construct the first dense layer using a
        # pre-built implementation in flax.
        # features is the number of output nodes
        dense = nn.Dense(features=self.alpha * x.shape[-1], param_dtype=float)

        # we apply the dense layer to the input
        y = dense(x)

        # the non-linearity is a simple ReLu
        y = nn.relu(y)

        # sum the output
        return jnp.sum(y, axis=-1)


# ### Define the sampler

sampler_gaussian = nk.sampler.MetropolisAdjustedLangevin(
    hilb, n_chains=16, n_sweeps=32
)   # sigma is the standard deviation of the gaussian that is used to sample the configuration space

sampler_rbm = nk.sampler.MetropolisGaussian(
    hilb, sigma=1e-4, n_chains=16, n_sweeps=16
)

sampler_FFN = nk.sampler.MetropolisGaussian(
    hilb, sigma=1e-3, n_chains=16, n_sweeps=32
)


# ### Models

model_gaussian = nk.models.Gaussian()
vs_gaussian = nk.vqs.MCState(
    sampler_gaussian,
    model_gaussian,
    n_samples=10**4,
    n_discard_per_chain=2000,
)

model_rbm = nk.models.RBM(alpha=2)
vs_rbm = nk.vqs.MCState(
    sampler_rbm, model_rbm, n_samples=10**4, n_discard_per_chain=2000
)

model_ffn = FFN(alpha=2)
vs_ffn = nk.vqs.MCState(
    sampler_FFN, model_ffn, n_samples=10**5, n_discard_per_chain=2000
)

# optimizer
op_gaussian = nk.optimizer.Sgd(learning_rate=0.1)
op_rbm = nk.optimizer.Sgd(learning_rate=1e-3)
op_ffn = nk.optimizer.Sgd(learning_rate=1e-5)

# Preconditioner
sr_gaussian = nk.optimizer.SR(diag_shift=0.001)
sr_rbm = nk.optimizer.SR(diag_shift=0.4)
sr_ffn = nk.optimizer.SR(diag_shift=0.1)

import pandas as pd
import numpy as np

gs_gaussian_0 = nk.VMC(
    ha_0,
    op_gaussian,
    sampler_gaussian,
    variational_state=vs_gaussian,
    preconditioner=sr_gaussian,
)   # gs stands for ground state
gs_gaussian_1 = nk.VMC(
    ha_1,
    op_gaussian,
    sampler_gaussian,
    variational_state=vs_gaussian,
    preconditioner=sr_gaussian,
)


iters = 200

log = nk.logging.RuntimeLog()
gs_gaussian_0.run(n_iter=iters, out=log)
data_gaussian_0 = log.data
# #
log = nk.logging.RuntimeLog()
gs_gaussian_1.run(n_iter=iters, out=log)
data_gaussian_1 = log.data

# eliminate 0th iteration

iters = np.arange(0, iters, 1)
df_gaussian_0 = pd.DataFrame(
    data={
        'iters': iters,
        'Energy': data_gaussian_0['Energy'].Mean,
        'Sigma': data_gaussian_0['Energy'].Sigma,
    }
)
df_gaussian_1 = pd.DataFrame(
    data={
        'iters': iters,
        'Energy': data_gaussian_1['Energy'].Mean,
        'Sigma': data_gaussian_1['Energy'].Sigma,
    }
)


sns.lineplot(
    data=df_gaussian_0, x='iters', y='Energy', label='Gaussian Non-interacting'
)
sns.lineplot(
    data=df_gaussian_1, x='iters', y='Energy', label='Gaussian Interacting'
)

plt.fill_between(
    iters,
    df_gaussian_0['Energy'] - df_gaussian_0['Sigma'],
    df_gaussian_0['Energy'] + df_gaussian_0['Sigma'],
    alpha=0.5,
)
plt.fill_between(
    iters,
    df_gaussian_1['Energy'] - df_gaussian_1['Sigma'],
    df_gaussian_1['Energy'] + df_gaussian_1['Sigma'],
    alpha=0.5,
)

plt.hlines(
    2,
    xmin=0,
    xmax=iters[-1] + 1,
    color='black',
    label='Exact Non-interacting',
    linestyle='dashed',
)
plt.hlines(
    3,
    xmin=0,
    xmax=iters[-1] + 1,
    color='Purple',
    label='Exact Interacting',
    linestyle='dashed',
)

# put legend in the top middle
plt.legend(
    loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2, fontsize=11.4
)


# make background white
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# add grid
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# set ylimit
plt.ylim(1.9, 4.2)

plt.xlabel('Epochs')

plt.ylabel(r'$\langle E_L \rangle$')

plt.savefig(f'./figs/netket_energy_gaussian.pdf', bbox_inches='tight')
