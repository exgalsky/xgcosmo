import jax.numpy as jnp
import matplotlib.pyplot as plt
import xgutil.backend      as bk
import logging

import xgcosmo.cosmology   as cosmo

backend = bk.Backend(force_no_gpu=True,force_no_mpi=True)
cosmo_wsp = cosmo.cosmology(backend, h=0.67, n_s=0.96, cosmo_backend='CAMB') # for background expansion consistent with websky
print(cosmo_wsp.camb_params)

k = jnp.logspace(-3,1,1000)
pk = cosmo_wsp.matter_power(k) # power spectrum

plt.loglog(k, pk, '-')
plt.savefig('pk_vs_k_example.png')