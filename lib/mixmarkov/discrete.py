# Copyright 2022 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discrete-time multistate models.

Implementations are based on a dense representation of the graph of admissible
transitions.
"""

import abc
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from jax import jacfwd, jacrev, jit, grad
from scipy.optimize import minimize

from .utils import random_gd

# Enable double-precision floating-point numbers.
jax.config.update("jax_enable_x64", True)


@partial(jit, static_argnums=(2,))
def _dtmc_distribution(p0, trans, horizon):
    """Compute DTMC distribution up to `horizon`."""
    def step(val, _):
        dist, trans = val
        dist = jnp.sum(dist[...,jnp.newaxis] * trans, axis=-2)
        return ((dist, trans), dist)
    _, res = jax.lax.scan(step, init=(p0, trans), xs=None, length=horizon)
    return jnp.moveaxis(res, 0, -2)


class DiscreteTimeModel(metaclass=abc.ABCMeta):

    def __init__(self, mask):
        """Initialize model.

        Parameters
        ----------
        mask : ndarray, shape (N, N)
            Binary mask indicating admissible transitions.
        """
        self.mask = mask
        self.params = None

    def fit(
            self,
            ks,
            xs=None,
            ws=None,
            method="Newton-CG",
            l2=0.0,
            verbose=True,
            tol=1e-5,
        ):
        """Fit model by maximum-likelihood estimation.

        Parameters
        ----------
        ks : ndarray, shape (M, N, N)
            Number of times each edge was visited.
        xs : ndarray, shape (M, D), optional
            Feature matrix, number of trajectories `M` by number of features
            `D`. Default (None) results in matrix `ones((M, 1))`.
        ws : ndarray, shape (M,), optional
            Weight assigned to each observation. Default (None) results in
            vector `ones(M)`.
        """
        if ws is None:
            ws = np.ones(ks.shape[0])
        if xs is None:
            xs = np.ones((ks.shape[0], 1))
        m, d = xs.shape
        n = ks.shape[-1]
        params_shape = (d, *self.params_trailing_shape)
        params = np.zeros(params_shape)
        # `minimize` expects 1D parameter vectors, hence the `reshape`, etc.
        def cost(params):
            return -self.loglike(
                params.reshape(params_shape), xs, ks, ws, self.mask, l2=l2
            )
        def grad(params):
            return -self.grad_loglike(
                params.reshape(params_shape), xs, ks, ws, self.mask, l2=l2
            ).ravel()
        def hess(params):
            return -self.hess_loglike(
                params.reshape(params_shape), xs, ks, ws, self.mask, l2=l2
            ).reshape(np.product(params_shape), -1)
        res = minimize(
            fun=cost,
            x0=params.ravel(),
            method=method,
            jac=grad,
            hess=hess,
            tol=tol,
            options={"disp": verbose},
        )
        self.res = res
        self.params = res.x.reshape(params_shape)

    @property
    @abc.abstractmethod
    def params_trailing_shape(self):
        """Return trailing shape of parameter tensor."""

    @abc.abstractmethod
    def predict(self, p0, horizon, xs=None):
        """Compute marginal state distribution up to a horizon.

        Parameters
        ----------
        p0 : ndarray, shape (M, N)
            Initial distribution (must sum to 1 along last dimension).
        horizon : int
            Maximum predictive horizon.
        xs : ndarray, shape (M, D), optional
            Feature matrix. Default (None) results in matrix `ones((M, 1))`.

        Returns
        -------
        res : ndarray, shape (M, horizon, N)
            Marginal state distribution after 1, ..., `horizon` steps.
        """

    def predictive_loglike(self, ks, xs=None, offset=None):
        """Compute the predictive log-likelihood of a trained model.

        Parameters
        ----------
        ks : ndarray, shape (M, N, N)
            Observed transitions (number of times each edge was visited).
        xs : ndarray, shape (M, D), optional
            Feature matrix. Default (None) results in matrix `ones((M, 1))`.
        """
        if self.params is None:
            raise ValueError("model is not fitted")
        if xs is None:
            xs = np.ones((ks.shape[0], 1))
        ws = np.ones(ks.shape[0])
        if offset is not None:
            return self.loglike(
                self.params, xs, ks, ws, self.mask, l2=0.0, offset=offset,
            )
        else:
            return self.loglike(self.params, xs, ks, ws, self.mask, l2=0.0)

    @staticmethod
    @abc.abstractmethod
    def loglike(params, xs, ks, ws, mask, l2=0.0):
        """Compute the log-likelihood of the model parameters given data.

        Parameters
        ----------
        params : ndarray, shape (D, ...)
            Parameter tensor.
        xs : ndarray, shape (..., D)
            Feature matrix.
        ks : ndarray, shape (M, N, N)
            Observed transitions (number of times each edge was visited).
        ws : ndarray, shape (M,)
            Weight assigned to each transition matrix.
        mask : ndarray, shape (N, N)
            Binary mask indicating admissible transitions.
        """

    @staticmethod
    @abc.abstractmethod
    def grad_loglike(params, xs, ks, ws, mask, l2=0.0):
        """Gradient of the log-likelihood."""

    @staticmethod
    @abc.abstractmethod
    def hess_loglike(params, xs, ks, ws, mask, l2=0.0):
        """Hessian of the log-likelihood."""


class DTMC(DiscreteTimeModel):

    @property
    def params_trailing_shape(self):
        n = self.mask.shape[0]
        return (n, n)

    def predict(self, p0, horizon, xs=None):
        if self.params is None:
            raise ValueError("model is not fitted")
        if xs is None:
            xs = np.ones((p0.shape[0], 1))
        logits = np.tensordot(xs, self.params, axes=(-1, 0))
        trans = jax.nn.softmax(np.where(self.mask, logits, -np.inf), axis=-1)
        p0 = jnp.broadcast_to(p0, trans.shape[:-1])
        return _dtmc_distribution(p0, trans, horizon)

    @staticmethod
    @jit
    def loglike(params, xs, ks, ws, mask, l2=0.0):
        logits = jnp.tensordot(xs, params, axes=(-1, 0))
        # Disable transitions that are masked out.
        logits = jnp.where(mask, logits, -np.inf)
        normalized = jax.nn.log_softmax(logits, axis=-1)
        ws = jnp.expand_dims(ws, axis=(-2,-1))
        return (
            jnp.nansum(ws * ks * normalized)
            - l2 * jnp.sum(jnp.square(params))
        )

    grad_loglike = staticmethod(jit(grad(loglike.__func__)))
    hess_loglike = staticmethod(jit(jacfwd(jacrev(loglike.__func__))))


class DirMixDTMC(DiscreteTimeModel):

    @property
    def params_trailing_shape(self):
        n = self.mask.shape[0]
        return (n, n)

    def predict(
            self, p0, horizon, xs=None, n_samples=100, offset=None, seed=0):
        if self.params is None:
            raise ValueError("model is not fitted")
        if xs is None:
            xs = np.ones((p0.shape[0], 1))
        alpha = jnp.exp(jnp.tensordot(xs, self.params, axes=(-1, 0)))
        alpha = jnp.where(self.mask, alpha, 0.0)
        if offset is not None:
            alpha += offset
        p0 = jnp.broadcast_to(p0, alpha.shape[:-1])
        return self._predict(alpha, p0, horizon, n_samples, seed)

    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _predict(alpha, p0, horizon, n_samples, seed):
        key = jax.random.PRNGKey(seed)
        res = jnp.zeros((*alpha.shape[:-2], horizon, p0.shape[-1]))
        def process_sample(_, tup):
            key, res = tup
            key, subkey = jax.random.split(key)
            trans = jax.random.dirichlet(subkey, alpha)
            return (key, res + _dtmc_distribution(p0, trans, horizon))
        _, res = jax.lax.fori_loop(0, n_samples, process_sample, (key, res))
        return res / n_samples

    @staticmethod
    @jit
    def loglike(params, xs, ks, ws, mask, l2=0.0):
        alpha = jnp.exp(jnp.tensordot(xs, params, axes=(-1, 0)))
        alpha = jnp.where(mask, alpha, 0.0)
        ws = jnp.expand_dims(ws, axis=(-2,-1))
        return (
            # Disable transitions that are masked out.
            jnp.sum(ws * jnp.where(
                mask, jax.lax.lgamma(alpha + ks) - jax.lax.lgamma(alpha), 0.0
            ))
            - jnp.nansum(ws[...,0] * (
                jax.lax.lgamma(jnp.sum(alpha + ks, axis=-1))
                - jax.lax.lgamma(jnp.sum(alpha, axis=-1))
            ))
            - l2 * jnp.sum(jnp.square(params))
        )

    grad_loglike = staticmethod(jit(grad(loglike.__func__)))
    hess_loglike = staticmethod(jit(jacfwd(jacrev(loglike.__func__))))


class GDirMixDTMC(DiscreteTimeModel):

    @property
    def params_trailing_shape(self):
        n = self.mask.shape[0]
        return (n, n-1, 2)

    def predict(
            self, p0, horizon, xs=None, n_samples=100, offset=None, seed=0):
        if self.params is None:
            raise ValueError("model is not fitted")
        if xs is None:
            xs = np.ones((p0.shape[0], 1))
        params = jnp.exp(jnp.tensordot(xs, self.params, axes=(-1, 0)))
        alpha = params[...,0]
        beta = params[...,1]
        # Adjust mask for parametrization of generalized Dirichlet.
        # `tail[...,i]` is true iff there is an admissible edge `>= i`.
        tail = jnp.flip(
            jnp.cumsum(jnp.flip(self.mask, axis=1), axis=1).astype(bool),
            axis=1,
        )
        mask = self.mask[:,:-1] & tail[:,1:]
        # Indices of last admissible transition.
        last = (len(self.mask) - 1) - np.argmax(self.mask[:,::-1], axis=1)
        if offset is not None:
            # Compute tail aggregates: `agg[...,i] = sum(ks[...,i:], axis=-1)`.
            agg = jnp.flip(
                jnp.cumsum(jnp.flip(offset, axis=-1), axis=-1), axis=-1
            )
            alpha += offset[...,:-1]
            beta += agg[...,1:]
        p0 = jnp.broadcast_to(p0, alpha.shape[:-1])
        return self._predict(alpha, beta, mask, last, p0, horizon, n_samples, seed)

    @staticmethod
    @partial(jit, static_argnums=(5,))
    def _predict(alpha, beta, mask, last, p0, horizon, n_samples, seed):
        key = jax.random.PRNGKey(seed)
        res = jnp.zeros((*alpha.shape[:-2], horizon, p0.shape[-1]))
        def process_sample(_, tup):
            key, res = tup
            key, subkey = jax.random.split(key)
            trans = random_gd(subkey, alpha, beta, mask, last)
            return (key, res + _dtmc_distribution(p0, trans, horizon))
        _, res = jax.lax.fori_loop(0, n_samples, process_sample, (key, res))
        return res / n_samples

    @staticmethod
    @jit
    def loglike(params, xs, ks, ws, mask, l2=0.0, offset=None):
        zs = jnp.exp(jnp.tensordot(xs, params, axes=(-1, 0)))
        alpha = zs[...,0]
        beta = zs[...,1]
        if offset is not None:
            # Compute tail aggregates: `agg[...,i] = sum(ks[...,i:], axis=-1)`.
            agg = jnp.flip(
                jnp.cumsum(jnp.flip(offset, axis=-1), axis=-1), axis=-1
            )
            alpha += offset[...,:-1]
            beta += agg[...,1:]
        # Adjust mask for parametrization of generalized Dirichlet.
        # `tail[...,i]` is true iff there is an admissible edge `>= i`.
        tail = jnp.flip(
            jnp.cumsum(jnp.flip(mask, axis=1), axis=1).astype(bool), axis=1
        )
        mask = mask[:,:-1] & tail[:,1:]
        # Compute tail aggregates: `agg[...,i] = sum(ks[...,i:], axis=-1)`.
        agg = jnp.flip(jnp.cumsum(jnp.flip(ks, axis=-1), axis=-1), axis=-1)
        vals = (
            jax.lax.lgamma(alpha + ks[...,:-1])
            + jax.lax.lgamma(beta + agg[...,1:])
            + jax.lax.lgamma(alpha + beta)
            - jax.lax.lgamma(alpha)
            - jax.lax.lgamma(beta)
            - jax.lax.lgamma(alpha + beta + agg[...,:-1])
        )
        ws = jnp.expand_dims(ws, axis=(-2,-1))
        return (
            # Disable transitions that are masked out.
            jnp.sum(ws * jnp.where(mask, vals, 0.0))
            - l2 * jnp.sum(jnp.square(params))
        )

    grad_loglike = staticmethod(jit(grad(loglike.__func__)))
    hess_loglike = staticmethod(jit(jacfwd(jacrev(loglike.__func__))))


class MDirMixDTMC(DiscreteTimeModel):

    """Model of MacKay & Bauman Peto (1995)."""

    @property
    def params_trailing_shape(self):
        n = self.mask.shape[0]
        return (n,)

    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    @jit
    def loglike(params, xs, ks, ws, mask, l2=0.0):
        n = mask.shape[0]
        zs = jnp.exp(jnp.tensordot(xs, params, axes=(-1, 0)))
        alpha = jnp.where(mask, jnp.tile(zs[...,jnp.newaxis,:], (n, 1)), 0.0)
        ws = jnp.expand_dims(ws, axis=(-2,-1))
        return (
            # Disable transitions that are masked out.
            jnp.sum(ws * jnp.where(
                mask, jax.lax.lgamma(alpha + ks) - jax.lax.lgamma(alpha), 0.0
            ))
            - jnp.nansum(ws[...,0] * (
                jax.lax.lgamma(jnp.sum(alpha + ks, axis=-1))
                - jax.lax.lgamma(jnp.sum(alpha, axis=-1))
            ))
            - l2 * jnp.sum(jnp.square(params))
        )

    grad_loglike = staticmethod(jit(grad(loglike.__func__)))
    hess_loglike = staticmethod(jit(jacfwd(jacrev(loglike.__func__))))
