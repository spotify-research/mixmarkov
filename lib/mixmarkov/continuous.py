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

"""Continuous-time multistate models.

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

from .utils import parallel_expm

# Enable double-precision floating-point numbers.
jax.config.update("jax_enable_x64", True)


class ContinuousTimeModel(metaclass=abc.ABCMeta):

    def __init__(self, mask):
        """Initialize model.

        Parameters
        ----------
        mask : ndarray, shape (N, N)
            Binary mask indicating admissible transitions. Diagonal elements
            (self-transitions) are ignored.
        """
        if np.diag(mask).any():
            raise ValueError("self-transitions must be `False` in `mask`")
        self.mask = mask
        self.params = None

    def fit(
        self,
        ks,
        ts,
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
        ts : ndarray, shape (M, N)
            Total time spent in each state.
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
                params.reshape(params_shape), xs, ks, ts, ws, self.mask, l2=l2
            )
        def grad(params):
            return -self.grad_loglike(
                params.reshape(params_shape), xs, ks, ts, ws, self.mask, l2=l2
            ).ravel()
        def hess(params):
            return -self.hess_loglike(
                params.reshape(params_shape), xs, ks, ts, ws, self.mask, l2=l2
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
    def predict(self, p0, t, xs=None):
        """Compute marginal state distribution at time `t`.

        Parameters
        ----------
        p0 : ndarray, shape (M, N)
            Initial distribution (must sum to 1 along last dimension).
        t : float
            Maximum predictive horizon.
        xs : ndarray, shape (M, D), optional
            Feature matrix. Default (None) results in matrix `ones((M, 1))`.

        Returns
        -------
        res : ndarray, shape (M, N)
            Marginal state distribution at time `t`.
        """

    def predictive_loglike(self, ks, ts, xs=None, offset=None):
        """Compute the predictive log-likelihood of a trained model.

        Parameters
        ----------
        xs : ndarray, shape (M, D)
            Feature matrix.
        ks : ndarray, shape (M, N, N)
            Observed transitions (number of times each edge was visited).
        ts : ndarray, shape (M, N)
            Total time spent in each state.
        """
        if self.params is None:
            raise ValueError("model is not fitted")
        if xs is None:
            xs = np.ones((ks.shape[0], 1))
        ws = np.ones(ks.shape[0])
        if offset is not None:
            return self.loglike(
                self.params, xs, ks, ts, ws, self.mask, l2=0.0, offset=offset,
            )
        else:
            return self.loglike(self.params, xs, ks, ts, ws, self.mask, l2=0.0)

    @staticmethod
    @abc.abstractmethod
    def loglike(params, xs, ks, ts, ws, mask, l2=0.0):
        """Compute the log-likelihood of the model parameters given data.

        Parameters
        ----------
        params : ndarray, shape (D, ...)
            Parameter tensor.
        xs : ndarray, shape (M, D)
            Feature matrix.
        ks : ndarray, shape (M, N, N)
            Observed transitions (number of times each edge was visited).
        ts : ndarray, shape (M, N)
            Total time spent in each state.
        ws : ndarray, shape (M,)
            Weight assigned to each transition matrix.
        mask : ndarray, shape (N, N)
            Binary mask indicating admissible transitions.
        """

    @staticmethod
    @abc.abstractmethod
    def grad_loglike(params, xs, ks, ts, ws, mask, l2=0.0):
        """Gradient of the log-likelihood."""

    @staticmethod
    @abc.abstractmethod
    def hess_loglike(params, xs, ks, ts, ws, mask, l2=0.0):
        """Hessian of the log-likelihood."""


class CTMC(ContinuousTimeModel):

    @property
    def params_trailing_shape(self):
        n = self.mask.shape[0]
        return (n, n)

    def predict(self, p0, t, xs=None):
        if xs is None:
            xs = np.ones((p0.shape[0], 1))
        rates = jnp.where(
            self.mask,
            jnp.exp(jnp.tensordot(xs, self.params, axes=(-1, 0))),
            0.0,
        )
        gen = jnp.where(
            ~jnp.eye(rates.shape[-1], dtype=bool),
            rates,
            -rates.sum(axis=-1, keepdims=True)
        )
        return jnp.sum(p0[...,jnp.newaxis] * parallel_expm(t * gen), axis=-2)

    @staticmethod
    @jit
    def loglike(params, xs, ks, ts, ws, mask, l2=0.0):
        logits = jnp.tensordot(xs, params, axes=(-1, 0))
        ws = jnp.expand_dims(ws, axis=(-2,-1))
        return jnp.sum(ws * jnp.where(
            mask, ks * logits - jnp.exp(logits) * ts[...,jnp.newaxis], 0.0
        )) - l2 * jnp.sum(jnp.square(params))

    grad_loglike = staticmethod(jit(grad(loglike.__func__)))
    hess_loglike = staticmethod(jit(jacfwd(jacrev(loglike.__func__))))


class GamMixCTMC(ContinuousTimeModel):

    @property
    def params_trailing_shape(self):
        n = self.mask.shape[0]
        return (n, n, 2)

    def predict(self, p0, t, xs=None, n_samples=100, offset=None, seed=0):
        if xs is None:
            xs = np.ones((p0.shape[0], 1))
        zs = jnp.exp(jnp.tensordot(xs, self.params, axes=(-1, 0)))
        alpha = zs[...,0]
        beta = zs[...,1]
        if offset is not None:
            offset_ks, offset_ts = offset
            alpha += offset_ks
            beta += offset_ts[...,jnp.newaxis]
        p0 = jnp.broadcast_to(p0, alpha.shape[:-1])
        return self._predict(alpha, beta, self.mask, p0, t, n_samples, seed)

    @staticmethod
    @jit
    def _predict(alpha, beta, mask, p0, t, n_samples, seed):
        key = jax.random.PRNGKey(seed)
        res = jnp.zeros(alpha.shape[:-1])
        def process_sample(_, tup):
            key, res = tup
            key, subkey = jax.random.split(key)
            rates = jnp.where(
                mask,
                jax.random.gamma(subkey, alpha) / beta,
                0.0,
            )
            gen = jnp.where(
                ~jnp.eye(rates.shape[-1], dtype=bool),
                rates,
                -rates.sum(axis=-1, keepdims=True)
            )
            dist = jnp.sum(
                p0[...,jnp.newaxis] * parallel_expm(t * gen), axis=-2
            )
            return (key, res + dist)
        _, res = jax.lax.fori_loop(0, n_samples, process_sample, (key, res))
        return res / n_samples

    @staticmethod
    @jit
    def loglike(params, xs, ks, ts, ws, mask, l2=0.0, offset=None):
        zs = jnp.tensordot(xs, params, axes=(-1, 0))
        log_alpha = zs[...,0]
        log_beta = zs[...,1]
        alpha = jnp.exp(log_alpha)
        if offset is not None:
            offset_ks, offset_ts = offset
            alpha += offset_ks
            log_alpha = jnp.logaddexp(log_alpha, jnp.log(offset_ks))
            log_beta = jnp.logaddexp(log_beta, jnp.log(offset_ts[...,jnp.newaxis]))
        vals = (
            jax.lax.lgamma(alpha + ks)
            - jax.lax.lgamma(alpha)
            + alpha * log_beta
            - (alpha + ks) * jnp.logaddexp(log_beta, jnp.log(ts[...,jnp.newaxis]))
        )
        ws = jnp.expand_dims(ws, axis=(-2,-1))
        # Disable transitions that are masked out.
        return (
            jnp.sum(ws * jnp.where(mask, vals, 0.0))
            - l2 * jnp.sum(jnp.square(params))
        )

    grad_loglike = staticmethod(jit(grad(loglike.__func__)))
    hess_loglike = staticmethod(jit(jacfwd(jacrev(loglike.__func__))))


class SGamMixCTMC(ContinuousTimeModel):

    """Simplified Gamma mixture model with beta shared across transitions."""

    @property
    def params_trailing_shape(self):
        n = self.mask.shape[0]
        return (n, n)

    def predict(self, p0, t, xs=None, n_samples=100, offset=None, seed=0):
        raise NotImplementedError()

    @staticmethod
    @jit
    def loglike(params, xs, ks, ts, ws, mask, l2=0.0, offset=None):
        alpha = jnp.exp(jnp.tensordot(xs, params, axes=(-1, 0)))
        beta = jnp.diagonal(alpha, axis1=-2, axis2=-1)[...,jnp.newaxis]
        if offset is not None:
            offset_ks, offset_ts = offset
            alpha += offset_ks
            beta += offset_ts[...,jnp.newaxis]
        vals = (
            jax.lax.lgamma(alpha + ks)
            - jax.lax.lgamma(alpha)
            + alpha * jnp.log(beta)
            - (alpha + ks) * jnp.log(beta + ts[...,jnp.newaxis])
        )
        ws = jnp.expand_dims(ws, axis=(-2,-1))
        # Disable transitions that are masked out.
        return (
            jnp.sum(ws * jnp.where(mask, vals, 0.0))
            - l2 * jnp.sum(jnp.square(params))
        )

    grad_loglike = staticmethod(jit(grad(loglike.__func__)))
    hess_loglike = staticmethod(jit(jacfwd(jacrev(loglike.__func__))))


class MGamMixCTMC(ContinuousTimeModel):

    """Continuous-time variant of MacKay & Bauman Peto (1995)."""

    @property
    def params_trailing_shape(self):
        n = self.mask.shape[0]
        return (n + 1,)

    def predict(self, p0, t, xs=None, n_samples=100, offset=None, seed=0):
        raise NotImplementedError()

    @staticmethod
    @jit
    def loglike(params, xs, ks, ts, ws, mask, l2=0.0, offset=None):
        zs = jnp.exp(jnp.tensordot(xs, params, axes=(-1, 0)))
        n = mask.shape[0]
        alpha = jnp.tile(zs[...,jnp.newaxis,:-1], (n, 1))
        beta = jnp.tile(zs[...,jnp.newaxis,-1:], (n, n))
        if offset is not None:
            offset_ks, offset_ts = offset
            alpha += offset_ks
            beta += offset_ts[...,jnp.newaxis]
        vals = (
            jax.lax.lgamma(alpha + ks)
            - jax.lax.lgamma(alpha)
            + alpha * jnp.log(beta)
            - (alpha + ks) * jnp.log(beta + ts[...,jnp.newaxis])
        )
        ws = jnp.expand_dims(ws, axis=(-2,-1))
        # Disable transitions that are masked out.
        return (
            jnp.sum(ws * jnp.where(mask, vals, 0.0))
            - l2 * jnp.sum(jnp.square(params))
        )

    grad_loglike = staticmethod(jit(grad(loglike.__func__)))
    hess_loglike = staticmethod(jit(jacfwd(jacrev(loglike.__func__))))
