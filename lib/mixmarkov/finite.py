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

import abc
import jax
import jax.numpy as jnp
import numpy as np

from jax import jit, vmap
from scipy.special import logsumexp
from .discrete import DTMC
from .continuous import CTMC


class FiniteMixDTMC:

    """Discrete-time finite mixture of Markov chains."""

    def __init__(self, mask, n_comps):
        self.components = [DTMC(mask) for _ in range(n_comps)]
        self.weights = None

    def fit(self, ks, max_iters=50, xs=None, l2=0.0, verbose=True, seed=0):
        """Train a finite mixture model using the EM algorithm."""
        m = ks.shape[0]
        n_comps = len(self.components)
        key = jax.random.PRNGKey(seed)
        fast = False
        if xs is None:
            false = True
            xs = np.ones((m, 1))
        # Initial E-step (randomized soft clustering).
        ws = jax.random.dirichlet(key, jnp.ones(n_comps), shape=(m,))
        old_val = np.inf
        for i in range(max_iters):
            # M-step.
            val = 0.0
            for j, mc in enumerate(self.components):
                if fast:
                    ks2 = np.sum(ws[:, j, None, None] * ks, axis=0)
                    mc.fit(ks2, verbose=False, l2=l2)
                else:
                    mc.fit(ks, xs=xs, ws=ws[:, j], verbose=False, l2=l2)
                val += mc.res.fun
            self.weights = jnp.sum(ws, axis=0) / jnp.sum(ws)
            imp = 100 * (old_val / val - 1)  # % improvement.
            if verbose:
                print("cost: {:.3f}, imp: {:.2f}%".format(val, imp))
            if i > 4 and imp < 0.1:
                return
            old_val = val
            # E-step.
            ws = self._compute_weights(ks, xs)

    def _compute_weights(self, ks, xs):
        """Compute soft cluster assignments."""
        ws = np.zeros((ks.shape[0], len(self.components)))
        for i, mc in enumerate(self.components):
            ws[:, i] = np.log(self.weights[i]) + self.vec_loglike(
                mc.params, xs, ks, 1.0, mc.mask, 0.0
            )
        return jax.nn.softmax(ws, axis=-1)

    def predict(self, p0, horizon, xs=None, offset=None):
        if xs is not None:
            m = xs.shape[0]
        else:
            m = p0.shape[0]
            xs = np.ones((m, 1))
        n = p0.shape[-1]
        if offset is not None:
            ws = self._compute_weights(offset, xs)
        else:
            ws = np.tile(self.weights, reps=(m, 1))
        res = np.zeros((m, horizon, n))
        for i, mc in enumerate(self.components):
            res += np.expand_dims(ws[:, i], axis=(-2, -1)) * mc.predict(
                p0, horizon, xs=xs
            )
        return res

    def predictive_loglike(self, ks, xs=None, offset=None):
        m = ks.shape[0]
        n_comps = len(self.components)
        if xs is None:
            xs = np.ones((m, 1))
        if offset is not None:
            ws = self._compute_weights(offset, xs)
        else:
            ws = np.tile(self.weights, reps=(ks.shape[0], 1))
        res = np.zeros((m, n_comps))
        for i, mc in enumerate(self.components):
            res[:, i] = np.log(ws[:, i]) + self.vec_loglike(
                mc.params,
                xs,
                ks,
                1.0,
                mc.mask,
                0.0,
            )
        return logsumexp(res, axis=-1).sum()

    vec_loglike = staticmethod(
        jit(vmap(DTMC.loglike, in_axes=(None, 0, 0, None, None, None)))
    )


class FiniteMixCTMC:

    """Continuous-time finite mixture of Markov chains."""

    def __init__(self, mask, n_comps):
        self.components = [CTMC(mask) for _ in range(n_comps)]
        self.weights = None

    def fit(self, ks, ts, max_iters=50, xs=None, l2=0.0, verbose=True, seed=0):
        """Train a finite mixture model using the EM algorithm."""
        m = ks.shape[0]
        n_comps = len(self.components)
        key = jax.random.PRNGKey(seed)
        fast = False
        if xs is None:
            fast = True
            xs = np.ones((m, 1))
        # Initial E-step (randomized soft clustering).
        ws = jax.random.dirichlet(key, jnp.ones(n_comps), shape=(m,))
        old_val = np.inf
        for i in range(max_iters):
            # M-step.
            val = 0.0
            for j, mc in enumerate(self.components):
                if fast:
                    ks2 = np.sum(ws[:, j, None, None] * ks, axis=0)
                    ts2 = np.sum(ws[:, j, None] * ts, axis=0)
                    mc.fit(ks2, ts2, verbose=False, l2=l2)
                else:
                    mc.fit(ks, ts, xs=xs, ws=ws[:, j], verbose=False, l2=l2)
                val += mc.res.fun
            self.weights = jnp.sum(ws, axis=0) / jnp.sum(ws)
            imp = 100 * (old_val / val - 1)  # % improvement.
            if verbose:
                print("cost: {:.3f}, imp: {:.2f}%".format(val, imp))
            if i > 6 and imp < 0.1:
                return
            old_val = val
            # E-step.
            ws = self._compute_weights(ks, ts, xs)

    def _compute_weights(self, ks, ts, xs):
        """Compute soft cluster assignments."""
        ws = np.zeros((ks.shape[0], len(self.components)))
        for i, mc in enumerate(self.components):
            ws[:, i] = np.log(self.weights[i]) + self.vec_loglike(
                mc.params, xs, ks, ts, 1.0, mc.mask, 0.0
            )
        return jax.nn.softmax(ws, axis=-1)

    def predict(self, p0, t, xs=None, offset=None):
        if xs is not None:
            m = xs.shape[0]
        else:
            m = p0.shape[0]
            xs = np.ones((m, 1))
        n = p0.shape[-1]
        if offset is not None:
            ws = self._compute_weights(*offset, xs)
        else:
            ws = np.tile(self.weights, reps=(m, 1))
        res = np.zeros((m, n))
        for i, mc in enumerate(self.components):
            res += np.expand_dims(ws[:, i], axis=-1) * mc.predict(p0, t, xs=xs)
        return res

    def predictive_loglike(self, ks, ts, xs=None, offset=None):
        m = ks.shape[0]
        n_comps = len(self.components)
        if xs is None:
            xs = np.ones((m, 1))
        if offset is not None:
            ws = self._compute_weights(*offset, xs)
        else:
            ws = np.tile(self.weights, reps=(ks.shape[0], 1))
        res = np.zeros((m, n_comps))
        for i, mc in enumerate(self.components):
            res[:, i] = np.log(ws[:, i]) + self.vec_loglike(
                mc.params,
                xs,
                ks,
                ts,
                1.0,
                mc.mask,
                0.0,
            )
        return logsumexp(res, axis=-1).sum()

    vec_loglike = staticmethod(
        jit(vmap(CTMC.loglike, in_axes=(None, 0, 0, 0, None, None, None)))
    )
