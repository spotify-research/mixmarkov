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
import collections
import jax
import jax.numpy as jnp
import numpy as np

from jax.example_libraries import optimizers
from scipy.special import digamma, logsumexp, log_softmax


def _s2t_continuous(seqs, n_states):
    max_len = max(len(seq) for seq in seqs)
    res = np.tile(
        np.append(np.zeros(n_states), -1.0),
        (len(seqs), max_len, 1),
    )
    basis = np.eye(n_states)
    for i, seq in enumerate(seqs):
        states, ts = map(tuple, zip(*seq))
        ts = np.insert(np.diff(ts), 0, 0.0)
        res[i,:len(seq),:] = np.hstack((
            basis[states,:], ts[:,np.newaxis],
        ))
    return jnp.array(res)


def _s2t_discrete(seqs, n_states):
    max_len = max(len(seq) for seq in seqs)
    res = np.tile(np.zeros(n_states), (len(seqs), max_len, 1))
    basis = np.eye(n_states)
    for i, seq in enumerate(seqs):
        res[i,:len(seq),:] = basis[seq,:]
    return jnp.array(res)


def seqs_to_tensor(seqs, n):
    """Transform sequences of states into RNN-ready tensors."""
    if isinstance(seqs[0][0], np.integer):
        return _s2t_discrete(seqs, n)
    elif len(seqs[0][0]) == 2:
        return _s2t_continuous(seqs, n)
    else:
        raise ValueError("data format not understood")


RNNFunctions = collections.namedtuple(
    "RNNFunctions",
    ["init", "predict", "loss"]
)


def _make_ct_rnn():
    """Continuous-time RNN."""

    def init(n_states, hidden_size, scale=0.01, seed=0):
        def rp(key, *shape):
            return scale * jax.random.normal(key, shape)
        key = jax.random.PRNGKey(seed)
        sks = jax.random.split(key, num=6)
        return {
            "h_0":   rp(sks[0], hidden_size),
            "W_in":  rp(sks[1], n_states, hidden_size),
            "W_h":   rp(sks[2], hidden_size, hidden_size),
            "b_h":   rp(sks[3], hidden_size),
            "W_out": rp(sks[4], hidden_size, n_states + 1),
            "b_out": rp(sks[5], n_states + 1),
        }

    @jax.jit
    def predict(params, xs):
        def step(h, x):
            h = jnp.tanh(
                jnp.dot(h, params["W_h"])
                + jnp.dot(x[:-1], params["W_in"])
                + params["b_h"]
            )
            y = jnp.dot(h, params["W_out"]) + params["b_out"]
            return h, y
        _, ys = jax.lax.scan(step, params["h_0"], xs)
        return ys

    @jax.jit
    def loss(params, xs, ys):
        pred = predict(params, xs)
        # Log-probability of transition.
        pred_st = jax.nn.log_softmax(pred[...,:-1])
        loglike_st = jnp.sum(jnp.where(
            # Check if actual transition (!= censored obs).
            jnp.any(xs[...,:-1] != ys[...,:-1], axis=-1, keepdims=True),
            pred_st * ys[...,:-1],
            0.0,
        ))
        # Log-density of time-to-transition.
        pred_z = jax.nn.softplus(pred[...,-1])
        loglike_z = jnp.sum(jnp.where(
            ys[...,-1] >= 0,
            jnp.where(
                jnp.any(xs[...,:-1] != ys[...,:-1], axis=-1),
                - ys[...,-1] / pred_z - jnp.log(pred_z),  # PDF.
                - ys[...,-1] / pred_z,  # 1 - CDF.
            ),
            0.0,
        ))
        # Combine both contributions into aggregate loss.
        return -(loglike_st + loglike_z)

    return RNNFunctions(init=init, predict=predict, loss=loss)


def _make_dt_rnn():
    """Discrete-time RNN."""

    def init(n_states, hidden_size, scale=0.01, seed=0):
        def rp(key, *shape):
            return scale * jax.random.normal(key, shape)
        key = jax.random.PRNGKey(seed)
        sks = jax.random.split(key, num=6)
        return {
            "h_0":   rp(sks[0], hidden_size),
            "W_in":  rp(sks[1], n_states, hidden_size),
            "W_h":   rp(sks[2], hidden_size, hidden_size),
            "b_h":   rp(sks[3], hidden_size),
            "W_out": rp(sks[4], hidden_size, n_states),
            "b_out": rp(sks[5], n_states),
        }

    @jax.jit
    def predict(params, xs):
        def step(h, x):
            h = jnp.tanh(
                jnp.dot(h, params["W_h"])
                + jnp.dot(x, params["W_in"])
                + params["b_h"]
            )
            y = jnp.dot(h, params["W_out"]) + params["b_out"]
            return h, y
        _, ys = jax.lax.scan(step, params["h_0"], xs)
        return ys

    @jax.jit
    def loss(params, xs, ys):
        pred = predict(params, xs)
        # Log-probability of transition.
        pred_st = jax.nn.log_softmax(pred)
        return -jnp.sum(pred_st * ys)

    return RNNFunctions(init=init, predict=predict, loss=loss)


class RNN(metaclass=abc.ABCMeta):

    def __init__(self, n_states, hidden_size):
        self.params = self.fns.init(n_states, hidden_size)

    def fit(self, tensor, lr=0.1, init_params=None, n_iters=10, verbose=True):
        if init_params is None:
            params = self.params
        else:
            params = init_params

        @jax.jit
        @jax.value_and_grad
        def loss(params):
            def step(tot, arr):
                tot += self.fns.loss(params, arr[:-1], arr[1:])
                return tot, None
            tot, _ = jax.lax.scan(step, 0.0, tensor)
            return tot

        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)

        for i in range(n_iters):
            value, grads = loss(get_params(opt_state))
            opt_state = opt_update(i, grads, opt_state)
            if verbose:
                print(f"iteration {i}, loss = {value}")
        self.params = get_params(opt_state)

    def predictive_loglike(self, tensor):
        def step(loss, arr):
            loss += self.fns.loss(self.params, arr[:-1], arr[1:])
            return loss, None
        loss, _ = jax.lax.scan(step, 0.0, tensor)
        return -loss


ct_rnn_fns = _make_ct_rnn()
dt_rnn_fns = _make_dt_rnn()


class DTRNN(RNN):

    """Discrete-time RNN."""

    fns = dt_rnn_fns


class CTRNN(RNN):

    """Continuous-time RNN."""

    fns = ct_rnn_fns


class Girolami2003:

    """
    Implementation of Girolami and Kaban, Simplicial Mixtures of Markov Chains,
    NIPS 2003 (variational Bayes algorithm).
    """

    def __init__(self, mask, n_comps, alpha):
        self.mask = mask
        self.n_comps = n_comps
        self.alpha = alpha

    @np.errstate(invalid="ignore", divide="ignore")
    def fit(self, ks, n_iters=10, seed=0):
        rng = np.random.default_rng(seed=seed)
        m, n, _ = ks.shape
        log_zs = np.zeros((m, self.n_comps, n, n))
        gs = self.alpha * np.ones((m, self.n_comps))
        log_thetas = log_softmax(
            np.where(self.mask, rng.normal(size=(self.n_comps, n, n)), -np.inf),
            axis=-1,
        )
        for i in range(n_iters):
            # Update zs.
            log_zs = np.where(
                self.mask,
                log_softmax(
                    log_thetas[None,:,:,:] + digamma(gs[:,:,None,None]),
                    axis=1,
                ),
                0.0,
            )
            # Update gammas.
            gs = self.alpha + np.sum(
                ks[:,None,:,:] * np.exp(log_zs),
                axis=(-1, -2),
            )
            # Update zs.
            log_zs = np.where(
                self.mask,
                log_softmax(
                    log_thetas[None,:,:,:] + digamma(gs[:,:,None,None]),
                    axis=1,
                ),
                0.0,
            )
            # Update transition matrices.
            log_thetas = log_softmax(
                logsumexp(np.log(ks[:,None,:,:]) + log_zs, axis=0),
                axis=-1,
            )
        self.gs = gs
        self.log_thetas = log_thetas

    @np.errstate(invalid="ignore")
    def predictive_loglike(self, ks, n_samples=100, seed=0):
        rng = np.random.default_rng(seed=seed)
        dists = rng.dirichlet(
            self.alpha * np.ones(self.n_comps),
            size=n_samples,
        )
        res = np.zeros((len(ks), n_samples))
        for i in range(n_samples):
            log_trans = logsumexp(
                np.log(dists[i,:,None,None]) + self.log_thetas,
                axis=0,
            )
            res[:,i] = np.sum(
                np.where(self.mask, ks * log_trans, 0.0),
                axis=(1, 2),
            )
        return np.sum(logsumexp(res - np.log(n_samples), axis=1))
