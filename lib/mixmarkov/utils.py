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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings

from jax.scipy.linalg import expm
from functools import partial

expm_ = partial(expm, max_squarings=100)
parallel_expm = jax.vmap(expm_)


@jax.jit
def random_gd(key, alpha, beta, mask, last):
    """Sample from a generalized Dirichlet disitribution.
    
    Stick-breaking algorithm based on Wong (1998).
    """
    def f(sum_, x):
        val = x * (1 - sum_)
        sum_ += val
        return (sum_, val)
    rvs = jnp.transpose(
        jnp.where(mask, jax.random.beta(key, alpha, beta), 0.0)
    )
    _, res = jax.lax.scan(f, jnp.zeros(rvs.shape[1:]), rvs)
    # Add the remainder of the probability mass to the correct position.
    res = jnp.concatenate((res, jnp.zeros_like(res[:1])))
    res = jax.ops.index_update(
        res,
        jax.ops.index[last,jnp.arange(len(mask))],
        1.0 - jnp.sum(res, axis=0)
    )
    return jnp.transpose(res)


def draw_chain(mask):
    graph = nx.DiGraph()
    graph.add_edges_from(np.argwhere(mask))
    if nx.number_of_selfloops(graph) > 0:
        warnings.warn("chain has self-loops that are not drawn")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_color="white",
        edgecolors="black",
        node_size=800
    )
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        connectionstyle="arc3,rad=0.1",
        node_size=800,
        arrowsize=15
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos
    )


def _summarize_continuous(seqs, n, t_split=None):
    ts = np.zeros((len(seqs), n))
    ks = np.zeros((len(seqs), n, n), dtype=int)
    if t_split is not None:
        ts_offset = np.zeros((len(seqs), n))
        ks_offset = np.zeros((len(seqs), n, n), dtype=int)
    for i, traj in enumerate(seqs):
        for (src, t1), (dst, t2) in zip(traj, traj[1:]):
            if t_split is not None and t1 < t_split:
                ts_offset[i, src] += min(t2, t_split) - t1
                if t2 <= t_split:
                    if src != dst:
                        ks_offset[i, src, dst] += 1
                    continue
                # t2 > t_split
                t1 = t_split
            if src != dst:
                ks[i, src, dst] += 1
            ts[i, src] += t2 - t1
    if t_split is not None:
        return (ks_offset, ts_offset), (ks, ts)
    else:
        return ks, ts


def _summarize_discrete(seqs, n, split):
    ks = np.zeros((len(seqs), n, n), dtype=int)
    if split is not None:
        ks_offset = np.zeros((len(seqs), n, n), dtype=int)
    for i, traj in enumerate(seqs):
        for j, (src, dst) in enumerate(zip(traj, traj[1:])):
            if split is not None and j < split:
                ks_offset[i, src, dst] += 1
            else:
                ks[i, src, dst] += 1
    if split is not None:
        return ks_offset, ks
    else:
        return ks


def summarize_sequences(seqs, n, split=None):
    """Transform sequences of states into transition count matrices.

    seqs : list[list]
        Discrete-time or continuous-time sequences of states.
    n : int
        Number of states.
    split : int, optional
        Split each sequence into head and tail subsequences.
    """
    if isinstance(seqs[0][0], (int, np.integer)):
        return _summarize_discrete(seqs, n, split)
    elif len(seqs[0][0]) == 2:
        return _summarize_continuous(seqs, n, split)
    else:
        raise ValueError("data format not understood")
