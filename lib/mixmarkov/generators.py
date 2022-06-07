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

import numpy as np
import typing


class ContinuousData(typing.NamedTuple):
    mask: np.ndarray
    params: np.ndarray
    xs: np.ndarray
    ks: np.ndarray
    ts: np.ndarray
    ks_offset: np.ndarray = None
    ts_offset: np.ndarray = None


def generate_gam_mix_ctmc(
        *, mask, d=1, m=1, n_trans=10, split=None, p0=None, seed=None):
    rng = np.random.default_rng(seed=seed)
    n = len(mask)
    params = 0.3 * rng.normal(size=(d, n, n, 2))
    # Last column is always the offset.
    xs = np.hstack((rng.normal(size=(m, d-1)), np.ones((m, 1))))
    zs = np.exp(np.tensordot(xs, params, axes=(-1, 0)))
    alpha = zs[...,0]
    beta = zs[...,1]
    ks = np.zeros((m, n, n), dtype=int)
    ts = np.zeros((m, n))
    ks_offset = np.zeros((m, n, n), dtype=int) if split is not None else None
    ts_offset = np.zeros((m, n)) if split is not None else None
    for i in range(m):
        rates = np.where(mask, rng.gamma(shape=alpha[i], scale=1/beta[i]), 0.0)
        cur = rng.choice(n, p=p0)
        for j in range(n_trans):
            t = np.random.exponential(scale=1/rates[cur].sum())
            nxt = np.random.choice(n, p=rates[cur]/rates[cur].sum())
            if split is not None and j < split:
                ts_offset[i,cur] += t
                ks_offset[i,cur,nxt] += 1
            else:
                ts[i,cur] += t
                ks[i,cur,nxt] += 1
            cur = nxt
    return ContinuousData(
        mask=mask,
        params=params,
        xs=xs,
        ks=ks,
        ts=ts,
        ks_offset=ks_offset,
        ts_offset=ts_offset,
    )
