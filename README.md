# mixmarkov

Reproducibility package for the paper:

> Lucas Maystre, Tiffany Wu, Roberto Sanchis Ojeda, Tony Jebara.
> _[Multistate Analysis with Infinite Mixtures of Markov Chains](#)_, UAI 2022.

This repository contains

- a reference implementation of the algorithms presented in the paper, and
- Jupyter notebooks enabling the reproduction of some of the experiments.

The paper and the library address the problem of predicting trajectories over a
small number of states. The main goal is to estimate a model that makes
accurate and calibrated probabilistic predictions about states at future points
in time, given a sequence's past.

## Getting started

To get started, follow these steps:

- Clone the repo locally with: `git clone
  https://github.com/spotify-research/mixmarkov.git`
- Move to the repository: `cd mixmarkov`
- Install the dependencies: `pip install -r requirements.txt`
- Install the package: `pip install -e lib/`
- Move to the notebook folder: `cd notebooks`
- Start a notebook server: `jupyter notebok`

Our codebase was tested with Python 3.8. The following libraries are required
(and installed automatically via the first `pip` command above):

- `numpy` (tested with version 1.22.4)
- `scipy` (tested with version 1.8.1)
- `matplotlib` (tested with version 3.5.2)
- `networkx` (tested with version 2.8.3)
- `jax` (tested with version 0.3.13)
- `notebook` (tested with version 6.4.11)

## Support

Create a [new issue](https://github.com/spotify-research/mixmarkov/issues/new)


## Contributing

We feel that a welcoming community is important and we ask that you follow
Spotify's [Open Source Code of
Conduct](https://github.com/spotify/code-of-conduct/blob/master/code-of-conduct.md)
in all interactions with the community.


## Author

[Lucas Maystre](mailto:lucasm@spotify.com)

A full list of [contributors](https://github.com/spotify-research/cosernn/graphs/contributors?type=a) can
be found on GitHub.

Follow [@SpotifyResearch](https://twitter.com/SpotifyResearch) on Twitter for
updates.


## License

Copyright 2022 Spotify AB.

Licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0


## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program
(https://hackerone.com/spotify) rather than GitHub.
