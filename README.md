# SiliconAI

[![10.5281/zenodo.17568416][zenodo-img]][zenodo]
[![Latest release][release-img]][release]
[![License][license-img]][license]
[![pre-commit][pre-commit-img]][pre-commit]
[![Continuous Integration][ci-img]][ci]
[![codecov.io][codecov-img]][codecov]

Silicon detector simulation using AI.

## Setup

The Python project uses [uv](https://docs.astral.sh/uv/).
It should be installed in the system and available in the `PATH`.

To make the virtual environment and install the required packages,
run the following command:

```bash
uv sync
```

The virtual environment can be then entered as usual:

```bash
source .venv/bin/activate
```

## Main workflow

- Prepare inputs/tokenizers (before training):
  - Convert raw inputs: `siliconai convert_inputs -c config/Muons_windowed.toml`
  - Train tokenizer: `siliconai tokenize -c config/Muons_windowed.toml`
- Train and validate results:
  - `siliconai train -c config/Muons_windowed.toml`
  - `siliconai validate -c config/Muons_windowed.toml`

Furthermore basic Slurm batch processing support is available.
For this and full available options consult `siliconai --help`.

## License

Copyright (C) 2024 Tadej Novak

This project is published under the terms of the Mozilla Public
License, v. 2.0, available in the file [LICENSE.md](LICENSE.md)
and at <http://mozilla.org/MPL/2.0/>.

<!--
SPDX-License-Identifier: MPL-2.0
-->

## Acknowledgements

This software is supported by the European Union's Horizon
Europe research and innovation programme under the Marie Skłodowska-Curie
Postdoctoral Fellowship Programme, SMASH, co-funded under the grant agreement
No. 101081355. The SMASH project is co-funded by the Republic of Slovenia and
the European Union from the European Regional Development Fund.

[zenodo]: https://doi.org/10.5281/zenodo.17568416
[zenodo-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.17568416.svg
[release]: https://github.com/ntadej/SiliconAI/releases/latest
[release-img]: https://img.shields.io/github/release/ntadej/SiliconAI.svg
[license]: https://github.com/ntadej/SiliconAI/blob/main/LICENSE.md
[license-img]: https://img.shields.io/github/license/ntadej/SiliconAI.svg
[pre-commit]: https://results.pre-commit.ci/latest/github/ntadej/SiliconAI/main
[pre-commit-img]: https://results.pre-commit.ci/badge/github/ntadej/SiliconAI/main.svg
[ci]: https://github.com/ntadej/SiliconAI/actions
[ci-img]: https://github.com/ntadej/SiliconAI/workflows/Continuous%20Integration/badge.svg
[codecov]: https://codecov.io/github/ntadej/SiliconAI?branch=main
[codecov-img]: https://codecov.io/github/ntadej/SiliconAI/coverage.svg?branch=main
