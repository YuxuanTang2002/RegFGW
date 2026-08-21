# RegFGW

[![Tests](https://github.com/YuxuanTang2002/RegFGW/actions/workflows/tests.yaml/badge.svg)](https://github.com/YuxuanTang2002/RegFGW/actions/workflows/tests.yaml)
[![PyPI](https://img.shields.io/pypi/v/regfgw)](https://pypi.org/project/regfgw/)
[![License](https://img.shields.io/github/license/YuxuanTang2002/RegFGW)](https://github.com/YuxuanTang2002/RegFGW/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2608.19933-b31b1b.svg)](https://arxiv.org/abs/2608.19933)

## Overview

RegFGW is a graph-based framework for pre-screening atomic interface registries before expensive structural relaxation. 
It uses the fused Gromov–Wasserstein (FGW) distance to quantify the structural deviation of each registry from its corresponding bulk reference. 
Bayesian optimization is then used to efficiently explore the registry space. 
This enables low-cost and physically interpretable interface modeling across diverse functional materials.

## Installation

### Installation from PyPI

The recommended installation method is via PyPI.

```bash
pip install --upgrade pip
pip install regfgw
```

### Installation from source

```bash
git clone https://github.com/YuxuanTang2002/RegFGW.git
pip install ./RegFGW
```

## Usage

RegFGW provides the `regfgw_coherent` command-line interface for coherent interface construction and registry optimization. Use `--help` to view all available options.

### Interface construction

Construct coherent interface candidates from substrate and film bulk structures:

```bash
regfgw_coherent \
  --mode build \
  --substrate substrate.cif \
  --film film.cif \
  --out-dir results
```

Construction settings can be specified as needed:

```bash
regfgw_coherent \
  --mode build \
  --substrate substrate.cif \
  --film film.cif \
  --out-dir results \
  --max-miller-idx 1 \
  --substrate-layers 3 \
  --film-layers 3 \
  --gap 5.0 \
  --vacuum 20.0 \
  --zsl-max-area 150.0 \
  --zsl-area-ratio 0.06 \
  --zsl-length 0.03 \
  --zsl-angle 0.02 
```

### Registry optimization

Construct coherent interface candidates and perform FGW-guided Bayesian optimization of selected candidates:

```bash
regfgw_coherent \
  --mode optimize \
  --substrate substrate.cif \
  --film film.cif \
  --out-dir result \
  --embedding embedding.json \
  --budget 3 
```

The construction arguments above also apply in this mode. Generated interface candidates are displayed for selection before registry optimization.

### YAML configuration

Options can also be supplied through a YAML configuration file:

```bash
regfgw_coherent --config config.yaml
```

Example configuration:

```yaml
mode: optimize
substrate: substrate.cif
film: film.cif
out-dir: results
embedding: embedding.json
max-miller-idx: 1
substrate-layers: 3
film-layers: 3
gap: 5.0
vacuum: 20.0
budget: 3
```

Command-line arguments override values specified in the configuration file.

## Reproducibility

Structures, calculated results and analysis notebooks are provided in [`examples`](examples), with supporting scripts in [`scripts`](scripts). Detailed computational settings are described in the correponding paper.

## Citation

If you use RegFGW in your research, please cite our paper:

```bibtex
@misc{tang2026heterointerfaces, 
      title={Building atomistic models of heterointerfaces with optimal transport}, 
      author={Yuxuan Tang and Keith T. Butler},
      year={2026},
      eprint={2608.19933},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2608.19933}, 
}
```

## License
This project is licensed under the [MIT License](LICENSE).
