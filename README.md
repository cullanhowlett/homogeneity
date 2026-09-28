# Homogeneity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for measuring the scale of transition to large-scale homogeneity in the DESI DR1 galaxy and quasar samples.

## Contents

### Measuring the mean scaled counts

| Script | Description |
|---|---|
| `script_NERSC_data.py` | Compute mean scaled counts for DESI DR1 data |
| `script_NERSC.py` | Compute mean scaled counts for altmtl AbacusSummit mocks |
| `script_NERSC_EZmock.py` | Compute mean scaled counts for ffa EZmocks |

### Toy models

| Script | Description | Figure |
|---|---|:---:|
| `generate_toy_models.py` | Generate particle boxes with uniform, Gaussian, void and Swiss-cheese distributions | |
| `compute_toy_models.py` | Compute mean scaled counts for the toy models | |
| `plot_toy_models.py` | Plot toy-model particle distributions and mean scaled counts | 3 |

### Model-independent fits (B-spline + emcee)

| Script | Description | Figure |
|---|---|:---:|
| `fit_data.py` | Fit the DESI data | |
| `fit_data_binning.py` | Fit the data with varying bin widths and numbers of spline knots | 10 |
| `fit_data_scalecuts.py` | Fit the data with varying minimum and maximum scale cuts | 11 |
| `fit_mock_average.py` | Fit the mean of the AbacusSummit mocks | |
| `fit_mock_individual.py` | Fit individual AbacusSummit realisations | |
| `plot_fit_data.py` | Plot mean scaled counts and correlation dimensions for each tracer | 4, 5 |

### Full-shape fits (desilike + pocomc)

| Script | Description | Figure |
|---|---|:---:|
| `desilike_routines.py` | Correlation-dimension models built on desilike full-shape theories, plus fitting routines | |
| `plot_RH.py` | Plot model correlation dimensions for different input parameters | 6 |
| `fit_data_desilike.py` | Fit the DESI data | |
| `fit_mock_average_desilike.py` | Fit the mean of the AbacusSummit mocks | |
| `fit_mock_individual_desilike.py` | Fit individual AbacusSummit realisations | |
| `compare_RH_desilike.py` | Compare chains for individual and mean mocks | 7 |
| `plot_fit_data_desilike.py` | Plot mean scaled counts and correlation dimensions for each tracer | 8 |
| `rescale_MSC.py` | Rescale the mock-average correlation dimension to that of matter | |
| `rescale_MSC_data.py` | Rescale the data correlation dimension to that of matter | |

### Results and utilities

| Script | Description | Figure |
|---|---|:---:|
| `plot_homogeneity_vs_redshift.py` | Plot homogeneity scale against redshift, with literature comparisons | 9 |
| `python_routines.py` | Utilities for reading mean scaled counts, plotting and model-independent fitting | |

## Data

- **Input catalogues and mocks:** DESI DR1 public release, [data.desi.lbl.gov/public/dr1](https://data.desi.lbl.gov/public/dr1/)
- **Derived outputs** (mean scaled counts, MCMC chains): Zenodo, [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

Extract the Zenodo archive into the repository root:

```
homogeneity/
├── BGS/  LRG1/  LRG2/  LRG3/  ELG2/  QSO/  toy_models/
├── *.yaml
└── *.py
```

## Requirements

- Python ≥ 3.10
- [numpy](https://numpy.org), [scipy](https://scipy.org), [pandas](https://pandas.pydata.org), [matplotlib](https://matplotlib.org)
- [astropy](https://www.astropy.org), [scikit-learn](https://scikit-learn.org)
- [emcee](https://emcee.readthedocs.io), [pocomc](https://pocomc.readthedocs.io), [ChainConsumer](https://samreay.github.io/ChainConsumer/) ≥ 1.0
- [JAX](https://github.com/jax-ml/jax), [desilike](https://github.com/cosmodesi/desilike), [velocileptors](https://github.com/sfschen/velocileptors)

```bash
pip install numpy scipy pandas matplotlib astropy scikit-learn emcee pocomc "chainconsumer>=1.0" jax
pip install git+https://github.com/cosmodesi/desilike
pip install git+https://github.com/sfschen/velocileptors
```

## Citation

If you use this code or data, please cite:

> Howlett C., et al., *The scale of homogeneity in DESI Data Release 1*, in prep. [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Licence

Released under the MIT License. See [LICENSE](LICENSE) for details.
