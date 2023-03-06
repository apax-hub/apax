# `apax`: Atomstic learned Potentials in JAX!
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`apax` is a high-performance, extendable package for training of and inference with atomistic neural networks.
It implements the Gaussian Moment Neural Network model [2, 3].
It is based on [JAX](https://jax.readthedocs.io/en/latest/) and uses [JaxMD](https://github.com/jax-md/jax-md) as a molecular dynamics engine.


## Installation

You can install [Poetry](https://python-poetry.org/) via

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Now you can install apax in your project by running

```bash
poetry add git+https://github.com/GM-NN/apax.git
```

As a developer, you can clone the repository and install it via

```bash
git clone https://github.com/GM-NN/apax.git <dest_dir>
cd <dest_dir>
poetry install
```

### CUDA Support
Note that the above only installs the CPU version.
If you want to enable GPU support, please overwrite the jaxlib version:

```bash
pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See the [Jax installation instructions](https://github.com/google/jax#installation) for more details.



## Usage

### Your first apax Model

In order to train a model, you need to run

```python
apax train config.yaml
```

We offer some input file templates to get new users started as quickly as possible.
Simply run the following commands and add the appropriate entries in the marked fields

```python
apax template train # use --full for a template with all input options
```

Please refer to the documentation LINK for a detailed explanation of all parameters.
The documentation can convenienty be accessed by runnning `apax docs`.

## Molecular Dynamics

There are two ways in which `apax` models can be used for molecular dynamics out of the box.
High performance NVT simulations using JaxMD can be started with the CLI by running

```python
apax md config.yaml md_config.yaml
```

A template command for MD input files is provided as well.

The second way is to use the ASE calculator provided in `apax.md`.


## Authors
- Moritz René Schäfer
- Nico Segreto

Under the supervion of Johannes Kästner

## References
* [1] DOI PLACEHOLDER
* [2] V. Zaverkin and J. Kästner, [“Gaussian Moments as Physically Inspired Molecular Descriptors for Accurate and Scalable Machine Learning Potentials,”](https://doi.org/10.1021/acs.jctc.0c00347) J. Chem. Theory Comput. **16**, 5410–5421 (2020).
* [3] V. Zaverkin, D. Holzmüller, I. Steinwart,  and J. Kästner, [“Fast and Sample-Efficient Interatomic Neural Network Potentials for Molecules and Materials Based on Gaussian Moments,”](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00527) J. Chem. Theory Comput. **17**, 6658–6670 (2021).


## Contributing

We are happy to receive your issues and pull requests!

Do not hesitate to contact any of the authors above if you have any further questions.
