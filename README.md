# `apax`: Atomistic learned Potentials in JAX!
[![Read the Docs](https://readthedocs.org/projects/apax/badge/)](https://apax.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10040710.svg)](https://doi.org/10.5281/zenodo.10040710)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/1034511611802689557)](https://discord.gg/7ncfwhsnm4)

`apax`[1] is a high-performance, extendable package for training of and inference with atomistic neural networks.
It implements the Gaussian Moment Neural Network model [2, 3].
It is based on [JAX](https://jax.readthedocs.io/en/latest/) and uses [JaxMD](https://github.com/jax-md/jax-md) as a molecular dynamics engine.


## Installation

Apax is available on PyPI with a CPU version of JAX.

```bash
pip install apax
```

If you want to enable GPU support (only on Linux), please run
```bash
pip install "apax[cuda]"
```

For more detailed instructions, please refer to the [documentation](https://apax.readthedocs.io/en/latest/).

## Usage

### Your first apax Model

In order to train a model, you need to run

```bash
apax train config.yaml
```

We offer some input file templates to get new users started as quickly as possible.
Simply run the following commands and add the appropriate entries in the marked fields

```bash
apax template train # use --full for a template with all input options
```

Please refer to the [documentation](https://apax.readthedocs.io/en/latest/) for a detailed explanation of all parameters.
The documentation can convenienty be accessed by running `apax docs`.

## Molecular Dynamics

There are two ways in which `apax` models can be used for molecular dynamics out of the box.
High performance NVT simulations using JaxMD can be started with the CLI by running

```bash
apax md config.yaml md_config.yaml
```

A template command for MD input files is provided as well.

The second way is to use the ASE calculator provided in `apax.md`.


## Input File Auto-Completion

use the following command to generate JSON schemata for training and MD configuration files:

```bash
apax schema
```

If you are using VSCode, you can utilize them to lint and autocomplete your input files.
The command creates the 2 schemata and adds them to the projects `.vscode/settings.json`


## Authors
- Moritz René Schäfer
- Nico Segreto

Under the supervion of Johannes Kästner


## Contributing

We are happy to receive your issues and pull requests!

Do not hesitate to contact any of the authors above if you have any further questions.


## Acknowledgements

The creation of Apax was supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) in the framework of the priority program SPP 2363, “Utilization and Development of Machine Learning for Molecular Applications - Molecular Machine Learning” Project No. 497249646 and the Ministry of Science, Research and the Arts Baden-Württemberg in the Artificial Intelligence Software Academy (AISA).
Further funding though the DFG under Germany's Excellence Strategy - EXC 2075 - 390740016 and the Stuttgart Center for Simulation Science (SimTech) was provided.


## References
* [1] 10.5281/zenodo.10040711
* [2] V. Zaverkin and J. Kästner, [“Gaussian Moments as Physically Inspired Molecular Descriptors for Accurate and Scalable Machine Learning Potentials,”](https://doi.org/10.1021/acs.jctc.0c00347) J. Chem. Theory Comput. **16**, 5410–5421 (2020).
* [3] V. Zaverkin, D. Holzmüller, I. Steinwart,  and J. Kästner, [“Fast and Sample-Efficient Interatomic Neural Network Potentials for Molecules and Materials Based on Gaussian Moments,”](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00527) J. Chem. Theory Comput. **17**, 6658–6670 (2021).
