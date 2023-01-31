# `gmnn-jax`: Gaussian Moment Neural Networks in Jax!
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)




## Installation

You can install [Poetry](https://python-poetry.org/) via

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Now you can install GMNN in your project by running

```bash
poetry add git+https://github.com/GM-NN/gmnn-jax.git
```

As a developer, you can clone the repository and install it via

```bash
git clone https://github.com/GM-NN/gmnn-jax.git <dest_dir>
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


## References
* [1] ZENODO DOI PLACEHOLDER
* [2] V. Zaverkin and J. Kästner, [“Gaussian Moments as Physically Inspired Molecular Descriptors for Accurate and Scalable Machine Learning Potentials,”](https://doi.org/10.1021/acs.jctc.0c00347) J. Chem. Theory Comput. **16**, 5410–5421 (2020).
* [3] V. Zaverkin, D. Holzmüller, I. Steinwart,  and J. Kästner, [“Fast and Sample-Efficient Interatomic Neural Network Potentials for Molecules and Materials Based on Gaussian Moments,”](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00527) J. Chem. Theory Comput. **17**, 6658–6670 (2021).
