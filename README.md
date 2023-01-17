## Roadmap

- [x] basic loading of fixed size ASE structures into `tf.data.Dataset`
- [x] basic linear regressor atomic number -> energy
- [x] per-example model + `vmap utiliation`
- [x] loading model parameters from TF GMNN
- [x] basic training loop
  - [x] basic metrics
  - [x] hooks / tensorboard
  - [x] model checkpoints
  - [x] restart
- [ ] advanced training loop
  - [ ] MLIP metrics
  - [x] async checkpoints
  - [x] jit compiled metrics
- [x] dataset statistics
- [x] precomputing neighborlists with `jax_md`
- [ ] tests
- [ ] documentation
- [ ] generalize to differently sized molecules
- [x] Optimizer with different lr for different parameter groups
- [x] GMNN energy model with `jax_md`
- [x] force model
- [x] running MD with GMNN


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