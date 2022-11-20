## Roadmap

- [x] basic loading of fixed size ASE structures into `tf.data.Dataset`
- [x] basic linear regressor atomic number -> energy
- [ ] per-example model + `vmap utiliation`
- [x] basic training loop
  - [x] basic metrics
  - [x] hooks / tensorboard
  - [x] model checkpoints
  - [x] restart
- [ ] advanced training loop
  - [ ] MLIP metrics
  - [ ] async checkpoints
  - [ ] jit compiled metrics
- [ ] dataset statistics
- [x] precomputing neighborlists with `jax_md`
- [ ] tests
- [ ] documentation
- [ ] generalize to differently sized molecules
- [ ] Optimizer with different lr for different parameter groups
- [ ] GMNN energy model with `jax_md`
- [ ] force model
- [ ] running MD with GMNN
