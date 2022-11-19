## Roadmap

- [x] basic loading of fixed size ASE structures into `tf.data.Dataset`
- [x] basic linear regressor atomic number -> energy
- [ ] per-example model + `vmap utiliation`
- [ ] basic training loop
  - [ ] metrics
  - [ ] hooks / tensorboard
  - [ ] model checkpoints
  - [ ] restart
- [ ] dataset statistics
- [x] precomputing neighborlists with `jax_md`
- [ ] tests
- [ ] documentation
- [ ] generalize to differently sized molecules
- [ ] Optimizer with different lr for different parameter groups
- [ ] GMNN energy model with `jax_md`
- [ ] force model
- [ ] running MD with GMNN
