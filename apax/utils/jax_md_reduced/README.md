# Reduced `Jax M.D.`

apax relies on the amazing `jax_md` package for neighborlists and thermostats.
Some of our use cases run into bugs in functionality provided by `jax_md`.
Hence, this submodule contains a small number of fixes for some `jax_md` features.

We would like to thank the developers of `jax_md` for the work on this great package.

```
@inproceedings{jaxmd2020,
 author = {Schoenholz, Samuel S. and Cubuk, Ekin D.},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 title = {JAX M.D. A Framework for Differentiable Physics},
 url = {https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
