Installation
============

If you do not have Poetry_ installed already, you can obtain it by running

.. highlight:: bash
.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -


Now you can add apax to your project by running

.. highlight:: bash
.. code-block:: bash

    poetry add git+https://github.com/apax-hub/apax.git

As a developer, you first need to clone the repository and install it via

.. highlight:: bash
.. code-block:: bash

    git clone https://github.com/apax-hub/apax.git <dest_dir>
    cd <dest_dir>
    poetry install

CUDA Support
============
Note that the above only installs the CPU version.
If you want to enable GPU support, please overwrite the jaxlib version:

.. highlight:: bash
.. code-block:: bash

    pip install --upgrade pip

    # CUDA 12 installation
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # CUDA 11 installation
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


See the `Jax installation instructions <https://github.com/google/jax#installation>`_ for more details.


.. _Poetry: https://python-poetry.org/
