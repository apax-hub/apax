============
Installation
============

From PyPI (soon)

.. highlight:: bash
.. code-block:: bash

    pip install apax


From GitHub
-----------

If you would like to have a pre-release version,
you can install Apax from GitHub directly

.. highlight:: bash
.. code-block:: bash

    pip install git+https://github.com/apax-hub/apax.git


For Developers
--------------

As a developer, you first need to install Poetry_.
You can obtain it by running

.. highlight:: bash
.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -


You can then clone and install the project.

.. highlight:: bash
.. code-block:: bash

    git clone https://github.com/apax-hub/apax.git <dest_dir>
    cd <dest_dir>
    poetry install


=========================
CUDA Support (Linux only)
=========================

Note that all of the above only install the CPU version.
If you want to enable GPU support, please overwrite the jaxlib version:

CUDA 12:

.. highlight:: bash
.. code-block:: bash

    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CUDA 11:

.. highlight:: bash
.. code-block:: bash

    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


See the `Jax installation instructions <https://github.com/google/jax#installation>`_ for more details.


.. _Poetry: https://python-poetry.org/
