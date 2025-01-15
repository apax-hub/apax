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

As a developer, you first need to install uv_.
You can obtain it by running

.. highlight:: bash
.. code-block:: bash

    pip install uv


You can then clone and install the project.

.. highlight:: bash
.. code-block:: bash

    git clone https://github.com/apax-hub/apax.git <dest_dir>
    cd <dest_dir>
    uv sync --all-extras --dev


=========================
CUDA Support (Linux only)
=========================

Note that all of the above only install the CPU version.
If you want to enable GPU support, please overwrite the jaxlib version:

CUDA 12:

.. highlight:: bash
.. code-block:: bash

    pip install -U "jax[cuda12]"

See the `Jax installation instructions <https://github.com/google/jax#installation>`_ for more details.


.. _uv: https://astral.sh/blog/uv
