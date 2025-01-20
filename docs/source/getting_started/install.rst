============
Installation
============

The standard installation installs the CPU version of Apax. To enable GPU
support (available only on **Linux**), install Apax with the extra CUDA
dependencies. See the
`Jax installation instructions <https://github.com/google/jax#installation>`_
for more details.

From PyPI
---------

**CPU:**

.. highlight:: bash
.. code-block:: bash

    pip install apax

**GPU:**

.. highlight:: bash
.. code-block:: bash

    pip install "apax[cuda]"

From GitHub
-----------

For a pre-release version, install Apax directly from GitHub.

**CPU:**

.. highlight:: bash
.. code-block:: bash

    pip install git+https://github.com/apax-hub/apax.git

**GPU:**

.. highlight:: bash
.. code-block:: bash

    pip install apax[cuda] git+https://github.com/apax-hub/apax.git

For Developers
--------------

To set up a development environment, first install `uv`_.

.. highlight:: bash
.. code-block:: bash

    pip install uv


Then clone the project from GitHub,

.. highlight:: bash
.. code-block:: bash

    git clone https://github.com/apax-hub/apax.git <dest_dir>
    cd <dest_dir>

and install it.

**CPU:**

.. highlight:: bash
.. code-block:: bash

    uv sync --all-extras --no-extra cuda

**GPU:**

.. highlight:: bash
.. code-block:: bash

    uv sync --extra cuda

Extra Dependencies
------------------

If you want to use Apax in the IPSuite framework and use the predefined
`apax.nodes`, you can install the extra dependencies for IPSuite:

.. highlight:: bash
.. code-block:: bash

    pip install "apax[ipsuite]"

Additionally, you have the option to install the extra dependencies for MLFlow:

.. highlight:: bash
.. code-block:: bash

    pip install "apax[mlflow]"

.. _uv: https://astral.sh/blog/uv
