Property Prediction and Inference
================================

Apax allows for flexible property prediction during inference, which can be independent of the model's training configuration.
This is particularly useful for tasks like vibrational analysis (requiring Hessians) or stress tensor calculations for periodic systems.

Dynamic Property Control
------------------------

Models in Apax are capable of predicting derivatives of the energy (forces, stress, Hessians) even if they were not explicitly part of the training loss.
While training parameters like ``calc_stress`` or ``calc_hessian`` determine what is calculated during the training loop, you can override these behaviors during inference.

ASE Calculator
--------------

The ``ASECalculator`` is the primary interface for using Apax models with the Atomic Simulation Environment.
It accepts optional arguments to enable or disable specific property calculations.

Predicting Hessians
^^^^^^^^^^^^^^^^^^^

Even if a model was trained only on energies and forces, you can predict the Hessian matrix for tasks like frequency calculations.
Note that to use the analytical Hessian for vibrational analysis, you should use ASE's ``VibrationsData`` class, as the standard ``Vibrations`` class performs numerical differentiation of forces.

.. code-block:: python

    from apax.md.ase_calc import ASECalculator
    from ase.vibrations import VibrationsData

    # Enable Hessian calculation during initialization
    calc = ASECalculator("path/to/model", calc_hessian=True)
    atoms.calc = calc

    # Get the analytical Hessian (3N x 3N matrix)
    hessian = atoms.get_hessian()

    # Use the analytical Hessian for vibrational analysis
    vib = VibrationsData(atoms, hessian)
    freqs = vib.get_frequencies()

Disabling Expensive Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a model was trained with ``calc_stress: True``, but you are performing high-throughput inference on isolated molecules where stress is not needed, you can disable it to save computation:

.. code-block:: python

    # Disable stress calculation even if enabled in training config
    calc = ASECalculator("path/to/model", calc_stress=False)

Available Overrides
^^^^^^^^^^^^^^^^^^^

*   ``calc_stress``: Boolean. Enables/disables stress tensor calculation.
*   ``calc_hessian``: Boolean. Enables/disables Hessian matrix calculation.
*   ``force_variance``: Boolean. For shallow ensembles, enables/disables force uncertainty prediction.

Molecular Dynamics (JaxMD)
--------------------------

When running MD simulations with the CLI tool (``apax md``), the properties calculated are determined by the ``properties`` list in the MD configuration file.

.. code-block:: yaml

    # md_config.yaml
    properties:
      - energy
      - forces
      # Stress will only be calculated if it's in this list
      # - stress
      # Hessian is usually not needed during MD
      # - hessian

If a property like ``forces_uncertainty`` is requested in ``properties`` or used in ``dynamics_checks``, the model will automatically enable the necessary internal calculations (e.g., ``force_variance`` for shallow ensembles).
