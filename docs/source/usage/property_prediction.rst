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

    # Get the analytical Hessian (3N x 3N matrix) from the calculator
    hessian = calc.get_hessian(atoms)

    # Use the analytical Hessian for vibrational analysis
    vib = VibrationsData.from_2d(atoms, hessian)
    freqs = vib.get_frequencies()

Training with Hessians
----------------------

While apax models can predict Hessians without being trained on them (either via AD or finite differences), explicitely including them in the training can significantly improve the accuracy of vibrational frequencies and local curvature.

Datasets
^^^^^^^^

The most straightforward way to work with Hessians in apax is to include them in an H5MD dataset under the ´´hessian´´ key.
If you are preparing such a dataset, the shapes of the Hessian should be (N, 3, N, 3).
Apax handles the masking of Hessian elements in differently sized systems automatically.


Best Practices
^^^^^^^^^^^^^^

Due to the high cost of second order derivatives and the magnitude imbalance of Hessian elements compared to forces, we recommend the following workflow:

1.  **EF Pretraining**: Train a base model on energies and forces (EF) until convergence.
2.  **EFH Fine-tuning**: Fine-tune the pretrained model with the Hessian loss (EFH) enabled.
3.  **Mass-Weighted Loss**: Use the ``mw_hessian`` loss type. By dividing Hessian elements by :math:`\sqrt{m_i m_j}`, the loss becomes physically more representative of vibrational modes and helps balance the optimization.
4.  **Low Weight**: Use a low weight for the Hessian loss (e.g., ``0.01`` or ``0.005``). High Hessian weights can destabilize force accuracy due to the large scale of second-derivative elements.
5.  **Model Capacity**: Training on second derivatives increases the complexity of the loss landscape. Consider increasing ``n_radial`` or the neural network width if the model struggles to balance forces and Hessians.

This is a relatively basic implementation of Hessian training.
More involved setups could include projecting out rigid body degrees of freedom and accelerated training via Hessian vector product matching.
Feel free to reach out if you feel like your work would benefit from these additions.


Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    model:
      calc_hessian: True

    loss:
      - name: energy
      - name: forces
        weight: 4.0
      - name: hessian
        loss_type: mw_hessian
        weight: 0.01

    transfer_learning:
      base_model_checkpoint: path/to/ef_model

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
