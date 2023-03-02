# Molecular Dynamics with JaxMD


JaxMD LINK is a high performance molecular dynamics engine built on top of Jax LINK.
Out of the boy, apax ships with a simple simulation loop using the Nose-Hoover-Chain thermostat implemented in JaxMD.
Note that this tutorial assumes that you have a trained model at hand.
See the previous tutorial LINK for further information.

## Configuration
We can once again use the template command to give ourselves a quickstart.

`apax template md --minimal`

Open the config and specify the starting structure and simulation parameters.
If you specify the data set file itself, the first structure of the data set is going to be used as the initial structure.
Your `md_config_minimal.yaml` should look similar to this:

```yaml
duration: 20_000 # fs
initial_structure: md17.extxyz
```

As with training configurations, we can use the `validate` command to ensure our input is valid before we submit the calculation.

## Running the simulation

The simulation can be started by running

`apax md config.yaml md_config_minimal.yaml`

where `config.yaml` is the configuration file that was used to train the model.

During the simulation, a progress bar tracks the instantaneous temperature at each outer step.

`prog bar`

## Calculating Vibrational Spectra

The trajectory calculated above can be used to obtain physical observables.
For this tutorial, we are going to compute an anharmonic vibrational spectrum for benzene.
Note that the code below is intended simply demonstration purposes and more sophisticated trajectory analysis tools should be used in production simulations.

```python
# analysis.py:
...
```

PIC

Congratulations, you have calculated the first observable from a trajectory generated with apax and jaxMD!



## Custom Simulation Loops

More complex simulation loops are relatively easy to build yourself in JaxMD (see their colab notebooks for examples). 
Trained apax models can of course be used as `energy_fn` in such custom simulations.
If you have a suggestion for adding some MD feature or thermostat to the core of `apax`, feel free to open up an issue on Github LINK.



