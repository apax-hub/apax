# Training

## Acquiring a dataset

In this tutorial we are going to train a model from scratch on a molecular dataset from the MD17 collection.
Start by creating a project folder and downloading the dataset.

mkdir project
cd project

You can obtain the benzene dataset either by running the following command or manually from this website.

curl ... ...

apax uses ASE to read in datasets, so make sure to convert your own data into an ASE readable format (extxyz, traj etc).


## Configuration files

Next, we require a configuration file that specifies the model and training parameters.
In order to get users quickly up and running, our command line interface provides an easy way to generate input templates.
The provided templates come in in two levels of verbosity: minimal and full.
In the following we are going to use a minimal input file. To see a complete list and explanation of all parameters, consult the documentation page LINK.
For more information on the CLI,  simply run `apax -h`.

apax template train --minimal

Open the resulting `config_minimal.yaml` file in an editor of your choice and make sure to fill in the data path field with the name of the data set you just downloaded.
For the purposes of this tutorial we will train on 1000 data points and validate the model on 200 more during the training.

The filled in configuration file should look similar to this one.

```yaml
data:
    data_path: md17.extexyz
    epochs: 1000
    n_train: 1000
    ....
```

In order to check whether the a configuration file is valid, we provide the `validate` command. This is especially convenient when submitting training runs on a compute cluster.

`apax validate train config_minimal.yaml`

Configuration files are validated using Pydantic and the errors provided by the `validate` command give precise instructions on how to fix the input file.
For example, changing `epochs` to `-1000`, validate will give the following feedback to the user:

`PYDANTIC ERROR`

## Training

Model training can be started by running

`apax train config.yaml`

During training, apax displays a progress bar to keep track of the validation loss.
This progress bar is optional however and can be turned off in the config. LINK
The default configuration writes training metrics to a CSV file, but TensorBoard is also supported.
One can specify which to use by adding the following section to the input file:

```yaml
callbacks:
    - CSV
```

If training is interupted for any reason, re-running the above `train` command will resume training from the latest checkpoint.

## Evaluation

After the training is completed and we are satisfied with our choice of hyperparameters and vadliation loss, we can evaluate the model on the test set.
We provide a separate command for test set evaluation:

`apax evaluate config_minimal.yaml`

TODO pretty print results to the terminal

Congratulations, you have successfully trained and evaluated your fitrst apax model!