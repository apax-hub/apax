Transfer Learning
=================

apax comes with discriminative transfer learning capabilities out of the box.
In this tutorial we are going to fine tune a model trained on benzene data at the DFT level of theory to CCSDT.

First download the appropriate dataset from the sgdml website.


Transfer learning can be facilitated in apax by adding the path to a pre-trained model in the config.
Furthermore, we can freeze or reduce the learning rate of various components by adjusting the `optimizer` section of the config.

```yaml
optimizer:
    nn_lr: 0.004
    embedding_lr: 0.0
```

Learning rates of 0.0 will mask the respective weights during training steps.
Here, we will freeze the descriptor, reinitialize the scaling and shifting parameters and reduce the learning rate of all other components.

We can now fine tune the model by running
`apax train config.yaml`
