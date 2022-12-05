import optax


def map_nested_fn(fn):
    """Recursively apply `fn` to the key-value pairs of a nested dict"""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def get_schedule(lr, transition_begin, transition_steps):
    lr_schedule = optax.linear_schedule(
        init_value=lr,
        end_value=1e-6,
        transition_begin=transition_begin,
        transition_steps=transition_steps,
    )
    return lr_schedule


def get_opt(
    transition_begin,
    transition_steps,
    emb_lr=0.02,
    nn_lr=0.03,
    scale_lr=0.001,
    shift_lr=0.05,
    opt_name="adam",
    opt_kwargs={},
):
    opt = getattr(optax, opt_name)

    emb_schedule = get_schedule(emb_lr, transition_begin, transition_steps)
    nn_schedule = get_schedule(nn_lr, transition_begin, transition_steps)
    scale_schedule = get_schedule(scale_lr, transition_begin, transition_steps)
    shift_schedule = get_schedule(shift_lr, transition_begin, transition_steps)

    label_fn = map_nested_fn(lambda k, _: k)
    tx = optax.multi_transform(
        {
            "w": opt(emb_schedule, **opt_kwargs),
            "b": opt(emb_schedule, **opt_kwargs),
            "atomic_type_embedding": opt(nn_schedule, **opt_kwargs),
            "scale": opt(scale_schedule, **opt_kwargs),
            "shift": opt(shift_schedule, **opt_kwargs),
        },
        label_fn,
    )
    return tx
