def make_energy_only_model(energy_properties_model):
    energy_model = lambda *args, **kwargs: energy_properties_model(*args, **kwargs)[0]
    return energy_model
