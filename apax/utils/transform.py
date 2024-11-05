

def make_energy_only_model(energy_properties_model):
    energy_model = lambda *args: energy_properties_model(*args)[0]
    return energy_model