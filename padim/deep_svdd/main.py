from .mlp import MLPNet, MLPNetAutoencoder


def build_network(*args, **kwargs):
    """Builds the neural network."""
    net = MLPNet(*args, **kwargs)
    return net


def build_autoencoder(*args, **kwargs):
    """Builds the corresponding autoencoder network."""
    ae_net = MLPNetAutoencoder(*args, **kwargs)
    return ae_net
