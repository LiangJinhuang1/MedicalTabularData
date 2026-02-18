import torch


def encode_with_entropy(encoder, x):
    encoder_output = encoder(x)
    if isinstance(encoder_output, tuple):
        z = encoder_output[0]
        entropy = encoder_output[1] if len(encoder_output) > 1 else None
    else:
        z = encoder_output
        entropy = None
    if isinstance(z, tuple):
        z = z[0]
    if not isinstance(z, torch.Tensor):
        raise TypeError(
            f"Expected z to be a Tensor, but got {type(z)}. encoder_output type: {type(encoder_output)}"
        )
    return z, entropy
