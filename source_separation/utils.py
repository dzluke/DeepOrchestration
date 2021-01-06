import torch
from torch.nn import functional as F

import random
import tqdm

import gzip
import warnings


def center_trim(tensor, reference):
    """
    From https://github.com/facebookresearch/demucs
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def apply_model(model, mix, shifts=None, split=False, progress=False):
    """
    From https://github.com/facebookresearch/demucs
    Apply model to a given mixture.

    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress (bool): if True, show a progress bar (requires split=True)
    """
    channels, length = mix.size()
    device = mix.device
    if split:
        out = torch.zeros(4, channels, length, device=device)
        shift = 44_100 * 10
        offsets = range(0, length, shift)
        scale = 10
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
        for offset in offsets:
            chunk = mix[..., offset:offset + shift]
            chunk_out = apply_model(model, chunk, shifts=shifts)
            out[..., offset:offset + shift] = chunk_out
            offset += shift
        return out
    elif shifts:
        max_shift = 22050
        mix = F.pad(mix, (max_shift, max_shift))
        offsets = list(range(max_shift))
        random.shuffle(offsets)
        out = 0
        for offset in offsets[:shifts]:
            shifted = mix[..., offset:offset + length + max_shift]
            shifted_out = apply_model(model, shifted)
            out += shifted_out[..., max_shift - offset:max_shift - offset + length]
        out /= shifts
        return out
    else:
        valid_length = model.valid_length(length)
        delta = valid_length - length
        padded = F.pad(mix, (delta // 2, delta - delta // 2))
        with torch.no_grad():
            out = model(padded.unsqueeze(0))[0]
        return center_trim(out, mix)


def load_model(path):
    """
    From https://github.com/facebookresearch/demucs
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_from = path
        if str(path).endswith(".gz"):
            load_from = gzip.open(path, "rb")
        klass, args, kwargs, state = torch.load(load_from, 'cpu')
    model = klass(*args, **kwargs)
    model.load_state_dict(state)
    return model


def save_model(model, path):
    """
    From https://github.com/facebookresearch/demucs
    """
    args, kwargs = model._init_args_kwargs
    klass = model.__class__
    state = {k: p.data.to('cpu') for k, p in model.state_dict().items()}
    save_to = path
    if str(path).endswith(".gz"):
        save_to = gzip.open(path, "wb", compresslevel=5)
    torch.save((klass, args, kwargs, state), save_to)
