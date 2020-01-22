import librosa
import numpy as np
import copy
import random
from sklearn.utils import shuffle


def set_length(new_y, y):
    if len(new_y) < len(y):
        new_y = librosa.util.fix_length(new_y, len(y))
    elif len(new_y) > len(y):
        new_y = new_y[0:len(y)]
    return new_y


def random_stretch(y, sr, prob=1., min_stretch=0.9, max_stretch=1.1):
    if np.random.uniform(0, 1) < prob:
        factor = np.random.uniform(min_stretch, max_stretch)
        new_y = librosa.effects.time_stretch(y, factor)
        return set_length(new_y, y)
    else:
        return y


def random_pitch(y, sr, prob=1., min_shift=-0.3, max_shift=0.3, bins_per_octave=12):
    if np.random.uniform(0, 1) < prob:
        steps = np.random.uniform(min_shift, max_shift)
        new_y = librosa.effects.pitch_shift(
            y, sr, n_steps=steps, bins_per_octave=bins_per_octave)
        return set_length(new_y, y)
    else:
        return y


def wav_augment(y, sr):
    functions = shuffle([random_pitch, random_stretch])
    for fn in functions:
        y = fn(y, sr=sr)
    return y


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = copy.deepcopy(spec)
    num_mel_channels = cloned.shape[0]
    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)
        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f):
            return cloned
        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero):
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()
    return cloned


def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = copy.deepcopy(spec)
    len_spectro = cloned.shape[1]
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)
        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t):
            return cloned
        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[:, t_zero:mask_end] = cloned.mean()
    return cloned


def spec_augment(spec, prob=1.):
    # Adapted from https://github.com/zcaceres/spec_augment
    if np.random.uniform(0, 1) < prob:
        return freq_mask(time_mask(spec))
    else:
        return time_mask(freq_mask(spec))
