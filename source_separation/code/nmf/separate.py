# From https://github.com/CarmineCella/nmf_sources
from pathlib import Path

import librosa as lr
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from nmf.kl_nmf import kl_nmf

FFT_SIZE = 2048
HOPLEN = 256
# COMPONENTS = 32 # NMF components
# SOURCES = 4 # K in KMeans

# Given a complex signal, find its polar coordinates
def car2pol(sig):
    im, re = sig.imag, sig.real
    amp = np.sqrt(im**2 + re**2)
    angle = np.arctan2(im, re)
    return amp, angle


def get_sources (mix, n_sources=4, components=32, fft_size=4096, hoplen=512):
    # data preparation
    sources = np.zeros((n_sources, len (mix)))
    specgram = lr.stft(mix, n_fft=fft_size, hop_length=hoplen);
    A, Phi = car2pol(specgram)

    # NMF decomposition
    [W, H] = kl_nmf(A, components)

    # masking
    masks = np.zeros((A.shape[0],A.shape[1], components), dtype=complex)
    comps = np.zeros(masks.shape, dtype=complex)
    for i in range(0, components):
        masks[:, :, i] = np.outer(W[:, i], H[i, :]) / np.dot(W, H)
        masks[np.isnan(masks)] = 0
        comps[:, :, i] =  masks[:, :, i] * specgram

    # clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(H)
    clusters = KMeans (n_clusters=n_sources).fit(scaled_features)

    # assignment and reconstruction
    for s in range(0, n_sources):
        comp = np.zeros(specgram.shape)
        for k in range (0, components):
            if clusters.labels_[k] == s:
                comp = comp + comps[:, :, k]

        src = lr.istft(comp, hop_length=hoplen)
        ml = min(len(src), len(mix))
        r = range(0, ml)
        sources[s, r] = src[r]
    return W, H, clusters.labels_, sources


def separate(in_path, out_path, n_sources=4, components=32):
    out_path = Path(out_path)
    mix, sr = sf.read(in_path)
    W, H, labels, sources = get_sources(mix,
                                        n_sources=n_sources,
                                        components=components,
                                        fft_size=FFT_SIZE,
                                        hoplen=HOPLEN)
    out_path.mkdir(parents=True, exist_ok=True)
    for s in range (n_sources):
        sf.write(out_path.joinpath('source_' + str (s) + '.wav'), sources[s], sr)
