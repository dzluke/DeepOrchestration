# From https://github.com/CarmineCella/nmf_sources
from pathlib import Path

from librosa.core import fft
import soundfile as sf
import numpy as np
import librosa as lr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .kl_nmf import kl_nmf

FFT_SIZE = 2048
HOPLEN = 512
COMPONENTS = 32 # NMF components
SOURCES = 4 # K in KMeans
DIMENSIONS = 20 # PCA reduction (0 for no reduction)
PHI_ITER = 0 # if 0 uses original phases otherwise reconstruct by iteration

# Given a complex signal, find its polar coordinates
def car2pol(sig):
    im, re = sig.imag, sig.real
    amp = np.sqrt(im**2 + re**2)
    angle = np.arctan2(im, re)
    return amp, angle

# Given polar coordinates, reconstruct the complex signal
def pol2car(amp, angle):
    re = amp * np.cos(angle)
    im = amp * np.sin(angle)
    return re + im*1j

def get_sources (mix, nsources=4, components=32, dimensions=10, fft_size=4096,
                 hoplen=512, phi_iter=0):
    # data preparation
    sources = np.zeros((nsources, len (mix)))
    specgram = lr.stft(mix, n_fft=fft_size, hop_length=hoplen);
    A, Phi = car2pol(specgram)

    # NMF decomposition
    [W, H] = kl_nmf(A, components)

    # masking
    Wsum = np.sum(W, axis=1);
    masks = np.zeros(W.shape);
    for i in range(0, W.shape[1]):
        masks[:, i] = W[:, i] / Wsum;
        W[:, i] = W[:, i] * masks[:, i]

    # clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(H)
    if dimensions != 0:
        pca = PCA(n_components=dimensions)
        scaled_features = pca.fit_transform(scaled_features)
    clusters = KMeans(n_clusters=nsources).fit (scaled_features)

    # assignment and reconstruction
    y_out = np.random.rand(len(mix))
    for s in range(0, nsources):
        src = np.zeros(len(mix))

        for k in range(0, components):
            comp = np.zeros(len(mix))
            if clusters.labels_[k] == s:
                A1 = np.outer(W[:,k], H[k,:])
                if phi_iter == 0 :
                    specgramx = pol2car(A1, Phi)
                    out = lr.istft(specgramx, hop_length=hoplen)
                else:
                    out = np.random.rand(len(mix))
                    for _ in range (0, phi_iter):
                        out_spec = lr.stft(out, n_fft=fft_size, hop_length=hoplen)
                        specgramx = A1 * np.exp(complex(0.0, 1.0) * np.angle(out_spec))
                        out = lr.istft(specgramx, hop_length=hoplen)
                ml = min(len(src), len(out))
                r = range(0, ml)
                src[r] = src[r] + out[r]
        sources[s] = src
    return W, H, clusters.labels_, sources

def separate(in_path, out_path, verbose=False):
    out_path = Path(out_path)
    mix, sr = sf.read(in_path)
    if verbose:
        print('total samples: ', len (mix))
        print('sources      : ', SOURCES)
        print('components   : ', COMPONENTS)
    W, H, labels, sources = get_sources(mix, nsources=SOURCES, components=COMPONENTS,
        dimensions=DIMENSIONS, fft_size=FFT_SIZE, hoplen=HOPLEN, phi_iter=PHI_ITER)
    if verbose:
        print('W            : ', W.shape)
        print('H            : ', H.shape)
        print('labels       : ', labels)
    out_path.mkdir(parents=True, exist_ok=True)
    for s in range (SOURCES):
        sf.write(out_path.joinpath('source_' + str (s) + '.wav'), sources[s], sr)
