import numpy as np
import soundfile as sf
import librosa as lr
import cmath
import math
from sklearn.decomposition import NMF
from pathlib import Path

def separate(audio_path,
             output_path,
             model_name='nmf',
             n=4,
             blocksize=4096,
             hoplen=1024,
             maxiter=10000):
    y, sr = lr.load(audio_path)
    #y, sr = sf.read('coque.wav')
    #y = np.asfortranarray(y.T[0])
    D = lr.stft(y, n_fft=blocksize, hop_length=hoplen)
    fftf = lr.core.fft_frequencies(sr=sr, n_fft=blocksize)
    S, _ = lr.magphase(D)

    # Calculate NMF decomposition
    model = NMF(n_components=n, init='random', random_state=0, max_iter=maxiter)
    W = model.fit_transform(S)
    H = model.components_

    # Output individual features & full NMF appx.
    output = {}
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        S = np.array(np.matrix(W.T[i]).T * np.matrix(H[i]))
        # Recover phase data, algorithm idea from https://dsp.stackexchange.com/questions/9877/reconstruction-of-audio-signal-from-spectrogram
        y_out = np.random.rand(len(y))
        for _ in range(3):
            D_out = lr.stft(y_out, n_fft=blocksize, hop_length=hoplen)
            D_out = S * np.exp(complex(0.0, 1.0) * np.angle(D_out))
            y_out = lr.istft(D_out, hop_length=hoplen)
        output[i] = y_out
        sf.write(output_path + "/feat{}.wav".format(i+1), y_out, sr)
    return output
