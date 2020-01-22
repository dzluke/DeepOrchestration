import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pydub
from augment import wav_augment
time = 4

# y1, sr1 = librosa.load("./OrchDB/Acc-ord-A#2-mf.wav", sr=None, duration=time)
# y2, sr2 = librosa.load("./OrchDB/BClBb-flatt-A#4-f.wav",
#                        sr=None, duration=time)
# y3, sr3 = librosa.load("./OrchDB/ASax-ord-G5-ff.wav", sr=None, duration=time)

# y = (y1+y2+y3)/3
# librosa.output.write_wav('tmp.wav', y, sr1)

# y1, sr1 = librosa.load(
#     "./OrchDB/Cb-art-harm-D#4-mf-4c.wav", sr=None, duration=time)
# feature1 = librosa.feature.melspectrogram(y1, sr1)

# y2, sr2 = librosa.load(
#     "./OrchDB/Cb-art-harm-C4-mf-4c.wav", sr=None, duration=time)
# feature2 = librosa.feature.melspectrogram(y2, sr2)

# print(feature1)
# print(feature2)
# print(np.linalg.norm(feature1-feature2))

y1, sr1 = librosa.load(
    "./other/Ob-ord-A#3-ff.wav", sr=None, duration=time)
print(y1.shape)

y2, sr2 = librosa.load(
    "./other/Vn-ord-G6-pp-1c.wav", sr=None, duration=time)
y2 = wav_augment(y2, sr2)
print(y2.shape)

# y = (y1+y2)/2
# print()
# librosa.output.write_wav('mix.wav', y=y, sr=sr1)

# s = pydub.AudioSegment.from_wav("Va-ord-G5-ff-1c.wav").set_sample_width(2)
# s.export('s.wav', format='wav')
