# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate separated audio from a DCASE 2020 task 4 separation model."""

import argparse
import os

import librosa
import tensorflow.compat.v1 as tf

from .train import data_io
from . import inference

tf.enable_eager_execution()


def decode_wav(wav):
    audio_bytes = tf.read_file(wav)
    waveform, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1,
                                      desired_samples=-1)
    waveform = tf.reshape(waveform, (1, -1))
    return waveform


def separate(input_path, output_path, ckpt, mtgph, target_sr=44100):
    model = inference.SeparationModel(ckpt, mtgph)
    if not os.path.exists(input_path):
        raise Exception("Wrong input file {}".format(input_path))
    from scipy.io.wavfile import write as write_wav
    # sr, f = read_wav(file_list[0])
    target_waveform, _ = librosa.load(input_path, sr=16000)

    separated_waveforms = model.separate(target_waveform)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(separated_waveforms.shape[0]):
        resampled_waveform = librosa.resample(separated_waveforms[i, :],
                                              orig_sr=16000,
                                              target_sr=target_sr)
        write_wav(output_path + '/separated{}.wav'.format(i),
                  target_sr, resampled_waveform)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a source separation model.')
    parser.add_argument(
        '-cp', '--checkpoint_path', help='Path for model checkpoint files.',
        required=True)
    parser.add_argument(
        '-mp', '--metagraph_path', help='Path for inference metagraph.',
        required=True)
    parser.add_argument(
        '-ip', '--input_path', help='Path for input file',
        required=True)
    parser.add_argument(
        '-op', '--output_path', help='Path of resulting csv file.',
        required=True)
    args = parser.parse_args()

    separate(args.checkpoint_path, args.metagraph_path,
             args.input_path, args.output_path)


if __name__ == '__main__':
    main()