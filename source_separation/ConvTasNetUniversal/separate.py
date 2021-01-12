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

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from . import inference
from .train import data_io


def decode_wav(wav):
  audio_bytes = tf.read_file(wav)
  waveform, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1,
		                              desired_samples=-1)
  waveform = tf.reshape(waveform, (1, -1))
  return waveform

def separate(ckpt, mtgph, input_path, output_path):
  model = inference.SeparationModel(ckpt,
                                    mtgph)
  if not os.path.exists(input_path):
    raise Exception("Wrong input file {}".format(input_path))
  file_list = [input_path]
  from scipy.io.wavfile import read as read_wav
  from scipy.io.wavfile import write as write_wav
  sr,f = read_wav(file_list[0])

  with model.graph.as_default():
    dataset = data_io.wavs_to_dataset(file_list, batch_size=1,
                                      num_samples=len(f),
                                      repeat=False)
    # Strip batch and mic dimensions.
    dataset['receiver_audio'] = dataset['receiver_audio'][0, 0]
    dataset['source_images'] = dataset['source_images'][0, :, 0]

  waveforms = model.sess.run(dataset)
  separated_waveforms = model.separate(waveforms['receiver_audio'])
  # print(separated_waveforms)
  # print(separated_waveforms.shape)
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  for i in range(separated_waveforms.shape[0]):
    write_wav(output_path + '/sub_target{}.wav'.format(i), sr, separated_waveforms[i,:])

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

  separate(args.checkpoint_path, args.metagraph_path, args.input_path, args.output_path)

tf.enable_eager_execution()

if __name__ == '__main__':
  main()
