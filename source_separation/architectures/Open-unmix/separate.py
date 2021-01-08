# -*- coding: utf-8 -*-

import torch
import tqdm
import numpy as np
import soundfile as sf
import norbert
from pathlib import Path
import io
from contextlib import redirect_stderr
import json
import scipy.signal

import utils
import open_unmix as model


def load_model(target, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        # model path does not exist, use hubconf model
        try:
            # disable progress bar
            err = io.StringIO()
            with redirect_stderr(err):
                return torch.hub.load(
                    'sigsep/open-unmix-pytorch',
                    model_name,
                    target=target,
                    device=device,
                    pretrained=True
                )
            print(err.getvalue())
        except AttributeError:
            raise NameError('Model does not exist on torchhub')
            # assume model is a path to a local model_name direcotry
    else:
        # load model from disk
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(
            target_model_path,
            map_location=device
        )

        max_bin = utils.bandwidth_to_max_bin(
            state['sample_rate'],
            results['args']['nfft'],
            results['args']['bandwidth']
        )

        unmix = model.OpenUnmix(
            n_fft=results['args']['nfft'],
            n_hop=results['args']['nhop'],
            nb_channels=results['args']['nb_channels'],
            hidden_size=results['args']['hidden_size'],
            max_bin=max_bin
        )

        unmix.load_state_dict(state)
        unmix.stft.center = True
        unmix.eval()
        unmix.to(device)
        return unmix


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(input_path,
             output_path,
             model_name='umxhq',
             targets=('vocals', 'drums', 'bass', 'other'),
             device='cpu',
             softmask=False,
             residual_model=False,
             alpha=1.0,
             niter=1):
    """
    generate 4 subtargets
    """

    # ENTREE : input path
    # SORTIE : OUTPUT PATH NOM DE DOSSIER ECRIT LES SUBTARGETS EN .WAV DANS CE PATH

    audio, sample_rate = sf.read(input_path)
    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    V = []

    for j, target in enumerate(tqdm.tqdm(targets)):
        unmix_target = load_model(
            target=target,
            model_name=model_name,
            device=device
        )
        Vj = unmix_target(audio_torch).cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    X = unmix_target.stft(audio_torch).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(targets) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += (['residual'] if len(targets) > 1
                         else ['accompaniment'])

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=unmix_target.stft.n_fft,
            n_hopsize=unmix_target.stft.n_hop
        )
        estimates[name] = audio_hat.T

        # write wav file in output_path
        subtarget_path = Path(output_path).joinpath(name + '.wav')
        sf.write(subtarget_path, estimates[name], sample_rate)
    return estimates
