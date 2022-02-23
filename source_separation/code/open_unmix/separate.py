from pathlib import Path
import torch
import torchaudio
import json
import numpy as np


from open_unmix import utils
from open_unmix import predict
from open_unmix import data

import argparse


def separate(input, outdir=None, targets=None, model='umxhq', start=0.0,
             duration=None, niter=1, residual=None, ext='.wav',
             wiener_win_len=300, filterbank='torch', aggregate=None,
             no_cuda=False, audio_backend='sox_io'):
    if audio_backend != "stempeg":
        torchaudio.set_audio_backend(audio_backend)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using ", device)
    # parsing the output dict
    aggregate_dict = None if aggregate is None else json.loads(aggregate)

    # create separator only once to reduce model loading
    # when using multiple files
    separator = utils.load_separator(
        model_str_or_path=model,
        targets=targets,
        niter=niter,
        residual=residual,
        wiener_win_len=wiener_win_len,
        device=device,
        pretrained=True,
        filterbank=filterbank,
    )

    separator.freeze()
    separator.to(device)

    if audio_backend == "stempeg":
        try:
            import stempeg
        except ImportError:
            raise RuntimeError("Please install pip package `stempeg`")

    # If we're dealing with a single file.
    if type(input) != list:
        input = [input]

    # loop over the files
    for input_file in input:
        if audio_backend == "stempeg":
            audio, rate = stempeg.read_stems(
                input_file,
                start=start,
                duration=duration,
                sample_rate=separator.sample_rate,
                dtype=np.float32,
            )
            audio = torch.tensor(audio)
        else:
            audio, rate = data.load_audio(input_file, start=start, dur=duration)
        estimates = predict.separate(
            audio=audio,
            rate=rate,
            aggregate_dict=aggregate_dict,
            separator=separator,
            device=device,
        )
        if not outdir:
            model_path = Path(model)
            if not model_path.exists():
                outdir = Path(Path(input_file).stem + "_" + model)
            else:
                outdir = Path(Path(input_file).stem + "_" + model_path.stem)
        else:
            outdir = Path(outdir) #/ Path(input_file).stem
        outdir.mkdir(exist_ok=True, parents=True)

        # write out estimates
        if audio_backend == "stempeg":
            target_path = str(outdir / Path("target").with_suffix(ext))
            # convert torch dict to numpy dict
            estimates_numpy = {}
            for target, estimate in estimates.items():
                estimates_numpy[target] = torch.squeeze(estimate).detach().cpu().numpy().T

            stempeg.write_stems(
                target_path,
                estimates_numpy,
                sample_rate=separator.sample_rate,
                writer=stempeg.FilesWriter(multiprocess=True, output_sample_rate=rate),
            )
        else:
            for target, estimate in estimates.items():
                target_path = str(outdir / Path(target).with_suffix(ext))
                torchaudio.save(
                    target_path,
                    torch.squeeze(estimate).to("cpu"),
                    sample_rate=separator.sample_rate,
                )

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="UMX Inference",
#         add_help=True,
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )

#     parser.add_argument("input", type=str, nargs="+", help="List of paths to wav/flac files.")

#     parser.add_argument(
#         "--model",
#         default="umxhq",
#         type=str,
#         help="path to mode base directory of pretrained models",
#     )

#     parser.add_argument(
#         "--targets",
#         nargs="+",
#         type=str,
#         help="provide targets to be processed. \
#               If none, all available targets will be computed",
#     )

#     parser.add_argument(
#         "--outdir",
#         type=str,
#         help="Results path where audio evaluation results are stored",
#     )

#     parser.add_argument(
#         "--ext",
#         type=str,
#         default=".wav",
#         help="Output extension which sets the audio format",
#     )

#     parser.add_argument("--start", type=float, default=0.0, help="Audio chunk start in seconds")

#     parser.add_argument(
#         "--duration",
#         type=float,
#         help="Audio chunk duration in seconds, negative values load full track",
#     )

#     parser.add_argument(
#         "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
#     )

#     parser.add_argument(
#         "--audio-backend",
#         type=str,
#         default="sox_io",
#         help="Set torchaudio backend "
#         "(`sox_io`, `sox`, `soundfile` or `stempeg`), defaults to `sox_io`",
#     )

#     parser.add_argument(
#         "--niter",
#         type=int,
#         default=1,
#         help="number of iterations for refining results.",
#     )

#     parser.add_argument(
#         "--wiener-win-len",
#         type=int,
#         default=300,
#         help="Number of frames on which to apply filtering independently",
#     )

#     parser.add_argument(
#         "--residual",
#         type=str,
#         default=None,
#         help="if provided, build a source with given name"
#         "for the mix minus all estimated targets",
#     )

#     parser.add_argument(
#         "--aggregate",
#         type=str,
#         default=None,
#         help="if provided, must be a string containing a valid expression for "
#         "a dictionary, with keys as output target names, and values "
#         "a list of targets that are used to build it. For instance: "
#         '\'{"vocals":["vocals"], "accompaniment":["drums",'
#         '"bass","other"]}\'',
#     )

#     parser.add_argument(
#         "--filterbank",
#         type=str,
#         default="torch",
    #     help="filterbank implementation method. "
    #     "Supported: `['torch', 'asteroid']`. `torch` is ~30% faster"
    #     "compared to `asteroid` on large FFT sizes such as 4096. However"
    #     "asteroids stft can be exported to onnx, which makes is practical"
    #     "for deployment.",
    # )
    # args = parser.parse_args()
