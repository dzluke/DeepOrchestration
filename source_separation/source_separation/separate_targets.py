import os
from pathlib import Path
import warnings
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import soundfile as sf
from librosa.util import find_files
from tqdm import tqdm

from utils import remove_extension

config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def separate(method, target_fname, outdir, n_sources):
    assert method in config["separation"]["methods"].split(", ")
    sample_rate = config["audio"].getint("sample_rate")

    # Separate target waveform into n_sources separated_waveforms
    if method == "Demucs":
        warnings.warn("This is deprecated", DeprecationWarning)
        import demucs.separate as demucs
        demucs.separate(target_fname, outdir)
    elif method == "NMF":
        import nmf.separate as nmf
        separated_waveforms = nmf.separate(target_fname, sample_rate, n_sources)
    elif method == "OpenUnmix":
        warnings.warn("This is deprecated", DeprecationWarning)
        import open_unmix.separate as open_unmix
        open_unmix.separate(target_fname, outdir)
    elif method == "TDCN++":
        import tdcn.separate as tdcn
        tdcn_model_path = config["paths"]["tdcn_model"]
        ckpt = os.path.join(tdcn_model_path, "baseline_model")
        mtgph = os.path.join(tdcn_model_path, "baseline_inference.meta")
        separated_waveforms = tdcn.separate(
            target_fname,
            outdir,
            ckpt,
            mtgph,
            target_sr=config["audio"].getint("sample_rate"),
        )
        if n_sources !=4:
            # Select the best ones!
            sep_powers = np.mean(separated_waveforms**2, axis=-1)
            # Get the n_sources waveforms that have the more power
            max_indices = sep_powers.argsort()[-n_sources:][::-1]
            separated_waveforms = separated_waveforms[max_indices]

    # Save the waveforms
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    for i in range(n_sources):
        sf.write(
            out_path.joinpath("source_" + str(i) + ".wav"),
            separated_waveforms[i],
            sample_rate,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ds_path", type=str)
    parser.add_argument("--n-sources", type=int, default=4)
    args, _ = parser.parse_known_args()

    separation_methods = config["separation"]["methods"].split(", ")
    num_subtargets = config["separation"].getint("num_subtargets)")
    if num_subtargets != 4 and (
        "Demucs" in separation_methods or "OpenUnmix" in separation_methods
    ):
        warnings.warn(
            "Music source separation methods cannot be used with a number of subtargets other than 4 (Demucs, Open-Unmix)."
        )
        if "Demucs" in separation_methods:
            separation_methods.remove("Demucs")
        if "OpenUnmix" in separation_methods:
            separation_methods.remove("OpenUnmix")

    targets = find_files(
        os.path.join(args.ds_path, "targets", f"{args.n_sources}sources")
    )

    for target_fname in tqdm(targets):
        for method in separation_methods:
            outdir = os.path.join(
                args.ds_path,
                "separated",
                f"{args.n_sources}sources",
                method,
                remove_extension(os.path.basename(target_fname)),
            )
            separate(method, target_fname, outdir, args.n_sources)
