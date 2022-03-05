import os
import warnings
from argparse import ArgumentParser
from configparser import ConfigParser

from librosa.util import find_files

import demucs.separate as demucs
import nmf.separate as nmf
import open_unmix.separate as open_unmix
import tdcn.separate as tdcn

config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def separate(method, target_fname, outdir, n_sources, **kwargs):
    assert method in config['separation']['methods'].split(", ")

    if method == "demucs":
        demucs.separate(target_fname, outdir)
    elif method == "nmf":
        nmf.separate(target_fname, outdir, n_sources)
    elif method == "open_unmix":
        open_unmix.separate(target_fname, outdir)
    elif method == "tdcn":
        ckpt = os.path.join(kwargs['tdcn_model_path'], "baseline_model")
        mtgph = os.path.join(kwargs['tdcn_model_path'], "baseline_inference.meta")
        tdcn.separate(target_fname, outdir, ckpt, mtgph)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("targets_path",
                        type=str,
                        help="")
    parser.add_argument("outdir")

    args, _ = parser.parse_known_args()

    separation_methods = config['separation']['methods'].split(", ")
    num_subtargets = config['separation'].getint('num_subtargets)')
    if num_subtargets !=4 and ("demucs" in separation_methods
                               or "open_unmix" in separation_methods):
        warnings.warn("Music source separation methods cannot be used "
                    "with a number of subtargets other than 4 (Demucs, "
                    "Open-Unmix).")
        if "demucs" in separation_methods:
            separation_methods.remove("demucs")
        if "open_unmix" in separation_methods:
            separation_methods.remove("open_unmix")

    targets = find_files(args.targets_path)

    for target_fname in targets:
        for method in separation_methods:
            separate(method, target_fname, args.outdir, args.n_sources,
                     tdcn_model_path=config['paths']['tdcn_model'])
