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
separation_functions = {
    "demucs": demucs.separate,
    "nmf": nmf.separate,
    "open_unmix": open_unmix.separate,
    "tdcn": tdcn.separate
}


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

    for t in targets:
        for method in separation_methods:
            if num_subtargets == 4:
                separation_functions[method](t, args.outdir)
            else:
                #TODO: Add support for num_subtargets != 4
                separation_functions[method]
