import os
import argparse

from tqdm import tqdm

from pipeline import generate_separation_function

# TEMP_OUTPUT_PATH = "./TEMP"
# targets_directory = input("Targets_directory: ")

def create_separated_subtargets(targets_directory, output_path):
    for target_name in tqdm(os.listdir(targets_directory)):
        # for model in ["TDCN++", "TDCN", "OpenUnmix", "Demucs"]:
        for model in ["Demucs", "OpenUnmix", "TDCN++", "TDCN"]:
            print(target_name, model)
            target = targets_directory + '/' + target_name
            l = generate_separation_function(model, num_subtargets=4)
            l[0](target, num_subtargets=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('targets_directory', type=str,
                        help="Path to the folder containing targets.")
    parser.add_argument('output_path', type=str,
                        help="Path to the output folder.")
    args, _ = parser.parse_known_args()
    create_separated_subtargets(args.targets_directory, args.output_path)
