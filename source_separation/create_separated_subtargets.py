import os
from tqdm import tqdm

from pipeline import generate_separation_function


TEMP_OUTPUT_PATH = "./TEMP"
targets_directory = input("Targets_directory: ")

for target_name in tqdm(os.listdir(targets_directory)):
    for model in ["TDCNN++", "TDCNN", "OpenUnmix", "Demucs"]:
        print(target_name, model)
        target = targets_directory + '/' + target_name
        l = generate_separation_function(model, num_sub_targets=2)
        l[0](target, 2)
