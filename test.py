import torch
import numpy as np
import librosa
import librosa.display
import os
import random
import soundfile as sf

from model import OrchMatchNet
from OrchDataset import RawDatabase
from train import getPosNMax
from parameters import GLOBAL_PARAMS, SimParams

""" 
To run this file:
- You need a folder of targets to be orchestrated, and set 'target_path' to point to this
- You need to create a folder to store the orchestrated solutions, and set 'solutions_path'
to be the path to this folder
- You need a trained model saved as a .pth file, and set 'state_path' to point to this

- Set 'tinysol_path' to point to your TinySOL database
- Set model_type, instr_filter, and n

"""


# path to TinySOL data
tinysol_path = './TinySOL'

# path to target samples
target_path = './target_samples'

# number of samples to be used in solution
n = 10

# instruments to be used (all instruments will be used)
instr_filter = ['Hn', 'Ob', 'Vn', 'Va',
                'Vc', 'Fl', 'Tbn', 'Bn', 'TpC', 'ClBb']

# if true, doesn't orchestrate targets but instead creates combinations of TinySOL samples to input to the model
sanity_check = False


def test(model, state_path, data, targets):
    device = torch.device('cpu')
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    model.eval()

    outputs = model(data)
    outputs = outputs.detach().cpu().clone().numpy()

    if sanity_check:
        accuracy = evaluate(outputs, labels)
        print(accuracy)

    # text file to write solutions to
    f = open(solutions_path + '/orchestration_results.txt', 'w+')

    for i in range(len(outputs)):
        output = outputs[i]
        target = targets[i]
        # get indices of top n probabilities
        indices = getPosNMax(output, n)
        # turn indices into [instr, pitch]
        classes = get_classes(indices)
        # get top n probabiltiies
        probs = [output[i] for i in indices]
        # turn probabilities into dynamic markings (pp, mf, ff)
        dynamics = prob_to_dynamic(probs)
        # combine into (instr, pitch, dynamic)
        for j in range(len(classes)):
            classes[j].append(dynamics[j])
        
        # turn (instr, pitch, dynamic) into list of actual TinySOL sample paths
        sample_paths = find_sample_paths(classes)
        target['classes'] = classes

        # combine samples
        mixed_file, sr = combine(sample_paths, target)
        # write wav
        file_name = solutions_path + '/orchestrated_' + target['name'] + '.wav'
        sf.write(file_name, mixed_file, sr)

        if sanity_check:
            target['distance'] = 0
        else:
            target['distance'] = compute_distance(target, mixed_file)
        
        # write to text file
        f.write('Target: {}; Distance: {:,.8f}\nSamples used: {}\n\n'.format(target['name'], target['distance'], target['classes']))

    # compute avg distance
    sum = 0
    for target in targets:
        sum += target['distance']
    sum /= len(targets)
    f.write('Average distance: {:,.8f}'.format(sum))
    f.close()

        
def find_sample_paths(classes):
    '''
    classes is a list where each element is a list like this: [instr, pitch, dynamic]
    '''
    samples = []
    for c in classes:
        instr, pitch, dynamic = c
        instr_samples = rdb.db[instr]
        for lst in instr_samples:
            for sample in lst:
                if sample['instrument'] == instr and sample['pitch_name'] == pitch and sample['nuance'] == dynamic:
                    samples.append(sample['path'])
                    break
    return samples

'''
converts a list of probabilities to a list of dynamics
0 - 0.33 -> pp
0.34 - 0.66 -> mf
0.67 - 1 -> ff
'''
def prob_to_dynamic(probs):
    dynamics = []
    pp = 0.33
    mf = 0.66
    for prob in probs:
        if prob > mf:
            dynamics.append('ff')
        elif prob > pp:
            dynamics.append('mf')
        else:
            dynamics.append('pp')
    return dynamics    

# given a list of indices, return the corresponding [instrument, pitch] in lab_class
def get_classes(indices):
    classes = [None for i in indices]
    for instrument, pitch_dict in lab_class.items():
        for pitch, class_number in pitch_dict.items():
            if class_number in indices:
                index = indices.index(class_number)
                classes[index] = [instrument, pitch]
    assert len(indices) == len(classes)
    return classes

# load a sample at the given path and return the melspectrogram  and duration of the sample
def load_sample(path):
    y, sr = librosa.load(path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    mel_hop_length = sr * duration / (87 - 1) # based on training data size
    mel_hop_length = int(mel_hop_length)
   
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=mel_hop_length)
    return torch.tensor(np.array([mel_spec])), duration


# load in a folder of data and return a list of melspectrograms
def load_data(folder_path):
    data = []
    targets = []
    for entry in os.listdir(folder_path):
        if entry.endswith('.wav'):
            full_path = os.path.join(folder_path, entry)
            mel_spec, duration = load_sample(full_path)
            data.append(mel_spec)
            targets.append({'name': entry[:-4], 'duration': duration, 'path': full_path})
    print("Loaded {} target samples".format(len(data)))

    return torch.stack(data), targets


def combine(soundlist, target):
    mixed_file = np.zeros((1, 1))
    for sound in soundlist:
        sfile, sr = librosa.load(sound, sr=None)
        mixed_file = mix(mixed_file, sfile)
    mixed_file = mixed_file/len(soundlist)

    # trim to target length
    trim_index = int(target['duration']*sr)
    mixed_file = mixed_file[:trim_index]

    return mixed_file, sr


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def compute_distance(target, solution):
    target, _ = librosa.load(target['path'], sr=None)

    # if the target is longer than the solution, must trim the target
    if (target.size > solution.size):
        target = target[:solution.size]
        N = next_power_of_2(target.size)
    else:
        N = next_power_of_2(solution.size)

    target_fft = np.abs(np.fft.rfft(target, N))
    solution_fft = np.abs(np.fft.rfft(solution, N))

    target_fft /= (np.max(target_fft) * N)
    solution_fft /= (np.max(solution_fft) * N)

    lambda_1 = 0.5
    lambda_2 = 10

    sum_1 = 0
    sum_2 = 0
    for i in range(target_fft.size):
        k_target = target_fft[i]
        k_solution = solution_fft[i]
        if k_target >= k_solution:
            sum_1 += k_target - k_solution
        else:
            sum_2 += abs(k_target - k_solution)
    distance = lambda_1 * sum_1 + lambda_2 * sum_2
    return distance * 1000


def mix(fa, fb):
    diff = len(fa) - len(fb)

    if diff >= 0:
        add = np.zeros((1, diff), dtype=np.float32)
        fb = np.append(fb, add)
    else:
        add = np.zeros((1, -diff), dtype=np.float32)
        fa = np.append(fa, add)

    return fa+fb


def make_fake_targets(num_classes):
    num_targets = 10
    data = []
    targets = []
    labels = []
    for i in range(num_targets):
        samples = random.sample(range(num_classes), n)
        labels.append(samples)
        samples = get_classes(samples)
        dynamics = [random.choice(['pp', 'mf', 'ff']) for _ in range(n)]
        for j in range(len(samples)):
            samples[j].append(dynamics[j])
        paths = find_sample_paths(samples)
        mixed_file, _ = combine(paths, {'duration':4})
        mel_spec = librosa.feature.melspectrogram(y=mixed_file,sr=44100,hop_length=GLOBAL_PARAMS.MEL_HOP_LENGTH)[:128,:87]
        data.append(torch.tensor([mel_spec]))
        
        name = ''
        for l in samples:
            name += l[0] + l[1] + '_'
        target = {'name':name[:-1], 'duration':4}
        targets.append(target)

    encoded_labels = []
    for label in labels:
        l = np.zeros(num_classes, dtype=np.float32)
        for x in label:
            l[x] = 1.0
        encoded_labels.append(l)
    encoded_labels = np.array(encoded_labels)

    print('Created {} fake data targets'.format(num_targets))

    return torch.stack(data), targets, encoded_labels


if __name__ == "__main__":

    # a dictionary that maps a model type (cnn, resnet, or orchidea) to a list of targets
    all_targets = {}

    rdb = RawDatabase(tinysol_path, 10, instr_filter)

    for model_type in ['cnn', 'resnet']:

        print('--------------------------')
        print('MODEL TYPE:', model_type)
        print('NUMBER OF INSTRUMENTS:', len(instr_filter))
        print('N:', n)
        print('--------------------------')

        # path to store solutions as .wav
        solutions_path = './orchestrated_targets/{}_n={}'.format(model_type, n)

        # The argument must be the folder where the params.pkl file is
        GLOBAL_PARAMS.load_parameters(
            './orchestrated_targets/params_{}'.format(model_type))

        # path to a trained version of the model
        if model_type == 'cnn':
            state_path = './orchestrated_targets/new_models/CNN_epoch39.pth'
        else:
            state_path = './orchestrated_targets/new_models/ResNet_epoch19.pth'

        lab_class = GLOBAL_PARAMS.lab_class

        # calculate num classes
        num_classes = 0
        for k in rdb.db:
            a = set()
            for l in rdb.db[k]:
                for e in l:
                    a.add(e['pitch_name'])
            for p in rdb.pr:
                if p in a:
                    num_classes += 1

        
        def class_encoder(list_samp):
            label = [0 for i in range(num_classes)]
            for s in list_samp:
                label[lab_class[s['instrument']][s['pitch_name']]] = 1
            return np.array(label).astype(np.float32)

        def evaluate(preds, labels):
            if preds.shape != labels.shape:
                print("[Error]: size difference")
            # compute the label-based accuracy
            result = {}

            result['acc'] = np.sum(preds*labels)/max(1.0, np.sum(labels))
            pitch_acc = {}
            for i in lab_class:
                l = [lab_class[i][x] for x in lab_class[i]]
                f = np.zeros(preds.shape, dtype=np.float32)
                f[:, min(l):max(l)+1] = 1.0
                f = labels*f
                pitch_acc[i] = np.sum(preds*f)/max(1.0, np.sum(f))
            result['pitch_acc'] = pitch_acc

            return result
        
        features_shape = [128, 87] # the dim of the data used to train the network
        
        model = OrchMatchNet(num_classes, model_type, features_shape)
        
        if sanity_check:
            data, targets, labels = make_fake_targets(num_classes)
        else:
            data, targets = load_data(target_path)

        test(model, state_path, data, targets)

        all_targets[model_type] = targets

        print('Finished', model_type)

    print('Beginning OrchIdea solutions')
    orchidea_solutions_path = './orchidea_solutions'
    data, orchidea_targets = load_data(target_path)
    targets_to_remove = []
    total = 0
    f = open(orchidea_solutions_path + '/orchestration_results.txt', 'w+')
    for target in orchidea_targets:
        # print("Target Name:", target['name'])
        name = target['name'] + '.wav.solution'
        solution_path = os.path.join(orchidea_solutions_path, name, 'connection.wav')
        # print('Soln Path:', solution_path)
        try:
            solution, _ = librosa.load(solution_path, sr=None)
        except:
            targets_to_remove.append(target)
            continue
        target['distance'] = compute_distance(target, solution)
        # print('Distance', target['distance'])
        total += target['distance']
        f.write('Target: {}; Distance: {:,.8f}\n\n'.format(target['name'], target['distance']))
    f.write('Average Distance: {:,.8f}'.format(total/len(orchidea_targets)))
    f.close()

    for t in targets_to_remove:
        orchidea_targets.remove(t)

    # compute ratios
    for orch_target in orchidea_targets:
        name = orch_target['name']
        for model_type in ['cnn', 'resnet']:
            for t in all_targets[model_type]:
                if t['name'] == name:
                    ratio = orch_target['distance'] / t['distance']
                    t['ratio'] = ratio
       
    # write ratios
    for model_type in ['cnn', 'resnet']:
        solutions_path = './orchestrated_targets/{}_n={}'.format(model_type, n)
        f = open(solutions_path + '/orchestration_results.txt', 'a')

        f.write('\n\n\nRatios:\n\n')

        for target in all_targets[model_type]:
            if 'ratio' in target:
                f.write('Target: {}; Ratio: {:,.8f}\n\n'.format(target['name'], target['ratio']))

    print('Finished writing ratios.')



    





    
