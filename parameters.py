import pickle

path = "./TinySOL"
MEL_HOP_LENGTH = 2048
RATE = 44100
N_FFT = 2048
N_MELS = 128
TIME_LENGTH = 4
FEATURE_TYPE = 'mel'
PITCH_REGROUP = True

N = 2
nb_samples = 100000
rdm_granularity = 10
nb_pitch_range = 8
instr_filter = None
batch_size = 16
model_type = 'cnn'
nb_epoch = 80
model_path = './model'
model_resume_path = 'epoch_199.pth'
resume_model = False
train_proportion = 0.8

coeff_freq_shift_data_augment = 0.005 # For data augmentation, proportional change to sampling rate
delay_offset_avg = 0.005 # Average of offset applied to delay filters (ie the position of the first impulse)
delay_period_avg = 0.002 # Average of the period of the impulses for delaying
delay_feedback_avg = 0.5 # Average of the feedback factor of delay impulses
prop_zero_row = 0.01
prop_zero_col = 0.01
noise_kernel_var = 0.0001




def load_parameters(path):
    global MEL_HOP_LENGTH
    global RATE
    global N_FFT
    global N_MELS
    global TIME_LENGTH
    global FEATURE_TYPE
    global PITCH_REGROUP
    
    global N
    global nb_samples
    global rdm_granularity
    global nb_pitch_range
    global instr_filter
    global batch_size
    global model_type
    global nb_epoch
    global model_path
    global model_resume_path
    global resume_model
    global train_proportion
    
    f = open(path + '/params.pkl', 'rb')
    params = pickle.load(f)
    f.close()
    
    MEL_HOP_LENGTH = params['MEL_HOP_LENGTH']
    RATE = params['RATE']
    N_FFT = params['N_FFT']
    N_MELS = params['N_MELS']
    TIME_LENGTH = params['TIME_LENGTH']
    FEATURE_TYPE = params['FEATURE_TYPE']
    PITCH_REGROUP = params['PITCH_REGROUP']
    
    N = params['N']
    nb_samples = params['nb_samples']
    rdm_granularity = params['rdm_granularity']
    nb_pitch_range = params['nb_pitch_range']
    instr_filter = params['instr_filter']
    batch_size = params['batch_size']
    model_type = params['model_type']
    nb_epoch = params['nb_epoch']
    model_path = params['model_path']
    model_resume_path = params['model_resume_path']
    resume_model = params['resume_model']
    train_proportion = params['train_proportion']

def save_parameters(path):
    params={}
    params['MEL_HOP_LENGTH'] = MEL_HOP_LENGTH
    params['RATE'] = RATE
    params['N_FFT'] = N_FFT
    params['N_MELS'] = N_MELS
    params['TIME_LENGTH'] = TIME_LENGTH
    params['FEATURE_TYPE'] = FEATURE_TYPE
    params['PITCH_REGROUP'] = PITCH_REGROUP
    
    params['N'] = N
    params['nb_samples'] = nb_samples
    params['rdm_granularity'] = rdm_granularity
    params['nb_pitch_range'] = nb_pitch_range
    params['instr_filter'] = instr_filter
    params['batch_size'] = batch_size
    params['model_type'] = model_type
    params['nb_epoch'] = nb_epoch
    params['model_path'] = model_path
    params['model_resume_path'] = model_resume_path
    params['resume_model'] = resume_model
    params['train_proportion'] = train_proportion
    
    f = open(path + '/params.pkl', 'wb')
    pickle.dump(params, f)
    f.close()


# Change every time we use one sample on generation
# Alter it before combination
# Frequency shift (modify sample rate)

# Can be done dynamically
# Before feeding the batch tensor, apply a filtering in frequency 
# => Zeroing rows or columns of each matrix
# => Convolution with small gaussian nois kernel