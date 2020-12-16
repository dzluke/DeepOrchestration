import pickle

class SimParams:
    def __init__(self):
        self.path = "./TinySOL"
        self.MEL_HOP_LENGTH = 2048
        self.RATE = 44100
        self.N_FFT = 2048
        self.N_MELS = 128
        self.TIME_LENGTH = 4
        self.FEATURE_TYPE = 'mel'
        
        self.N = 2
        self.nb_samples = 500
        self.rdm_granularity = 10
        self.nb_pitch_range = 8
        self.instr_filter = ['Hn', 'Ob', 'Vn', 'Va', 'Vc', 'Fl', 'Tbn', 'Bn', 'TpC', 'ClBb'][:self.N]
        self.batch_size = 16
        self.model_type = 'resnet'
        self.nb_epoch = 2
        self.train_proportion = 0.8
        
        self.coeff_freq_shift_data_augment = 0.008 # For data augmentation, proportional change to sampling rate
        self.delay_offset_avg = 0.005 # Average of offset applied to delay filters (ie the position of the first impulse)
        self.delay_period_avg = 0.002 # Average of the period of the impulses for delaying
        self.delay_feedback_avg = 0.5 # Average of the feedback factor of delay impulses
        self.prop_zero_row = 0.01
        self.prop_zero_col = 0.01
        self.noise_kernel_var = 0.0001

        # nested dictionary, usage: sample2index[instrument][pitch] = index in label vector
        self.sample2index = {}
        # index i in index2sample gives a tuple (instr, pitch) that corresponds
        # to the sample in index i in the label vector
        # this is used to create constraints
        self.index2sample = []

        self.harmonic_threshold = 5e-7
        
    def load_parameters(self, path):
        f = open(path + '/params.pkl', 'rb')
        p = pickle.load(f)
        f.close()
        
        for attr in p.__dict__:
            setattr(self, attr, getattr(p, attr))
    
    def save_parameters(self, path):
        f = open(path + '/params.pkl', 'wb')
        pickle.dump(self, f)
        f.close()

GLOBAL_PARAMS = SimParams()
