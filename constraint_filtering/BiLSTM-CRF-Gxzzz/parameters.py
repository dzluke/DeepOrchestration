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
        self.nb_samples = 1000
        self.rdm_granularity = 10
        self.nb_pitch_range = 8
        self.instr_filter = ['Hn', 'Ob', 'Vn', 'Va', 'Vc', 'Fl', 'Tbn', 'Bn', 'TpC', 'ClBb'][:10]
        self.batch_size = 16
        self.model_type = 'resnet'
        self.nb_epoch = 100
        self.train_proportion = 0.8
        
        self.coeff_freq_shift_data_augment = 0.008 # For data augmentation, proportional change to sampling rate
        self.delay_offset_avg = 0.005 # Average of offset applied to delay filters (ie the position of the first impulse)
        self.delay_period_avg = 0.002 # Average of the period of the impulses for delaying
        self.delay_feedback_avg = 0.5 # Average of the feedback factor of delay impulses
        self.prop_zero_row = 0.01
        self.prop_zero_col = 0.01
        self.noise_kernel_var = 0.0001
        
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
