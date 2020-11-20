import torch
# import encoding ## pip install torch-encoding . For synchnonized Batch norm in pytorch 1.0.0
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch import Tensor

import time

'''
    MMDenseLSTM is comprised of several instances of MDenseLSTM, one for each frequency band + one for the whole frequency range.
    
    The flow of data is as follows:
        - First convolution to generate [first_channel] channels
        - Succession of [scale+1] dense blocksused to downsample the feature matrices. Each dense block has [l] layers.
            * Layer number [i] takes as input [k0+i*k] channels and outputs [k] channels.
            * Layer [i] has a first Norm-ReLU-1x1Conv layer to reduce input features to [bn_size*k] channels
            * After bottleneck, another layer of Norm-ReLU-3x3Conv is applied to obtain [k] channels.
        - 
'''
global DLC
global DBC
global LLC
global MBC
DLC = 0
DBC = 0
LLC = 0
MBC = 0

CUMULATIVE_DENSE_BLOCK_OUTPUT = False
VERBOSE = False

class MMDenseLSTMParams:
    def __init__(self, init_nb_channels, band_splits, band_growth_rates, scales, input_shape, final_conv_channels, final_dense_block_layers, final_dense_block_growth_rate, drop_rate = 0.1, bn_size=4):
        self.N = len(band_splits) + 2
        self.drop_rate = drop_rate
        self.init_nb_channels = init_nb_channels
        self.bn_size = bn_size
        self.final_conv_channels = final_conv_channels
        self.final_dense_block_layers = final_dense_block_layers
        self.final_dense_block_growth_rate = final_dense_block_growth_rate
        if len(band_growth_rates) != self.N:
            raise Exception("band_growth_rates must contain {} elements : 1 for each band and 1 for the full range".format(self.N))
        if len(scales) != self.N:
            raise Exception("scales must contain {} elements : 1 for each band and 1 for the full range".format(self.N))
        self.layer_shapes = [[input_shape]]
        ext_band_splits = [0] + band_splits + [input_shape[1]]
        for i in range(len(ext_band_splits)-1):
            self.layer_shapes.append([(input_shape[0], ext_band_splits[i+1]-ext_band_splits[i])])
        self.scales = scales # Determines the number of dense layers for each sub-model (2*scale+1)
        for i in range(self.N):
            for j in range(scales[i]):
                s = self.layer_shapes[i][-1]
                self.layer_shapes[i].append((s[0]//2, s[1]//2))
            for j in range(scales[i]):
                s = self.layer_shapes[i][scales[i]-j-1]
                self.layer_shapes[i].append(s)
        self.band_growth_rates = band_growth_rates # growth_rate parameter [k] for each band and the full range
        self.band_splits = band_splits
        
        
        
        print("MMDenseLSTM parameters :")
        print("    Input is audio with {} channels".format(init_nb_channels))
        print("    Dense layers bottleneck size : {}".format(self.bn_size))
        print("    Drop rate : {}".format(self.drop_rate))
        print("    {} bands separated at {}".format(len(band_splits)+1, band_splits))
        print("    Full range network scale : {} ({} downsample dense blocks and {} upsample dense blocks)".format(scales[0],scales[0]+1,scales[0]))
        print("    Full range input feature matrices shape across blocks {}".format(self.layer_shapes[0]))
        for i in range(1,self.N):
            print("    Band {} network scale : {} ({} downsample dense blocks and {} upsample dense blocks)".format(i, scales[i],scales[i]+1,scales[i]))
            print("    Band {} input feature matrices shape across blocks {}".format(i, self.layer_shapes[i]))
        print("    Final conv block : {} output channels".format(final_conv_channels))
        print("    Final dense block : {} layers with growth rate {}".format(final_dense_block_layers, final_dense_block_growth_rate))
        
        '''
        Each element of the following lists are associated to one band, 0 being the full range and subsequent indexes being the respective frequency band.
        '''
        self.downsample_dense_layers = [[] for i in range(self.N)] # List of size (scale+1) containing the number of dense layers for each dense block for downsampling
        self.downsample_lstm = [[] for i in range(self.N)]
        self.upsample_dense_layers = [[] for i in range(self.N)] # List of size (scale) containing the number of dense layers for each dense block for upsampling
        self.upsample_lstm = [[] for i in range(self.N)]
        
    def set_downsample_dense_layers(self, i, l):
        if isinstance(l, int) or len(l) == self.scales[i]+1:
            if isinstance(l, int):
                self.downsample_dense_layers[i] = [l] * (self.scales[i]+1)
            else:
                self.downsample_dense_layers[i] = l
            
            if i == 0:
                print("    Full range downsample dense layers : {}".format(self.downsample_dense_layers[i]))
            else:
                print("    Band {} downsample dense layers : {}".format(i, self.downsample_dense_layers[i]))
        else:
            raise Exception("List of downsample layers must be int or list of size {}".format(self.scales[i]+1))
            
    def set_upsample_dense_layers(self, i, l):
        if isinstance(l, int) or len(l) == self.scales[i]:
            if isinstance(l, int):
                self.upsample_dense_layers[i] = [l] * self.scales[i]
            else:
                self.upsample_dense_layers[i] = l
            if i == 0:
                print("    Full range upsample dense layers : {}".format(self.upsample_dense_layers[i]))
            else:
                print("    Band {} upsample dense layers : {}".format(i, self.upsample_dense_layers[i]))
        else:
            raise Exception("List of upsample layers must be int or list of size {}".format(self.scales[i]))
            
    def set_downsample_lstm(self, i, l):
        if len(l) == self.scales[i]+1:
            self.downsample_lstm[i] = l
            if i == 0:
                print("    Full range downsample lstm units : {}".format(l))
            else:
                print("    Band {} downsample lstm units : {}".format(i, l))
        else:
            raise Exception("List of downsample lstm must be of size {}".format(self.scales[i]+1))
            
    def set_upsample_lstm(self, i, l):
        if len(l) == self.scales[i]:
            self.upsample_lstm[i] = l
            if i == 0:
                print("    Full range upsample lstm units : {}".format(l))
            else:
                print("    Band {} upsample lstm units : {}".format(i, l))
        else:
            raise Exception("List of upsample lstm must be of size {}".format(self.scales[i]))
        

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        global DLC
        if VERBOSE:
            print("    Dense layer call {}".format(DLC))
        DLC += 1
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
    
class _DenseLSTMBlock(nn.Module):
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, lstm_params, memory_efficient=False):
        '''
            Parameters
            
                num_layers :            Number of internal layers
                num_input_features :    Number of channels in the input tensor
                bn_size :               Bottleneck size for all dense layers
                growth_rate :           Growth rate for all dense layers
                drop_rate :             Drop rate for all dense layers
                lstm_params :           If not None, (l,f) where l corresponds to the number of hidden layers
                                        inside LSTM block and f the size of the frequency dimension of the input.
                                        The LSTM block is placed after the output of the last dense layer
                                        
            Forward
            
                Features go through each dense layer and the LSTM block if it exists.
                Each input of a dense layer is the concatenation of the initial feature tensor and
                all the consecutive outputs (each of size k) of previous dense layers.
                If CUMULATIVE_DENSE_BLOCK_OUTPUT is True, the final output is [init_features, x1, x2, ..., xn, lstm] where xi is the output (size k)
                of dense layer i and lstm is the output of the lstm unit if it is set.
                The number of channels goes from I (initial number of channels) to I + k*l or I + k*l + 1
                depending on whether the LSTM block is set or not.
                If CUMULATIVE_DENSE_BLOCK_OUTPUT is set to False, only the output of the last layer is returned.
                The number of channels goes from I to k (or k+1 if LSTM is set)
        '''
        super(_DenseLSTMBlock, self).__init__()
        self.layers = nn.ModuleDict()
        self.num_input_features = num_input_features
        self.lstm = None
        if lstm_params is not None:
            self.lstm = _LSTMBlock(num_input_features + num_layers * growth_rate, lstm_params[1], lstm_params[0])
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.layers['denselayer%d' % (i + 1)] = layer

    def forward(self, init_features):
        global DBC
        if VERBOSE:
            print("  Dense block call {}".format(DBC))
        DBC += 1
        features = [init_features]
        for name, layer in self.layers.items():
            new_features = layer(features)
            features.append(new_features)
        
        if self.lstm is not None:
            features = torch.cat(features, 1)
            if CUMULATIVE_DENSE_BLOCK_OUTPUT:
                features = torch.cat([features, self.lstm(features)], 1) # Take all the outputs and lstm output (size I+l*k+1)
            else:
                features = torch.cat([new_features, self.lstm(features)], 1) # Only take the last output and lstm output (size k+1)
        else:
            if CUMULATIVE_DENSE_BLOCK_OUTPUT:
                features = torch.cat(features, 1) # Take all the outputs (size I+l*k)
            else:
                features = new_features # Only take the last output (size k)
        
        return features
        
class _LSTMBlock(nn.Module):
    def __init__(self,input_layers, freq_bins, hidden_layers):
        super(_LSTMBlock, self).__init__()
        self.input_layers = input_layers
        self.freq_bins = freq_bins
        self.hidden_layers = hidden_layers
        self.add_module('flattening', nn.Conv2d(input_layers, 1, kernel_size=1, stride=1,
                                           bias=False))
        self.add_module('lstm', nn.LSTM(input_size=freq_bins, hidden_size=hidden_layers, batch_first=True, bidirectional = True))
        self.add_module('linear', nn.Linear(2*hidden_layers, freq_bins))
        
    def forward(self, features):
        global LLC
        if VERBOSE:
            print("    LSTM layer call {}".format(LLC))
        LLC += 1
        y = self.flattening(features)
        y = y.squeeze()
        
        y,_ = self.lstm(y)
        
        y = self.linear(y)
        
        return y.unsqueeze(1)

    
class _MDenseLSTM_STEM(nn.Module):
    def __init__(self, scale, growth_rate, shapes, ds_layers, us_layers, ds_lstm, us_lstm, init_nb_channels, final_channels, first_kernel = (3,3), drop_rate = 0.1,bn_size=4):
        super(_MDenseLSTM_STEM,self).__init__()
        
        first_channel = 32
        self.scale = scale
        self.shapes = shapes
        
        self.first_conv = nn.Conv2d(init_nb_channels,first_channel,first_kernel,padding=1)
        self.downsample_layer = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.upsample_layers = nn.ModuleList()
        self.dense_blocks = nn.ModuleList()
        self.channels = [first_channel]
        ## [_,d1,...,ds,ds+1,u1,...,us]
        for i in range(scale+1):
            k = growth_rate
            l = ds_layers[i]
            if ds_lstm[i] == 0:
                self.dense_blocks.append(_DenseLSTMBlock( 
                    l, self.channels[-1], bn_size, k, drop_rate, None))
                if CUMULATIVE_DENSE_BLOCK_OUTPUT:
                    self.channels.append(self.channels[-1]+k*l)
                else:
                    self.channels.append(k)
            else:
                self.dense_blocks.append(_DenseLSTMBlock( 
                    l, self.channels[-1], bn_size, k, drop_rate, (ds_lstm[i], shapes[i][1])))
                if CUMULATIVE_DENSE_BLOCK_OUTPUT:
                    self.channels.append(self.channels[-1]+k*l+1)
                else:
                    self.channels.append(k+1)
        
        for i in range(scale):
            k = growth_rate
            l = us_layers[i]
            self.upsample_layers.append(nn.ConvTranspose2d(self.channels[-1],self.channels[-1], kernel_size=2, stride=2))
            self.channels.append(self.channels[-1]+self.channels[scale-i])
            if us_lstm[i] == 0:
                self.dense_blocks.append(_DenseLSTMBlock( 
                    l, self.channels[-1], bn_size, k, drop_rate, None))
                if CUMULATIVE_DENSE_BLOCK_OUTPUT:
                    self.channels.append(self.channels[-1]+k*l)
                else:
                    self.channels.append(k)
            else:
                self.dense_blocks.append(_DenseLSTMBlock( 
                    l, self.channels[-1], bn_size, k, drop_rate, (us_lstm[i], shapes[scale+1+i][1])))
                if CUMULATIVE_DENSE_BLOCK_OUTPUT:
                    self.channels.append(self.channels[-1]+k*l+1)
                else:
                    self.channels.append(k+1)
        
        self.final_conv = nn.Conv2d(self.channels[-1], final_channels, kernel_size=1, stride=1)
        self.channels.append(final_channels)
    
    def _pad(self,x,target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x,(padding_2,0,padding_1,0),'replicate')
        else:
            return x
    
    def forward(self,input):
        global MBC
        if VERBOSE:
            print("MMDense block call {}".format(MBC))
        MBC += 1
        ## stem
        output = self.first_conv(input)
        dense_outputs = []
        
        ## downsample way
        for i in range(self.scale):
            if VERBOSE:
                print("Shape : {}; Expected : {}".format(output.shape, self.shapes[i]))
            output = self.dense_blocks[i](output)
            dense_outputs.append(output)
            output = self.downsample_layer(output) ## downsample

        ## upsample way
        if VERBOSE:
            print("Shape : {}; Expected : {}".format(output.shape, self.shapes[self.scale]))
        output = self.dense_blocks[self.scale](output)
            
        for i in range(self.scale):
            output = self.upsample_layers[i](output)
            output = self._pad(output,dense_outputs[-(i+1)])
            output = torch.cat([output,dense_outputs[-(i+1)]],dim = 1)
            if VERBOSE:
                print("Shape : {}; Expected : {}".format(output.shape, self.shapes[i+self.scale+1]))
            output = self.dense_blocks[self.scale+1+i](output)
        output = self._pad(output,input)
        output = self.final_conv(output)
        if VERBOSE:
            print("########## Final output shape : {}".format(output.shape))
        return output
    
class MMDenseLSTM(nn.Module):
    def __init__(self, params):
        super(MMDenseLSTM,self).__init__()
        
        self.params = params
        self.band_dense_networks = []
        for i in range(params.N):
            d = _MDenseLSTM_STEM(params.scales[i], 
                                 params.band_growth_rates[i], 
                                 params.layer_shapes[i], 
                                 params.downsample_dense_layers[i], 
                                 params.upsample_dense_layers[i], 
                                 params.downsample_lstm[i], 
                                 params.upsample_lstm[i], 
                                 params.init_nb_channels, 
                                 params.final_conv_channels,
                                 first_kernel = (3,3), 
                                 drop_rate = params.drop_rate,
                                 bn_size=params.bn_size)
            self.band_dense_networks.append(d)
            print("Band {} sequence of channels : {}".format(i, d.channels))
        
        if CUMULATIVE_DENSE_BLOCK_OUTPUT:
            last_channel = 2*params.final_conv_channels + params.final_dense_block_growth_rate*params.final_dense_block_layers
        else:
            last_channel = params.final_dense_block_growth_rate
        self.out = nn.Sequential(
                _DenseLSTMBlock(params.final_dense_block_layers,
                                2*params.final_conv_channels,
                                params.bn_size,
                                params.final_dense_block_growth_rate,
                                params.drop_rate,
                                None),
                nn.Conv2d(last_channel,params.init_nb_channels,1))
        
    def to(self, device):
        '''
            Override the to device function in order to port all the band networks
            to the specified device.
        '''
        super(MMDenseLSTM,self).to(device)
        for bdn in self.band_dense_networks:
            bdn.to(device)
        return self
        
        
    def forward(self,input):
        '''
        Input is a tensor containing spectrograms and whose dimensions are [Batch, Channels, Time, Frequency]
        '''
#         print(input.shape)
        
        sub_band_output = []
        for i in range(1,self.params.N):
            if i == 1:
                sub_band_input = input[:,:,:,:self.params.band_splits[0]]
            elif i == self.params.N-1:
                sub_band_input = input[:,:,:,self.params.band_splits[-1]:]
            else:
                sub_band_input = input[:,:,:,self.params.band_splits[i-2]:self.params.band_splits[i-1]]
            sub_band_output.append(self.band_dense_networks[i](sub_band_input))
        output = torch.cat(sub_band_output,3)##Frequency 방향
        full_output = self.band_dense_networks[0](input)
        output = torch.cat([output,full_output],1) ## Channel 방향
        output = self.out(output)
        
        return output

# Split at 4100Hz and 11000Hz
B,C,T,Fr = 15,1,75,128
params = MMDenseLSTMParams(1, [43,86],[7,14,4,2],[4,3,3,2], (T,Fr), 32, 3, 12)

params.set_downsample_dense_layers(0,[3,3,4,5,5])
params.set_upsample_dense_layers(0, [5,4,3,3])
params.set_downsample_lstm(0, [0,0,0,128,0])
params.set_upsample_lstm(0, [0,0,128,0])

params.set_downsample_dense_layers(1,5)
params.set_upsample_dense_layers(1,5)
params.set_downsample_lstm(1, [0,0,0,128])
params.set_upsample_lstm(1, [0,128,0])

params.set_downsample_dense_layers(2,4)
params.set_upsample_dense_layers(2,4)
params.set_downsample_lstm(2, [0,0,0,32])
params.set_upsample_lstm(2, [0,0,0])

params.set_downsample_dense_layers(3,1)
params.set_upsample_dense_layers(3,1)
params.set_downsample_lstm(3, [0,0,8])
params.set_upsample_lstm(3, [0,0])

m = MMDenseLSTM(params)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
m = m.to(device)

dt = []
for i in range(200):
    print("Computation n°{}".format(i+1))
    l = torch.tensor(np.random.randn(B,C,T,Fr)).float().cuda()
    l = l.to(device)
    t = time.time()
    r=m(l)
    dt.append(time.time()-t)
print("Computation time for 1 batch : {}".format(sum(x for x in dt)/len(dt)))