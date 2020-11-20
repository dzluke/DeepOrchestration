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

CUMULATIVE_DENSE_BLOCK_OUTPUT = True
VERBOSE = True

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
B,C,T,Fr = 16,1,50,100
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
    l = None
    r = None
    torch.cuda.empty_cache()
print("Computation time for 1 batch : {}".format(sum(x for x in dt)/len(dt)))

'''
dt = [1.2316265106201172,
 0.9567806720733643,
 1.0095221996307373,
 1.069530725479126,
 0.9404206275939941,
 0.9825644493103027,
 0.943284273147583,
 0.9432389736175537,
 1.3145031929016113,
 0.9436233043670654,
 0.9450030326843262,
 1.050917387008667,
 1.0431499481201172,
 0.9439489841461182,
 0.9484851360321045,
 0.9486229419708252,
 1.2562696933746338,
 0.9581849575042725,
 0.9514966011047363,
 0.9680142402648926,
 1.2484300136566162,
 0.9650261402130127,
 1.4151840209960938,
 1.220715045928955,
 0.9658191204071045,
 1.3943183422088623,
 1.1896495819091797,
 0.9695940017700195,
 1.3323705196380615,
 1.1225497722625732,
 1.0147838592529297,
 1.1678142547607422,
 0.9433512687683105,
 0.9390149116516113,
 0.9517979621887207,
 0.9657433032989502,
 1.0631542205810547,
 1.022209644317627,
 0.943354606628418,
 0.9545111656188965,
 0.9557633399963379,
 1.1303820610046387,
 1.083404302597046,
 0.9487698078155518,
 1.2408020496368408,
 0.9502379894256592,
 1.342728614807129,
 1.294419765472412,
 0.9474756717681885,
 1.2969038486480713,
 1.4546465873718262,
 1.1912527084350586,
 1.2074124813079834,
 0.9899642467498779,
 1.2196605205535889,
 0.9506998062133789,
 0.9908442497253418,
 1.3188652992248535,
 0.9528849124908447,
 1.279259204864502,
 1.1786959171295166,
 0.9713773727416992,
 0.9389374256134033,
 1.1426258087158203,
 1.0642056465148926,
 1.186218500137329,
 0.9981245994567871,
 1.3997631072998047,
 1.1189627647399902,
 1.0405290126800537,
 1.3648836612701416,
 1.0662879943847656,
 1.022313117980957,
 0.9391660690307617,
 0.9391911029815674,
 0.9620890617370605,
 1.0668361186981201,
 1.2674365043640137,
 1.1856820583343506,
 1.0293972492218018,
 1.3031399250030518,
 1.138730764389038,
 1.0588297843933105,
 1.385486364364624,
 1.036177396774292,
 1.0382273197174072,
 0.9424538612365723,
 0.9643757343292236,
 0.9614405632019043,
 1.3078789710998535,
 0.953054666519165,
 1.0540289878845215,
 1.0253117084503174,
 0.959517240524292,
 0.9616384506225586,
 0.9470815658569336,
 1.306283712387085,
 0.9532680511474609,
 1.0296728610992432,
 1.0025596618652344,
 0.946570634841919,
 0.9633052349090576,
 0.9601249694824219,
 0.9890496730804443,
 1.2968549728393555,
 1.1876068115234375,
 1.036175012588501,
 1.2370402812957764,
 0.9851698875427246,
 0.9555177688598633,
 0.9621031284332275,
 0.9731965065002441,
 1.2637629508972168,
 0.9514296054840088,
 0.9356498718261719,
 1.0348570346832275,
 0.9655821323394775,
 0.962233304977417,
 0.94547438621521,
 1.2785277366638184,
 0.9627499580383301,
 0.9711298942565918,
 1.121352195739746,
 1.0197389125823975,
 0.9379544258117676,
 0.9784066677093506,
 1.2541882991790771,
 1.007089614868164,
 1.2838566303253174,
 1.3060882091522217,
 1.0136072635650635,
 1.070906639099121,
 0.9605684280395508,
 0.9453601837158203,
 0.9861323833465576,
 1.510213851928711,
 0.9875192642211914,
 1.4164435863494873,
 1.1610324382781982,
 0.9994421005249023,
 1.4396517276763916,
 1.1621308326721191,
 1.1063604354858398,
 1.174323320388794,
 1.2133169174194336,
 1.292255163192749,
 1.363750696182251,
 1.3108129501342773,
 1.220296859741211,
 1.212690830230713,
 1.1392543315887451,
 1.358217477798462,
 1.2522783279418945,
 1.1573336124420166,
 1.2053053379058838,
 1.2482810020446777,
 1.0923717021942139,
 1.2053039073944092,
 1.2252938747406006,
 1.121354103088379,
 1.2222967147827148,
 1.1823744773864746,
 1.0563910007476807,
 0.9795022010803223,
 1.2768452167510986,
 0.9586150646209717,
 1.1140315532684326,
 1.0724542140960693,
 0.9564633369445801,
 1.1903486251831055,
 0.9793884754180908,
 0.9757850170135498,
 0.978316068649292,
 1.257188320159912,
 0.9507043361663818,
 1.318108320236206,
 1.2687256336212158,
 0.9509432315826416,
 1.2969093322753906,
 1.3379454612731934,
 0.9543306827545166,
 1.2835991382598877,
 1.431541919708252,
 1.1475369930267334,
 0.9532411098480225,
 1.165522813796997,
 0.9890444278717041,
 0.9760403633117676,
 1.3116679191589355,
 1.1355538368225098,
 1.0144577026367188,
 0.9459948539733887,
 1.2022440433502197,
 0.9548490047454834,
 0.94663405418396,
 0.9532356262207031,
 1.1211786270141602,
 1.0017344951629639,
 1.0443623065948486,
 1.2576415538787842]'''