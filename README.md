# DeepOrchestration

> It is a cool project using deep learning to help ochestration. Fighting!


### The first try is to implement cnn
#### Experiment on instrument regconition 
- take N(currently 2) instruments from TinySOL and then combine 
- take melspectrogram features (128*128, if the combined pieces are small, just add zeros)
- design a multi-label model and use one-hot encoding and cross entropy loss function 
- after 200 epochs training on gpu, accuracy results of every instruments shown as follow

![inst](./specific_acc_inst.png) 

![inst_all](./acc_inst.png)

#### Experiment on one-class-per-file recognition
- Take a similar way to deal with the data
- After 200 epochs training on gpu, results of every instruments shown

![all](./specific_acc.png)
![data](./best.png)

- Total acc and single acc are shown, which reaches 86.625 %.

![all_all](./acc.png)

### The second try is to use resnet
- the way to deal with data is similar
- take a classic model of resnet, but remove one maxpool layer to fit data 
- The training is slower and the current model does not perform as good as cnn
- Try different paras which is more suitable to solve our problem
- A comparison of total accuracy between cnn and resnet shown

![comp](./acc_compare.png)

### Other Architectures
#### BottleNeck
Best acc: 84.875%

![bn_acc](./bottleneck-300/bottleneck_acc.png)

for every instrument:

![bn_inst](./bottleneck-300/bottleneck_inst.png)

Loss:

![bn_loss](./bottleneck-300/bottleneck_loss.png)

#### BottleNeck+Residual
Best acc: 80.037%

![bn_res_acc](./bottleneck-300/res/bottleneck_acc.png)

for every instrument:

![bn_res_inst](./bottleneck-300/res/res_inst.png)

Loss:

![bn_res_loss](./bottleneck-300/res/bottleneck_loss.png)

#### CNN+LSTM
Best acc: 80.688%

![cl_acc](./lstm-300/slstm/lstm_acc.png)

for every instrument:

![cl_inst](./lstm-300/slstm/lstm_inst.png)

Loss:

![cl_loss](./lstm-300/slstm/lstm_loss.png)

#### CNN+BiLSTM
Best acc: 67.513%

![cbl_acc](./lstm-300/bilstm/lstm_acc.png)

for every instrument:

![cbl_inst](./lstm-300/bilstm/lstm_inst.png)

Loss:

![cbl_loss](./lstm-300/bilstm/lstm_loss.png)

#### CNN+LSTM+Residual
Best acc: 86.125%

![clr_acc](./lstm-300/lstm+res/lstm_acc.png)

for every instrument:

![clr_inst](./lstm-300/lstm+res/lstm_inst.png)

Loss:

![clr_loss](./lstm-300/lstm+res/lstm_loss.png)

### Baseline(Only instrument recognition)
#### Random forrest (file mixture)
Random forrest(with 60 estimators) performs better than SVM 

![rf](./rf.png)

#### SVM
- Grid Search para with rbf kernel from C: 1 to 10000, gama: 0.0001 - 1, and get a best result of 71.12 %
- SVM with linear kernel shows better result which reaches a result of 71.99 % on test set
