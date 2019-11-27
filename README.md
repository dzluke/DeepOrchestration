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

#### Experiment on all files (including dynamic) recognition
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

### The third try id to use SVM
- SVM with linear kernel shows better result which reaches a result of 71.99 % on test set
- maybe try random forrest 


