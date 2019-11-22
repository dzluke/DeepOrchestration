# DeepOrchestration

> It is a cool project using deep learning to help ochestration. Fighting!


### The first try is to implement cnn
#### Experiment on instrument regconition 
- take N(currently 2) instruments from TinySOL and then combine 
- take melspectrogram features (128*128, if the combined pieces are small, just add zeros)
- design a multi-label model and use one-hot encoding and cross entropy loss function 
- after 200 epochs training on cpu, accuracy results of every instruments shown as follow

![inst](./specific_acc_inst.png) 

- It shows the best result at epoch 171
- maybe it is still the local optimal solution
- Total(both prediction are right) acc and single(only one prediction is right) acc are shown:

![inst_all](./acc_inst.png)

#### Experiment on all files (including dynamic) recognition
- Current archtecture, specially the extracting features are limited because of the large number of class 
- It is easy to get a local optimal solution see from the loss
- so more epoch of trainings are needed to see the final result
- Training is time-comsuming and GPU will help a lot. 
- After 20 epochs(too small) training on cpu, results of every instruments shown

![all](./specific_acc.png)

- Total acc and single acc are shown

![all_all](./acc.png)



