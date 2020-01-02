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

#### CNN+LSTM+CNN
Best acc: 91.963%

### Experiment on mixtures of 2,3,5,10 
|   exp |   2   |   3   |   5   |   10  |
|   ---------- |   ---------- |   ---------- |   ---------- |   ---------- |
|   accuracy | 1/2: 60.735% <br> 2/2: 37.415% | 1/3: 45.307% <br> 2/3: 45.640% <br> 3/3: 6.447% | 1/5: 24.098% <br> 2/5: 42.587% <br> 3/5: 27.062% <br> 4/5: 3.812% <br> 5/5: 0.070%  | on training |
|   each instrument | Va: 3856.0/10265.0 = 37.565% <br> Cb: 5862.0/10276.0 = 57.046% <br> Cbs: 1941.0/2148.0 = 90.363% <br> Vns: 3404.0/3831.0 = 88.854% <br> Vc: 6838.0/10990.0 = 62.220% <br> Fl: 4840.0/6600.0 = 73.333% <br> Vn: 4109.0/9254.0 = 44.402% <br> BTbn: 972.0/1185.0 = 82.025% <br> BClBb: 1680.0/1811.0 = 92.766% <br> ClBb: 2460.0/3283.0 = 74.931% <br> BFl: 689.0/713.0 = 96.634% <br> CbTb: 1393.0/1542.0 = 90.337% <br> Picc: 1081.0/1176.0 = 91.922% <br> ClEb: 581.0/615.0 = 94.472% <br> Ob: 2837.0/3357.0 = 84.510% <br> EH: 1803.0/1973.0 = 91.384% <br> CbFl: 713.0/729.0 = 97.805% <br> Bn: 3060.0/3610.0 = 84.765% <br> Vas: 2948.0/3188.0 = 92.472% <br> Vcs: 2649.0/2925.0 = 90.564% <br> CbClBb: 510.0/529.0 = 96.408%  | Va: 3044.0/15703.0 = 19.385% <br> Cb: 5471.0/15217.0 = 35.953% <br> Cbs: 2702.0/3409.0 = 79.261% <br> Vns: 3999.0/5608.0 = 71.309% <br> Vc: 7220.0/16541.0 = 43.649% <br> Fl: 5834.0/9899.0 = 58.935% <br> Vn: 3685.0/14006.0 = 26.310% <br> BTbn: 1132.0/1711.0 = 66.160% <br> BClBb: 2182.0/2562.0 = 85.168% <br> ClBb: 2899.0/4912.0 = 59.019% <br> BFl: 973.0/1068.0 = 91.105%  <br> CbTb: 1897.0/2311.0 = 82.086% <br> Picc: 1579.0/1777.0 = 88.858% <br> ClEb: 942.0/1023.0 = 92.082% <br> Ob: 3680.0/5019.0 = 73.321% <br> EH: 2632.0/2995.0 = 87.880% <br> CbFl: 1024.0/1068.0 = 95.880% <br> Bn: 3954.0/5384.0 = 73.440% <br> Vas: 3575.0/4610.0 = 77.549% <br> Vcs: 3163.0/4358.0 = 72.579% <br> CbClBb: 785.0/819.0 = 95.849%  | Va: 3333.0/25908.0 = 12.865% <br> Cb: 5997.0/25285.0 = 23.718% <br> Cbs: 3370.0/5657.0 = 59.572% <br> Vns: 4706.0/9404.0 = 50.043% <br> Vc: 8975.0/27583.0 = 32.538% <br> Fl: 8349.0/16569.0 = 50.389% <br> Vn: 4524.0/23219.0 = 19.484% <br> BTbn: 1338.0/2942.0 = 45.479% <br> BClBb: 3407.0/4499.0 = 75.728% <br> ClBb: 4168.0/8368.0 = 49.809% <br> BFl: 1520.0/1767.0 = 86.022% <br> CbTb: 2536.0/3679.0 = 68.932% <br> Picc: 2527.0/2866.0 = 88.172% <br> ClEb: 1324.0/1598.0 = 82.854% <br> Ob: 5637.0/8507.0 = 66.263% <br> EH: 3981.0/4953.0 = 80.376% <br> CbFl: 1632.0/1788.0 = 91.275% <br> Bn: 5548.0/9064.0 = 61.209% <br> Vas: 4678.0/7778.0 = 60.144% <br> Vcs: 3791.0/7272.0 = 52.131% <br> CbClBb: 1083.0/1294.0 = 83.694% | on training |


