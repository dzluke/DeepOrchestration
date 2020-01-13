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
#### Loss:

![loss](./exp/loss.png)


#### Accuracy Table:
|   exp |   2   |   3   |   5   |   10  |
|   ---------- |   ---------- |   ---------- |   ---------- |   ---------- |
|   accuracy | 1/2: 15.350% <br> 2/2: 84.470% | 1/3: 3.680% <br> 2/3: 40.850% <br> 3/3: 55.340% | 1/5: 0.470% <br> 2/5: 5.800% <br> 3/5: 33.940% <br> 4/5: 49.590% <br> 5/5: 10.170% | 1/10: 0.040% <br> 2/10: 0.890% <br> 3/10: 5.300% <br> 4/10: 19.130% <br> 5/10: 34.120% <br> 6/10: 29.130% <br> 7/10: 9.970% <br> 8/10: 1.360% <br> 9/10: 0.060% <br> 10/10: 0.000% |
|   each instrument | Va: 786.0/815.0 = 96.442% <br> Cb: 568.0/702.0 = 80.912% <br> Vns: 807.0/829.0 = 97.346% <br> Vc: 772.0/793.0 = 97.352% <br> BTb: 946.0/1127.0 = 83.940% <br> Fl: 1077.0/1111.0 = 96.940% <br> Vn: 672.0/714.0 = 94.118% <br> Hn: 661.0/973.0 = 67.934% <br> BTbn: 671.0/732.0 = 91.667% <br> BClBb: 890.0/908.0 = 98.018% <br> ClBb: 803.0/842.0 = 95.368% <br> TpC: 453.0/489.0 = 92.638% <br> TTbn: 897.0/1124.0 = 79.804% <br> BFl: 186.0/188.0 = 98.936% <br> CbTb: 802.0/839.0 = 95.590% <br> Picc: 635.0/638.0 = 99.530% <br> ClEb: 260.0/261.0 = 99.617% <br> Acc: 1401.0/1437.0 = 97.495% <br> Ob: 1069.0/1129.0 = 94.686% <br> EH: 731.0/741.0 = 98.650% <br> CbFl: 224.0/224.0 = 100.000% <br> Bn: 1008.0/1139.0 = 88.499% <br> Vas: 768.0/785.0 = 97.834% <br> Vcs: 758.0/781.0 = 97.055% <br> ASax: 341.0/433.0 = 78.753% <br> CbClBb: 243.0/246.0 = 98.780% | Va: 1002.0/1151.0 = 87.055% <br> Cb: 662.0/1169.0 = 56.630% <br> Vns: 1267.0/1323.0 = 95.767% <br> Vc: 976.0/1119.0 = 87.221% <br> BTb: 1276.0/1694.0 = 75.325% <br> Fl: 1519.0/1628.0 = 93.305% <br> Vn: 908.0/1111.0 = 81.728% <br> Hn: 796.0/1434.0 = 55.509% <br> BTbn: 795.0/1161.0 = 68.475% <br> BClBb: 1345.0/1418.0 = 94.852% <br> ClBb: 1153.0/1238.0 = 93.134% <br> TpC: 678.0/793.0 = 85.498% <br> TTbn: 1242.0/1706.0 = 72.802% <br> BFl: 275.0/281.0 = 97.865% <br> CbTb: 1170.0/1315.0 = 88.973% <br> Picc: 883.0/909.0 = 97.140% <br> ClEb: 366.0/371.0 = 98.652% <br> Acc: 1893.0/2063.0 = 91.760% <br> Ob: 1399.0/1673.0 = 83.622% <br> EH: 971.0/1043.0 = 93.097% <br> CbFl: 340.0/346.0 = 98.266% <br> Bn: 1239.0/1635.0 = 75.780% <br> Vas: 1177.0/1224.0 = 96.160% <br> Vcs: 1077.0/1168.0 = 92.209% <br> ASax: 357.0/645.0 = 55.349% <br> CbClBb: 374.0/382.0 = 97.906% | Va: 1357.0/1987.0 = 68.294% <br> Cb: 730.0/1845.0 = 39.566% <br> Vns: 1869.0/2370.0 = 78.861% <br> Vc: 1391.0/2005.0 = 69.377% <br> BTb: 1907.0/2767.0 = 68.919% <br> Fl: 2424.0/2781.0 = 87.163% <br> Vn: 1085.0/1947.0 = 55.727% <br> Hn: 959.0/2333.0 = 41.106% <br> BTbn: 1037.0/1867.0 = 55.544% <br> BClBb: 2033.0/2358.0 = 86.217% <br> ClBb: 1695.0/1971.0 = 85.997% <br> TpC: 1046.0/1323.0 = 79.063% <br> TTbn: 1686.0/2769.0 = 60.888% <br> BFl: 534.0/554.0 = 96.390% <br> CbTb: 1625.0/2112.0 = 76.941% <br> Picc: 1381.0/1472.0 = 93.818% <br> ClEb: 601.0/625.0 = 96.160% <br> Acc: 2954.0/3506.0 = 84.256% <br> Ob: 2149.0/2832.0 = 75.883% <br> EH: 1531.0/1728.0 = 88.600% <br> CbFl: 465.0/480.0 = 96.875% <br> Bn: 1767.0/2816.0 = 62.749% <br> Vas: 1633.0/2000.0 = 81.650% <br> Vcs: 1490.0/1994.0 = 74.724% <br> ASax: 406.0/960.0 = 42.292% <br> CbClBb: 555.0/598.0 = 92.809%  | Va: 1362.0/3826.0 = 35.599% <br> Cb: 934.0/3755.0 = 24.874% <br> Vns: 2183.0/4539.0 = 48.094% <br> Vc: 1651.0/3890.0 = 42.442% <br> BTb: 2546.0/5454.0 = 46.681% <br> Fl: 4163.0/5504.0 = 75.636% <br> Vn: 1213.0/3682.0 = 32.944% <br> Hn: 994.0/4690.0 = 21.194% <br> BTbn: 1173.0/3935.0 = 29.809% <br> BClBb: 3243.0/4720.0 = 68.708% <br> ClBb: 2838.0/3958.0 = 71.703% <br> TpC: 1608.0/2657.0 = 60.519% <br> TTbn: 1837.0/5695.0 = 32.256% <br> BFl: 893.0/1059.0 = 84.325% <br> CbTb: 2041.0/4279.0 = 47.698% <br> Picc: 2713.0/3011.0 = 90.103% <br> ClEb: 1093.0/1224.0 = 89.297% <br> Acc: 4663.0/6720.0 = 69.390% <br> Ob: 3686.0/5780.0 = 63.772% <br> EH: 2745.0/3620.0 = 75.829% <br> CbFl: 877.0/1062.0 = 82.580% <br> Bn: 2720.0/5601.0 = 48.563% <br> Vas: 2007.0/4044.0 = 49.629% <br> Vcs: 1467.0/4018.0 = 36.511% <br> ASax: 418.0/2041.0 = 20.480% <br> CbClBb: 1015.0/1236.0 = 82.120%  |


### 5-mixture exp on more data (250,000)

#### loss: 

![loss_five](./exp/loss_five.png) 

overfitting 

#### try to add L1, L2 regularization

L2: weight decay: 0.005

![loss_L2](./exp/loss_five_L2_1.png)

L2: weight decay: 1e-6

![loss_L2](./exp/loss_five_L2.png)

L1: weight decay: 0.01

![loss_L1](./exp/loss_five_L1.png)