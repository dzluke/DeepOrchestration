# DeepOrchestration

> It is a cool project using deep learning to help ochestration. Fighting!


### The first try is to implement cnn
- take 2 instruments from TinySOL and then combine 
- take melspectrogram features (128*128, if the combined pieces are small, just add zeros)
- a multi-label model used to take the loss from both instruments
- after 50 epochs training on cpu, accuracy results shown as follow

![acc](./acc.png) 


### Updating!
- update the way to mix
- simplify the model which is faster and more accurate now
- encode labels or not can be successful, which depends on the loss function, more detials can be seen [here][1]. 

[1]: https://pytorch.org/docs/stable/nn.html




