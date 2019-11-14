# DeepOrchestration

> It is a cool project using deep learning to help ochestration. Fighting!


### The first try is to implement cnn
- take N(currently 2) instruments from TinySOL and then combine 
- take melspectrogram features (128*128, if the combined pieces are small, just add zeros)
- a multi-label model used to take the loss from both instruments
- after 10 epochs training on cpu, accuracy results of every instruments shown as follow

![acc](./specific_acc.png) 

### Updating
- extend from 2 specific instruments to 2 instruments with different kinds of combinations 
- encode the labels in the format of 0-1 mitrix and take binary cross entropy with logits as loss function

### Problem
- see from the figure, the accuracy of both right predictions(total_acc) is low but the single right prediction is high.
- some kinds of instruments show low accuracy in the result like Tbn, Cb, Fl ...




