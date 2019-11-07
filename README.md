# DeepOrchestration

> It is a cool project using deep learning to help ochestration. Fighting!


### The first try is to implement cnn
- take 2 instruments from TinySOL and then combine 
- take melspectrogram features (128*128, if the combined pieces are small, just add zeros)
- a multi-label model used to take the loss from both instruments
- after 30 epochs training on cpu, accuracy results shown as follow

![acc](./acc.png) 


### Some problems
- The second accuracy has a bad reslut which contributes to a bad total result.
- The second accuracy increases with the first one at first, but after 6 epoch it decreases.
- The accuracy fluctuates sometimes(I think it's normal).

### other info:
- The decreasing loss of both instruments 
- Use `overlay` function from `pydub` to combine
- The size of data, add zeros 
- The settings of model





