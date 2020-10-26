This repository serves as the code base for the paper ["A Study on Neural Models for Target-Based Computer-Assisted Musical Orchestration"](https://boblsturm.github.io/aimusic2020/papers/CSMC__MuMe_2020_paper_43.pdf)

**Visit our github.io page to listen to orchestrated examples**: https://dzluke.github.io/DeepOrchestration/

## train.py

Used to train the model.

## test.py

Orchestrates target samples given a pre-trained model

## model.py

Contains the CNN with LSTM model and generalized `OrchMatchNet` class
Imports the ResNet model from `resnet.py`

## resnet.py

Contains the ResNet model

## OrchDataset.py

Contains the `RawDatabase` and `OrchDataSet` classes.

`RawDatabase` contains all the raw TinySOL samples
`OrchDataSet` takes in a `RawDatabase`. The `generate` function creates combinations of samples. The features of each combination are calculated during training when `__getitem__` is called.

## parameters.py

Contains the class `SimParams` which is used to save and load the parameters of a trained model

## baseline_classification.py

Self-contained file that generates combinations of samples, uses MFCC as features, and runs SVM for classification. This is the baseline that our deep models were compared against.


