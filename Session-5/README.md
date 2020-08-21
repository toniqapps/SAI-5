# Assignment 5

### Dataset : MNIST

### Target : <8K Parameters, Validation Accuracy : 99.4

#### Absract: Design a network to achieve validation accuracy of 99.4 using less than 8K parameters. We had design a network with 7,668 parameters was able to achieve validation accuracy of 99.41 using regularization, dropout, image augmentation and then we reduced the parameters further to 6,765 parameters and were able to acheive validation accuracy of 99.52 %

#### Experiment:

##### Step 1:
```
Designed a basic network with 49,248 paramteres:

Results:
Parameters: 49,248

Best Train Accuracy: 99.61

Best Test Accuracy: 99.16

Analysis: The model parameters exceeds the required target and based on the training and test result we can see that the network is overfitting but still we can acheive the required target, let us try to reduce the parameters to the maximum
```
