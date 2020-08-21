# Assignment 5 - Rahul Jain, Ashish Jain

## Dataset : MNIST

## Target : <8K Parameters, Validation Accuracy : 99.4

### Absract: 

Design a network to achieve validation accuracy of 99.4 using less than 8K parameters. We had design a network with 7,668 parameters was able to achieve validation accuracy of 99.41 using regularization, dropout, image augmentation and then we reduced the parameters further to 6,765 parameters and were able to acheive validation accuracy of 99.52 %


### Experiment:

#### Step 1:

##### 1. Designed a basic network with 49,248 paramteres:

##### Results:

Parameters: 49,248

Best Train Accuracy: 99.61

Best Test Accuracy: 99.16

Analysis: The model parameters exceeds the required target and based on the training and test result we can see that the network is overfitting but still we can achieve the required target, next step is to reduce the parameters

##### 2. Reduced basic network parameters from 49,248 to 7,668 and then to 6,633 

##### Results:

Parameters: 6,633

Best Train Accuracy: 99.24

Best Test Accuracy: 98.97

Analysis: After multiple regression, 7,668 parametersgave us traininig accuracy of 99.24 and test accuracy of 98.94 and based on the result we decided to use 6,633 as our target parameters

#### Step 2:

##### Results:

Parameters: 6,765

Best Train Accuracy: 99.81

Best Test Accuracy: 99.23

Analysis: Batch normalization reduces internal covariate shift by controlling the mean and variance of input distributions helping in faster convergence. From the above network result we can abserve that its overfitting and with the help of dropout and image augmentation we can reduce this overfitting and acheive the required target

#### Step 3:

##### Results:

Parameters: 6,765

Best Train Accuracy: 99.49

Best Test Accuracy: 99.30

Analysis: Using dropout have reduce the overfitting but still we are not able to achieve required accuracy, we will next try to implement image augmentation to see if it helps in improving network accuracy

#### Step 4:

##### Results:

Parameters: 6,765

Best Train Accuracy: 99.28

Best Test Accuracy: 99.40

Analysis: Based on the result we can observe that network is underfitting but that is because we are trying to force our network to learn harder on train data. Using image augmentation has improved our network capacity and was able to achieve required target with 6,765 parameters

#### Step 5:

##### Results:

Parameters: 6,765

Best Train Accuracy: 98.30

Best Test Accuracy: 99.52

Analysis: We have fine tuned dropout from 0.02 to 0.01. Using StepLR and starting learning rate at 0.1 and reducing by 0.5 after every 5th epoch. Based on the result learning rate has helped acheive test accuracy much faster and consistently

Problem : We had ran this network multiple times and were not able to hit the same accuracy again and again the accuracy varied from 99.39 to 99.52, but every time the result where consistent

