<.## Churn Modelling

Implemented an Artificial Neural Network with two hidden layers to accurately predict if a customer would leave the bank or not.

## Architecture
Input Dimension = 12</br>
Hidden layer 1 = 6 Nodes, RELU activation with 'Dropout' = 0.1</br>
Hidden layer 2 = 6 Nodes, RELU activation with 'Dropout' = 0.1</br>
Output Dimension = 1 , SIGMOID activation</br>
Optimezer = 'ADAM'</br>

## Hyperparameter Tuning:
Utilized the Grid Search library in sklearn and got the best parameters as below</br>
Optimizer = 'rmsprop', batch_size = 32 , number of epochs = 500
