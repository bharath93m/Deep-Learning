## Churn Modelling

Implemented an Artificial Neural Network with two hidden layers to accurately predict if a customer would leave the bank or not.

## Architecture
Input Dimension = 12
Hidden layer 1 = 6 Nodes, RELU activation with 'Dropout' = 0.1
Hidden layer 2 = 6 Nodes, RELU activation with 'Dropout' = 0.1
Output Dimension = 1 , SIGMOID activation
Optimezer = 'ADAM'

## Hyperparameter Tuning:
Utilized the Grid Search library in sklearn and got the best parameters as below
Optimizer = 'rmsprop', batch_size = 32 , number of epochs = 500
