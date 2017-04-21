## Image Classification:

Architecture:</br>
1. Three convolutional layers with 64 filters in 3X3 dimenstion including a RELU activation to break linearity</br>
2. Max pooling is used with 2X2 dimension</br>
3. Flatten layer</br>
4. One fully connected layer with 256 nodes with RELU activation</br>
5. Output layer with one node since this is a binary classification with sigmoid activation</br>
6. 'rmsprop' is used an optimizer with 'binary_crossentropy' as loss function to get probabilites of output that sum upto 1.</br>


