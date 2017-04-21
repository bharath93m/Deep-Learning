#import the keras libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN
classifier = Sequential()

#Step 1 - Convolution Layer
classifier.add(Convolution2D(64,(3,3),input_shape = (64, 64, 3), activation = 'relu'))
#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#second convolution layer
classifier.add(Convolution2D(64,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#third convolution layer - test
classifier.add(Convolution2D(64,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Fully connected layer
classifier.add(Dense(units=256,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# fit CNN to image

from keras.preprocessing.image  import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'C:/Users/Brox/Desktop/Deep Learning/Convolutional_Neural_Networks/dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/Brox/Desktop/Deep Learning/Convolutional_Neural_Networks/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
                            steps_per_epoch=8000/32,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=2000/32,workers=32,max_q_size=16)
#New predictions

import numpy as np
from keras.preprocessing import image
# load the image
test_image = image.load_img('C:/Users/Brox/Desktop/Deep Learning/Convolutional_Neural_Networks/dataset/single_prediction/test1.jpg',target_size=(64, 64))
# convert it to an array to match the input dimension of the convolution layer
test_image = image.img_to_array(test_image)
# predict method expects a fourth argument 
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
if result == 1:
    print('Dog')
else:
    print('Cat')

