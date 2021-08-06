#Uncomment for visualising data-set
#import matplotlib.pyplot as plt
#import os
#import numpy as np 


# DATADIR = "D:/PYTHON_PROJECT/dataset/train"
# categories = ["NORMAL", "PNEUMONIA"]


# path = os.path.join(DATADIR,"Normal") #path to the train set examples
# countn = 0
# fig = plt.figure (figsize = (8,8))
# fig.suptitle(" Some Normal lungs X-rays with their shapes in the train data set")
# for img in os.listdir(path):
    
#     img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
    
#     if countn < 20:
#         fig.add_subplot(4,5,countn+1)
#         plt.imshow(img_array,cmap="gray")
#         shape = np.shape(img_array)
#         plt.xlabel(shape)
        
#     countn+=1
    
#     if countn == 20 :
#         plt.show()
        
    
# path = os.path.join(DATADIR,"PNEUMONIA") #path to the train set examples
# countp = 0
# fig = plt.figure (figsize = (8,8))
# fig.suptitle(" Some Pneumatic lungs X-rays with their shapes in the train data set")
# for img in os.listdir(path):
    
#     img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
    
#     if countp < 20:
#         fig.add_subplot(4,5,countp+1)
#         plt.imshow(img_array,cmap="gray")
#         shape = np.shape(img_array)
#         plt.xlabel(shape)
        
#     countp+=1
    
#     if countp == 20 :
#         plt.show()

# print ("Number of Normal lung X-rays in the train set = "+str(countn))   
# print ("Number of Pneumatic lung X-rays in the train set = "+str(countp))   
# print("Total Number of X-ray images in the Train set = "+str(countn+countp))
    
    
    



# Importing the Keras libraries and packages Deep Learning

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64,3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 5216,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 624 )



training_set.class_indices
classifier.save("classifier2.h5")

