# Classification of a DS containing hands playing rock, paper or scissor. 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import math as math


tfds.disable_progress_bar()
# print(tfds.list_builders())         # This shows all the list of DS from TF
builder = tfds.builder('rock_paper_scissors')
info = builder.info

# After printing the info, we get the following information:
# features=FeaturesDict({
#     'image': Image(shape=(300, 300, 3), dtype=tf.uint8),
#     'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=3),
# Shape is quite useful, as we get to know the dimensions of each image and that each one has 3 colours (RGB), inside the 300*300 pixels
# Data type uint8 means the images have a value of 0-255 for each pixel. 
# It's also interesting to check labels, as it says the number of image types there are. (num_classes)


# 1. Here we prepare the data. First we load it from the TF DS.
ds_train = tfds.load(name="rock_paper_scissors", split="train")
ds_test = tfds.load(name="rock_paper_scissors", split="test")
# We could also use this to get the train and test data, as TF already has separated them
# ds = tfds.load('rock_paper_scissors')
# ds_train, ds_test = ds['train'], ds['test']
# print(tfds.show_examples(info, ds_train))       # This shows some images as example

# Then we convert to numpy format, which is easier for working with it.
train_images = [example['image'] for example in ds_train]
# print(train_images[0])
print(type(train_images[0]))
# Type is <class 'tensorflow.python.framework.ops.EagerTensor'> so we change it to numpy format
train_images = [example['image'].numpy() for example in ds_train]
print(type(train_images[0]))        # The data is contained in a list. We prefer having it in a np array better
# We just use one color, as we want to 
train_images = np.array([example['image'].numpy()[:,:,0] for example in ds_train])
# predict by shapes, not colors.
print(type(train_images[0]))
print(train_images.shape)
# (2520, 300, 300) that's the shape

# Same as before but for labels instead of images, and for the test images and labels
train_labels = np.array([example['label'].numpy() for example in ds_train])
print('Train label example: ', train_labels[0])
print(type(train_labels[0]))

test_images = np.array([example['image'].numpy()[:,:,0] for example in ds_test])
test_labels = np.array([example['label'].numpy() for example in ds_test])
print(test_images.shape)
# (372, 300, 300) that's the shape

# We reshape train and test images, because Keras needs to have an input in the the color channel
train_images = train_images.reshape(2520, 300, 300, 1)
test_images = test_images.reshape(372, 300, 300, 1)

# It's frequent to work with normalized values for the pixels. We'll use 0-1 values, instead 0-255. Also, the type is
# an int value, so we change it to float to divide
print(train_images.dtype)
# We get uint8, which is has int values

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# If we check the values now, we'll see they are in the right range.
# print(train_images[0])
print(train_images.dtype)
# Now we get float32

# 2. Here the model is trained. (2.1: Define the layers, 2.2: Compile model, 3: Train the model)
model = keras.Sequential([
keras.layers.Flatten(),
keras.layers.Dense(512, activation='relu'),
keras.layers.Dense(256, activation='relu'),
keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
loss=keras.losses.SparseCategoricalCrossentropy(),
metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=32)
# On the first training attempt we got 0.8052 accuracy for the last Epoch, but lets try now with evaluate

model.evaluate(test_images, test_labels)
# We get 0.53 accuracy in the evaluate, which is much worse than the previous results. The model doesnt do that well when we use it on unseen data 


""" # 3. Alternatives. Now we try to improve the accuracy of the evaluation by using a convolutional network.
# The conv net is like using a smaller grid to learn the details of the images
model = keras.Sequential([
keras.layers.Conv2D(64, 3, activation='relu', input_shape=(300,300,1)),     # 64 times passing with the smaller grid, 3 is the size of that smaller grid (square)
keras.layers.Conv2D(32, 3, activation='relu'),
keras.layers.Flatten(),
keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
loss=keras.losses.SparseCategoricalCrossentropy(),
metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=32)
model.evaluate(test_images, test_labels)
# This time the evaluation gives 0.47, which is worse than before """


""" # The results of the convolutional net were almost the same. We're still overfitting to the train data and we dont have that much examples.
# This might be because the kernel size is quite small compared to each image. We'll try to reduce the quality of the images now, to have
# a smaller input size, meaning a smaller grid for each image, and a better result while mantaining the kernel size 3*3.
model = keras.Sequential([
keras.layers.AveragePooling2D(6, 3, input_shape = (300, 300, 1)),        # Here we're averaging with a 6*6 box, but we only advance 3 pixels each time
keras.layers.Conv2D(64, 3, activation = 'relu'),
keras.layers.Conv2D(32, 3, activation = 'relu'),
keras.layers.Flatten(),
keras.layers.Dense(3, activation = 'softmax')
])

model.compile(optimizer = 'adam', 
loss = keras.losses.SparseCategoricalCrossentropy(), 
metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=32)
model.evaluate(test_images, test_labels)
# This time we get 0.66 in the evaluation. Better than before. """


""" # Now we try the same conv net, but using the MaxPooling for taking pixels in pairs, a getting only the max value. Also we use the Dropout, where we
# prevent using 50% of the data, forcing the model to make conexions with all the data left. Last, we add another dense layer with 128 nodes, to 
# try to get even better results.
model = keras.Sequential([
keras.layers.AveragePooling2D(6, 3, input_shape = (300, 300, 1)),        # Here we're averaging with a 6*6 box, but we only advance 3 pixels each time
keras.layers.Conv2D(64, 3, activation = 'relu'),
keras.layers.Conv2D(32, 3, activation = 'relu'),
keras.layers.MaxPool2D(2, 2),
keras.layers.Dropout(0.5),            # Here we're taking pixels in pairs, and advancing 2 pixels each time
keras.layers.Flatten(),
keras.layers.Dense(128, activation = 'relu'),
keras.layers.Dense(3, activation = 'softmax')
])

model.compile(optimizer = 'adam', 
loss = keras.losses.SparseCategoricalCrossentropy(), 
metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=32)
model.evaluate(test_images, test_labels)
# Finally, we got better results in the evaluation.  """


# 4. Starting with the last model defined, we'll use Keras Tuner, to get the best parameters possible for the layers.
# pip install -U keras-tuner
from kerastuner.tuners import RandomSearch
def build_model(hp):
    model = keras.Sequential()

    model.add(keras.layers.AveragePooling2D(6, 3, input_shape = (300, 300, 1)))        # Here we're averaging with a 6 filters, but we only advance 3 pixels each time

    for i in range(hp.Int("Conv Layers", min_value=0, max_value=3)):        # This let's us select the best number of layers between min and max values
        model.add(keras.layers.Conv2D(hp.Choice(f"layer_{i}_filters", [16,32,64]), 3, activation='relu'))       # Here we can get the best number of filters

    model.add(keras.layers.Conv2D(32, 3, activation = 'relu'))
    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Dropout(0.5))            # Here we're taking pixels in pairs, and advancing 2 pixels each time
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Choice('Dense Layer', [64, 128, 256, 512, 1024]), activation = 'relu'))         # And here we get the best number of neurons
    model.add(keras.layers.Dense(3, activation = 'softmax'))

    model.compile(optimizer = 'adam', 
    loss = keras.losses.SparseCategoricalCrossentropy(), 
    metrics = ['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=32)
    model.evaluate(test_images, test_labels)
    
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=15,)

tuner.search(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=32)

best_model = tuner.get_best_models()[0]
best_model.evaluate(test_images, test_labels)
# This time we got accuracy: 0.7608, which is probably the best acc we got so far.
print(best_model.summary())
print(tuner.search_space_summary())
print(tuner.results_summary())

#  Layer (type)                Output Shape              Param #
# =================================================================
#  average_pooling2d (AverageP  (None, 99, 99, 1)        0
#  ooling2D)
#  conv2d (Conv2D)             (None, 97, 97, 32)        320
#  max_pooling2d (MaxPooling2D  (None, 48, 48, 32)       0
#  )
#  dropout (Dropout)           (None, 48, 48, 32)        0
#  flatten (Flatten)           (None, 73728)             0
#  dense (Dense)               (None, 256)               18874624
#  dense_1 (Dense)             (None, 3)                 771

# Trial summary (best score)
# Hyperparameters:
# Conv Layers: 0
# Dense Layer: 256
# layer_0_filters: 64       # This line and the next would be useful in case there were more than 0 convolutional layers indicated. Not in this case.
# layer_1_filters: 16
# Score: 0.7607526779174805
# Trial summary
# Hyperparameters:
# Conv Layers: 2
# Dense Layer: 1024
# layer_0_filters: 32
# layer_1_filters: 16
# Score: 0.7338709831237793
# Trial summary
# Hyperparameters:
# Conv Layers: 1
# Dense Layer: 256
# layer_0_filters: 32
# layer_1_filters: 16
# Score: 0.7096773982048035
# Trial summary
# Hyperparameters:
# Conv Layers: 1
# Dense Layer: 1024
# layer_0_filters: 16
# Score: 0.6908602118492126
# Trial summary
# Hyperparameters:
# Conv Layers: 1
# Dense Layer: 64
# layer_0_filters: 64
# layer_1_filters: 64
# Score: 0.6854838728904724

""" # Now we plot the results
print(train_images[0].shape)       # We need to reshape, to avoid the last 1 (shape is (300, 300, 1))
image = train_images[0].reshape(300,300)
plt.imshow(image, cmap = 'Greys_r')
plt.show() """

# # To plot while keeping colors
# color_images= np.array([example['image'].numpy() for example in ds_train.take(1)])
# color_im = color_images[0]
# image = train_images[0].reshape(300,300)
# plt.imshow(color_im)
# print(color_im.shape)   
# plt.show()

""" # Here we can predict a single image
result = best_model.predict(np.array([train_images[0]]))
print(result)

predicted_value = np.argmax(result)
print(predicted_value) """


# 5. Here we convert some JPG images I took with my phone, to Numpy format, so we can predict them.
import imageio
import numpy as np
import matplotlib.pyplot as plt

preds = []
# print(train_images[0:10])
for i in range(1,10):
    url_load = 'https://github.com/AleGL92/TensorFlow/blob/main/HandPics/M' + str(i) + '.jpg?raw=true'
    im = imageio.imread(url_load)
# print(im)

    print(type(im))
    im_np = np.asarray(im)
    print(im_np.shape)

    im_np = im_np[:,:,0]
    # im_np = im_np.reshape(300, 300)
    # print(im_np.shape)

    im_np = im_np.astype('float32')
    im_np /= 255
    # plt.imshow(im_np, cmap = 'Greys_r')
    # plt.show()

    # result = model.predict(np.array([im_np]))
    result = best_model.predict(np.array([im_np]))
    print(result)

    predicted_value = np.argmax(result)
    # print(predicted_value)
    preds.append(predicted_value)
    print(preds)

# Real_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
# Pred_labels = [2, 1, 1, 0, 2, 2, 0, 1, 2]
            #    N  Y  N  Y  N  Y  Y  Y  Y      6/10 correctly predicted
    
# Alejandro García Lagos
