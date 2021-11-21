# In this code we will use Ohm's Law (V = I*R), to predict the current in a basic electrical circuit, with a countinuous power source and and resistor, 
# with constant value. As we already know the results, we can see exactly how good the predicions are.
import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

# Create the logger, so we can keep track of what's happening.
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)

# Defining the pairs for training the model.
voltage = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype = float)
R = 13
current = voltage/R
# print(current)

# 1. We create the layer we will be applying to the model. Input_shape is just 1, as we're using a 1 dimension array. There's also going to be just one neuron in the
# layer. The model will be dense and fully connected, wich means all the neurons in different layers are connected with the ones in the following layer.
l0 = tf.keras.layers.Dense(units = 1, input_shape=[1])

# 2. Here we asseble the layers in the model, thus defining the model.
model = tf.keras.Sequential([l0])

# 3. The model is compiled. We indicate the error measure and the optimizer we'll be using.
model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))

# 4. Now we train the model with the pairs defined at the beginning. The number of epochs is the number of iterations done.  
history = model.fit(voltage, current, epochs=500, verbose=0)
print("\nModel 1 training succesful")

# Fit command returns a history type object, so we can directly represent it with Matplotlib. This way, we can easily see how the Mean Squared Error we selected
# compiling the model, gets lower as we execute more iterations. The error stabilizes near 500 epochs, getting almost a horizontal line, wich means
# it wouldnt get much better if we did more epochs.
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
# plt.show()

# 5. We predict the labels for the input values we're giving.
V = 15
print('The result for {} volts is {} amps'.format(V, model.predict([V])))
# The result we got is I = 1.5288615, and the correct result should be 1.1538461, so the prediction is not too good.

# We could check the weighs for each neuron used like this
# print("The layer weigh is: {}".format(l0.get_weights()))
mse1 = history.history['loss']
print('The error for model 1 is {}'.format(mse1[499]))


# Now we try using 3 layers, with 4 neurons in the first 2 layers and 1 neuron the last one (the last layer should have the same number of neurons as dimensions 
# expected in the prediction).
l1 = tf.keras.layers.Dense(units = 4, input_shape = [1])
l2 = tf.keras.layers.Dense(units = 4)
l3 = tf.keras.layers.Dense(units = 1)

model2 = tf.keras.Sequential([l1, l2, l3])
model2.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))
history2 = model2.fit(voltage, current, epochs=500, verbose=False)
print('\nModel 2 training succesful')
V2 = 15
print('The result for {} volts is {} amps'.format(V2, model2.predict([V2])))
mse2 = history2.history['loss']
print('The error for model 2 is {}'.format(mse2[499]))
# print("The layer l1 weighs are: {}".format(l1.get_weights()))
# print("The layer l2 weighs are: {}".format(l2.get_weights()))
# print("The layer l3 weigh is: {}".format(l3.get_weights()))

# In this case, the result was 1.1539, which is still a much better result, really close to the solution (1.1538461 A).

# Just to be sure, we'll try giving the model more pairs, to see if we could get a better result like this. Now we've got 50 pairs.
l_voltage = []
for i in range (10, 501, 10):
    l_voltage.append(i)
voltage3 = np.array(l_voltage)
# print(voltage)
R = 13
current3 = voltage3/R
l4 = tf.keras.layers.Dense(units = 1, input_shape = [1])
model3 = tf.keras.Sequential([l4])
model3.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))
history3 = model3.fit(voltage3, current3, epochs=500, verbose=0)
print('\nModel 3 training succesful')
V3 = 15
print('The result for {} volts is {} amps'.format(V3, model3.predict([V])))
mse3 = history3.history['loss']
print('The error for model 3 is {}'.format(mse3[499]))

# Here the prediction was 1.1538652, which is the best result so far. It seems its better to give the model more pairs, than giving it more layers.

# Note that when we run the code several times, we get different results each time. Most of the times, we get the best result for model 2, which has more layers 
# and just 10 pairs. Then model 3 is the best performer, having much more pairs but still one layer. Most of the times, the worse performer was model 1, with 10
# pairs and one layer. 
# Also, normally we would measure the performance of our model by measuring the error. In this case, we already know the answer, so we can evaluate comparing with
# the real value we should get.

# Alejandro García Lagos
