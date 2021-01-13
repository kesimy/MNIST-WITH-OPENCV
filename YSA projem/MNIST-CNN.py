
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#%%
(x_train, y_train), (x_test, y_test) = mnist.load_data() # veri setini indirip kullanıma hazır hale getiriyor.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
x_train = x_train / 255
x_test = x_test / 255
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
model = Sequential()

model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())#tek boyutlu diziye çeviriyor.
# fully connected
model.add(Dense(32, activation=tf.nn.relu))
#model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
fit = model.fit(x_train,
          y_train,
          validation_data=(x_test,y_test),
          batch_size=32,
          epochs=5)
model.save('MNIST-CNN.model')
#%%
a = plt.scatter(fit.history["accuracy"],fit.history["loss"], color = "red")
plt.ylabellabel("loss")
plt.xlabel("accuracy")
plt.title("PLOT-ACCURACY")
plt.show()
print("plot edildi")    




