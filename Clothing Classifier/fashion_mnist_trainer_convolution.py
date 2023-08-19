#Clothing classification with convoluton and pooling
from tensorflow import nn
from tensorflow import keras
import time

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images=train_images.reshape(60000, 28, 28, 1)
train_images=train_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

from accuracy_callback import Accuracy_Callback
acc_callback = Accuracy_Callback(0.9)
tb_callback = keras.callbacks.TensorBoard(log_dir = f"logs/fashion_with_convoluiton{time.time()}")

model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = nn.relu),
    keras.layers.Dense(512, activation = nn.relu),
    keras.layers.Dense(10, activation=nn.softmax)
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])
model.fit(train_images, train_labels, epochs = 5, callbacks = [tb_callback, acc_callback])

model.save('clothing_classification_model_with_convolution_and_pooling')