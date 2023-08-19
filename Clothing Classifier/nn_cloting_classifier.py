#Clothing classification with computer vision
from tensorflow import nn
from tensorflow import keras

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('loss') < 0.4):
            print("\nLoss is low so cancelling training.")
            self.model.stop_training = True

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

callbacks = myCallback()

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28),),
    keras.layers.Dense(512, activation = nn.relu),
    keras.layers.Dense(512, activation = nn.relu),
    keras.layers.Dense(512, activation = nn.relu),
    keras.layers.Dense(512, activation = nn.relu),
    keras.layers.Dense(10, activation=nn.softmax)
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])
model.fit(train_images, train_labels, epochs = 5)

model.save('clothing_classification_model')