from tensorflow import nn
from tensorflow import keras

mnist = keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

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
model.fit(X_train, Y_train, epochs = 5, callbacks = [callbacks])

model.save('digits_second_try')