from tensorflow import keras 

class Accuracy_Callback(keras.callbacks.Callback):
    def __init__(self,acc):
        self.acc = float(acc)

    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy')>self.acc):
            print(f"\nReached {self.acc}% accuracy so cancelling training!")
            self.model.stop_training = True