import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import random

clothes = {
    0: 'T-shirt/top',
    1: 'Pants',
    2: 'Sweater',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Handbag',
    9: 'Ankle Boot'
}

mnist = keras.datasets.fashion_mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

mnist_model = keras.models.load_model('clothing_classification_model_with_convolution_and_pooling')

choice = random.randint(0,len(Y_test))

img = X_train[choice]
label = Y_train[choice]

img_tens = np.asarray(img)
img_tens = np.reshape(img_tens,(1,28,28,1))

ans = mnist_model(img_tens)
print(ans)
print(label)
ans = int(np.argmax(ans[0]))
print(ans)

plt.imshow(img, cmap="Greys")
plt.title(clothes[ans], fontsize = 22)

plt.show()