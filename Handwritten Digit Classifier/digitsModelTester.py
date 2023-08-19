import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import random

nums = {
    0: 'zero',
    1: 'One',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine'
}

mnist = keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

mnist_model = keras.models.load_model('digits_second_try')

choice = random.randint(0,9999)

img = X_train[choice]
label = Y_train[choice]
# flat_img = []
# for sublist in img:
#     for item in sublist:
#         flat_img.append(item)
# flat_img = np.array(flat_img)

img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

ans = mnist_model(img_batch)
print(ans)
print(label)
ans = int(np.argmax(ans[0]))
print(ans)

plt.imshow(img, cmap="Greys")
if ans == 8:
    plt.title(f"It's an {nums[ans]}", fontsize = 32)
else:
    plt.title(f"It's a {nums[ans]}", fontsize = 32)

plt.show()