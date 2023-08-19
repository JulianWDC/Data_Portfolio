import tensorflow as tf

mnist = tf.keras.datasets.mnist
mnist_model = tf.keras.models.load_model('digits_second_try')

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

correct = 0

for i in range(len(X_test)):
    img = X_test[i]
    img_tens = tf.convert_to_tensor(img)
    img_tens = tf.reshape(img_tens,(1,784))
    ans = mnist_model(img_tens)
    ans = int(tf.argmax(ans[0]))
    if ans == Y_test[i]:
        correct += 1
    if i%100 == 0:
        print(f'{i*.01} % completed')

print(f'Test accuracy is {correct}')