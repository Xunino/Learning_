import tensorflow as tf
import numpy as np
from model_mnist import build_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

n = 5000

x_test = x_test[n]
x_test = x_test.reshape(28, 28, 1)
x_test = np.expand_dims(x_test, axis=0)
x_test = x_test.astype("float32") / 255.0


# y_test = y_test[n]
# y_test = tf.keras.utils.to_categorical(y_test, 10)

model = build_model(num_classes=10)
model.load_weights("mnist_Using_Compression\pruned_model_mnist.h5")
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer="Adam", metrics=["acc"])

score = model.predict(x_test)
label = model.predict_classes(x_test)
print(score)
print(label, y_test[n])