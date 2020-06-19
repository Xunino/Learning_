from keras.losses import categorical_crossentropy
from keras.models import save_model
from mnist_Using_Compression.model_mnist import build_model
import tensorflow as tf

batch_size = 128
num_classes = 10
epochs = 10


img_rows, img_clos = 28, 28
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


print("Data_train:", len(x_train))
print("Labels_test:", len(y_train))
print("Data_test:", len(x_test))
print("Labels_test:", len(y_test))

# print(y_test[1])

x_train = x_train.reshape(x_train.shape[0], img_rows, img_clos, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_clos, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255.0
x_test /= 255.0

print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")

# convert class vectors to binary class metrics
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# initialize model
model = build_model(num_classes)

model.compile(loss=categorical_crossentropy,
    optimizer="Adam",
    metrics=["acc"]
)

model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Acc loss:", score[1])

save_model(model, "mnist_Using_Compression/mnist.h5", include_optimizer=False)