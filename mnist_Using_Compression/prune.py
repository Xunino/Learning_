import tensorflow_model_optimization as tfmot
import numpy as np
import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras.models import save_model

batch_size = 128
num_classes = 10
epochs = 10


img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# print(y_test[1])

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255.0
x_test /= 255.0
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class metrics
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


num_train_samples = x_train.shape[0]
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print("End step:" + str(end_step))

pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.5,
        final_sparsity=0.9,
        begin_step=2000,
        end_step=end_step,
        frequency=100
    )
}

pruned_model = tf.keras.Sequential([
    tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Conv2D(32, 5, padding="same", activation="relu"),
        input_shape=input_shape,
        **pruning_params
    ),
    tf.keras.layers.MaxPooling2D((2,2), (2,2), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tfmot.sparsity.keras.prune_low_magnitude(
        tf.keras.layers.Conv2D(64, 5, padding="same", activation="relu"),
        **pruning_params
    ),
    tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same"),
    tf.keras.layers.Flatten(),
    tfmot.sparsity.keras.prune_low_magnitude(
        tf.keras.layers.Dense(512, activation="relu"),
        **pruning_params
    ),
    tf.keras.layers.Dropout(0.2),
    tfmot.sparsity.keras.prune_low_magnitude(
        tf.keras.layers.Dense(num_classes, activation="softmax"),
        **pruning_params
    )
])

pruned_model.summary()

pruned_model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer="Adam",
        metrics=["acc"])

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()   
]

pruned_model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=10,
    verbose=1,
    callbacks=callbacks,
    validation_data=(x_test, y_test)
)

score = pruned_model.evaluate(x_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:" score[1])
print(score)

final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
final_model.summary()
tf.keras.models.save_model(final_model, "pruned_model_mnist.h5", include_optimizer=True)