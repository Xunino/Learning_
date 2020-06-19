from keras.models import Sequential
from keras import layers

def build_model(num_classes):

    input_shape = (28, 28, 1)
    
    model = Sequential()
    model.add(layers.Conv2D(32, 5, padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, 5, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.summary()
    return model
