import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Load a model with specified configuration and compile it
def build_model(backbone, weights, classes, activation, img_size, channels, lr, crossentropy):

    base_model = tf.keras.applications.DenseNet121(include_top=False, weights=weights,
                                                       input_shape=(img_size, img_size, channels))

    base_model.trainable = True

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(classes, activation=activation)(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss=crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

# Train the segmentation model with generator input and save best weights
def train_model(model, train_gen, val_gen, path, model_name, foldno, save_best_only, epochs):
    # Define a checkpoint to save the best model based on validation F1 score
    checkpoint = ModelCheckpoint(
        path + model_name + '/models/model_' + str(foldno) + '.keras', save_best_only = True, verbose = 1)

    # Train the model using generators for training and validation data
    history = model.fit(train_gen,
                            epochs=epochs,
                            validation_data=val_gen,
                            callbacks=[checkpoint])

    return history

# Evaluate the trained model on validation/test data and return predictions
def predict_images(model, path, model_name, foldno, generator):

    Y_pred = model.predict(generator)

    y_pred = np.argmax(Y_pred, axis=1)

    prob = model.predict(generator)[:, 1]

    return y_pred, prob
