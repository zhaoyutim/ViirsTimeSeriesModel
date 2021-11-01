import datetime

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras.backend as K
class GRUModel:
    def __init__(self):
        self.model =  self.get_model()

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    def r2_keras(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    def get_model(self):
        gru_model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(256, input_shape=(10, 45), return_sequences=True),
            tf.keras.layers.GRU(128, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(64, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(32, return_sequences=True),
            tf.keras.layers.GRU(16, return_sequences=True),
            tf.keras.layers.Dense(2)
        ])
        return gru_model

    def compile_and_fit(self, model, train_dataset, patience=2):
        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      optimizer=tf.optimizers.Adam(1e-3),
                      metrics=[tf.keras.losses.CategoricalCrossentropy(from_logits=True), self.f1_m])

        history = model.fit(x_train, y_train, epochs=MAX_EPOCHS,
                            validation_split=0.2)
        return history