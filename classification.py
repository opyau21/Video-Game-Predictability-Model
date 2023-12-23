import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import keras


def Logistic_Regression(x_train, y_train, x_test, y_test):
    tf.random.set_seed(100)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
                  loss='binary_crossentropy',
                  metrics=[
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')
                    ]
                )
    model.fit(x_train, y_train, epochs=12)

    loss_and_metrics = model.evaluate(x_test, y_test)
    print(loss_and_metrics)
    print('Loss = ', loss_and_metrics[0])
    print('Accuracy = ', loss_and_metrics[1])
    model.save('Oliver Model.keras')

