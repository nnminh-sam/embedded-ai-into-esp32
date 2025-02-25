import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math
import os
from utils.hex_to_c_array import hex_to_c_array


class SineModel:
    def __init__(
        self,
        sample_count=1000,
        validation_ratio=0.2,
        test_ratio=0.2,
        model_name="model",
    ):
        # Configuration
        self.sample_count = sample_count
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.model_name = model_name
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.h_file_path = os.path.join(self.current_dir, f"{model_name}.h")
        self.tflite_file_path = os.path.join(self.current_dir, f"{model_name}.tflite")

        # Initialize dataset
        self.x_train, self.x_validation, self.x_test = None, None, None
        self.y_train, self.y_validation, self.y_test = None, None, None
        self.generate_dataset()

        # Initialize model
        self.model = self.create_model()

    def generate_dataset(self):
        x_values = np.random.uniform(low=0, high=(2 * math.pi), size=self.sample_count)
        y_values = np.sin(x_values) + (0.1 * np.random.randn(x_values.shape[0]))

        validation_set_count = int(self.validation_ratio * self.sample_count)
        test_set_count = validation_set_count + int(self.test_ratio * self.sample_count)

        self.x_validation, self.x_test, self.x_train = np.split(
            x_values, [validation_set_count, test_set_count]
        )
        self.y_validation, self.y_test, self.y_train = np.split(
            y_values, [validation_set_count, test_set_count]
        )

        assert (
            self.x_train.size + self.x_validation.size + self.x_test.size
        ) == self.sample_count

    def create_model(self):
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(1,)),
                layers.Dense(16, activation="sigmoid"),
                layers.Dense(16, activation="sigmoid"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="rmsprop", loss="mae", metrics=["mae"])
        return model

    def train(self, epochs=500, batch_size=32):
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_validation, self.y_validation),
        )

    def predict(self, x_data=None):
        if x_data is None:
            x_data = self.x_test
        return self.model.predict(x_data)

    def convert_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model = converter.convert()

        with open(self.tflite_file_path, "wb") as file:
            file.write(tflite_model)

        with open(self.h_file_path, "w") as file:
            file.write(hex_to_c_array(tflite_model, self.model_name))

        print(f"TFLite model saved to: {self.tflite_file_path}")
        print(f"Header file saved to: {self.h_file_path}")
