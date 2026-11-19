import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train_mnist.csv")
    parser.add_argument("--output_dir", type=str, default="output")
    return parser.parse_args()

# You can view model description, accuracy and best parameters in data/main_analysis.ipynb
def main():
    args = parse_args()
    mlflow.tensorflow.autolog()

    data_path = args.data_path if hasattr(args, "data_path") and args.data_path and os.path.exists(args.data_path) else "train_mnist.csv"
    if not os.path.exists(data_path) and os.path.exists("data/train_mnist.csv"):
        data_path = "data/train_mnist.csv"

    df = pd.read_csv(data_path)
    processed_images = []
    labels = []
    columns = df.columns
    for index, row in df.iterrows():
        size = 28
        image = []
        label = row["label"]
        for i in range(size):
            image_row = []
            for j in range(size):
                image_row.append(0)
            image.append(image_row)

        for column in columns:
            splited_column = column.split("x")
            if not len(splited_column) == 2:
                continue
            x = int(splited_column[0]) - 1
            y = int(splited_column[1]) - 1
            image[x][y] = row[column]

        labels.append(label)
        processed_images.append(image)

    X = np.array(processed_images, dtype="float32") / 255.0
    X = np.expand_dims(X, axis=-1)
    y = np.array(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def build_model(hp):
        model = keras.Sequential([keras.Input(shape=(28, 28, 1))])
        for i in range(hp.Int("num_conv_layers", 1, 2)):
            model.add(layers.Conv2D(
                filters=hp.Int(f"filters_{i}", 32, 64, step=16),
                kernel_size=hp.Choice(f"kernel_{i}", [3, 5]),
                padding="same"
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation("relu"))
            model.add(layers.MaxPooling2D(pool_size=2))
            model.add(layers.Dropout(hp.Float(f"dropout_{i}", 0.1, 0.3, step=0.1)))

        model.add(layers.Flatten())
        model.add(layers.Dense(hp.Int("dense_units", 64, 192, step=64), activation="relu"))
        model.add(layers.Dropout(hp.Float("dense_dropout", 0.2, 0.5, step=0.1)))
        model.add(layers.Dense(10, activation="softmax"))

        model.compile(
            optimizer=keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        directory="/tmp/kt_dir",
        project_name="mnist_cnn"
    )

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    tuner.search(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=2
    )

    best_model = tuner.get_best_models(num_models=1)[0]

    os.makedirs(args.output_dir, exist_ok=True)
    best_model.save(os.path.join(args.output_dir, "saved_model.keras"))

    y_pred = np.argmax(best_model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
