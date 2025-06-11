import os
from scripts.data_preparation import load_and_preprocess_data
from scripts.model_building import build_model
from scripts.model_training import build_and_train_model
from scripts.evaluation import evaluate_model


def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Split some validation data from training set
    x_val, y_val = x_train[-10000:], y_train[-10000:]
    x_train, y_train = x_train[:-10000], y_train[:-10000]

    model = build_model()
    model = build_and_train_model(x_train, y_train, x_test, y_test)

    evaluate_model(model, x_test, y_test)

    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/digit_recognition_model.h5")


if __name__ == "__main__":
    main()
