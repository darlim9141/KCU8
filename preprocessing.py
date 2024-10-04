import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(train_data_path, test_data_path):
    
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)

    y_train = train_data.iloc[:, 0]  # class labels
    X_train = train_data.iloc[:, 1:]  # pixel values 
    y_test = test_data.iloc[:, 0]
    X_test = test_data.iloc[:, 1:]

    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:8000]
    y_test = y_test[:8000]

    # (0 ~ 255) to (0 ~ 1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_test = encoder.transform(y_test.values.reshape(-1, 1))

    # 50:50
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(
        './archive/emnist-letters-train.csv', './archive/emnist-letters-test.csv'
    )
    print(f'Training set size: {X_train.shape}, Validation set size: {X_val.shape}, Test set size: {X_test.shape}')
    print(X_train[1])
    print(y_train[1])
