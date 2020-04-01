from model import DNN
import numpy as np

from utils import load_data

def main():
    X_train, Y_train, X_test, Y_test= load_data()

    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape: {X_test.shape}")
    print(f"Y_train Shape: {Y_train.shape}")
    print(f"Y_test Shape: {Y_test.shape}")

    model = DNN([X_train.shape[0], 128, 64, 4, Y_train.shape[0]])

    print("-----Model Accuracy Before Training-----")
    print("On Training Data:")
    model.accuracy(X_train, Y_train)

    print("On Testing Data:")
    model.accuracy(X_test, Y_test)

    model.train(X=X_train, Y=Y_train, learning_rate=0.0075, num_iterations=3000, print_cost=True)

    print("-----Model Accuracy After Training-----")
    print("On Training Data:")
    model.accuracy(X_train, Y_train)

    print("On Testing Data:")
    model.accuracy(X_test, Y_test)

if __name__ == '__main__':
    main()
