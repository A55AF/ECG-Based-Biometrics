import os
import numpy as np
import matplotlib.pyplot as plt

from utils import train, test


def main():
    clf = train()
    predictions = test(clf)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
