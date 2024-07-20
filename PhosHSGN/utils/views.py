# Drawing function
import os

import numpy as np
from matplotlib import pyplot as plt


def plot_loss(train_loss, val_loss, name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'images/{name}.jpg')
    plt.cla()