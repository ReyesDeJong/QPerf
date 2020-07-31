import os, sys
import matplotlib.pyplot as plt

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import numpy as np

if __name__ == "__main__":
    data_1 = 2
    data_1_path = os.path.join(PROJECT_PATH, 'deep_learning', 'analysis',
                               'data', '%i' % data_1, 'Gd_resized_training_norm.npy')
    data_1 = np.load(data_1_path)
    print(data_1.shape)

    i_1 = 9
    i_2 = 2
    plt.imshow(data_1[..., i_1, i_2], cmap='gray')
    plt.show()
