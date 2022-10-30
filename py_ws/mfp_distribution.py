import numpy as np
import matplotlib.pyplot as plt

SCALE = 9.9999999999e-1

if __name__ == '__main__':
    sample_num = 200000
    samples = np.random.rand(sample_num) * SCALE + (1. - SCALE)
    mfp_samples = -np.log(samples)

    plt.hist(mfp_samples, np.linspace(0, 5, 500))
    print("Proba for not being bigger than 1.0:", (mfp_samples < 1.0).sum() / sample_num)
    plt.show()
