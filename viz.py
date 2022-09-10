import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # with open("./test.txt", "r") as file:
    #     all_stuff = file.readlines()
    #     unfilter_pts = np.array([float(x[:-1]) for x in all_stuff])
    with open("./output.txt", "r") as file:
        all_stuff = file.readlines()
        filter_pts = np.array([float(x[:-1]) for x in all_stuff])
        
    xs = np.arange(len(filter_pts))
    plt.plot(xs, filter_pts, c = 'r', label = 'filtered')
    # plt.plot(xs, unfilter_pts, c = 'b', label = 'unfiltered')
    plt.legend()
    plt.grid(axis = 'both')
    plt.show()