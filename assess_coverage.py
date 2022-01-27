import argparse
import os
import constants
import numpy as np
import matplotlib.pyplot as plt


def v_to_g(index, grid_x_max, grid_y_max):
    grid_x = index // grid_x_max
    grid_y = index % grid_y_max

    return list(zip(grid_x, grid_y))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assess coverage of generated line')

    parser.add_argument('--model_folder', required=True, type=str)
    parser.add_argument('--grid_x_max', default=29, type=int)
    parser.add_argument('--grid_y_max', default=29, type=int)

    args = parser.parse_args()

    # read the created line from the given model, which is saved as a list of indices in file tour_idx.txt
    output_loc = os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'tour_idx.txt')
    with open(output_loc, 'r') as f: 
        tour_idx = f.readline()
    
    # convert the list of indices into a list of (x, y) pairs where x and y are the grid coordinates.
    tour_idx = np.array(tour_idx.split(','), dtype=np.int64)
    tour_g_idx = v_to_g(tour_idx, args.grid_x_max, args.grid_y_max)

    avg_price_loc = os.path.join(constants.WORKING_DIR, 'index_average_price.txt')
    ses = {}
    with open(avg_price_loc, 'r') as f:
        for line in f:
            g, s = line.rstrip().split('\t')
            ses[g] = float(s)

    print(ses)
    print(ses.values())
    plt.plot(ses.values())
