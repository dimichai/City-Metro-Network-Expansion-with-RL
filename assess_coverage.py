import argparse
import os
import constants
import numpy as np
import matplotlib.pyplot as plt


def v_to_g(index, grid_x_max, grid_y_max):
    grid_x = index // grid_x_max
    grid_y = index % grid_y_max

    return list(zip(grid_x, grid_y))

def g_to_v(gx, gy, grid_x_max):
    index = int(gx) * grid_x_max + int(gy)

    return index

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
            # convert grid index string to tuple
            gx = g.split(',')[0]
            gy = g.split(',')[1]

            v_idx = g_to_v(gx, gy, args.grid_x_max)

            ses[v_idx] = float(s)


    covered_grid_prices = [ses[v] for v in tour_idx if v in ses]
    non_covered_grid_prices = [ses[v] for v in ses.keys() if v not in tour_idx]

    # print("25% - " + str(np.percentile(ses_values, 10)))
    # print("75% - " + str(np.percentile(ses_values, 80)))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(ses.values(), bins=20)
    fig.suptitle('Xi’an, China - Distribution of average house price (RMB)')
    fig.savefig(os.path.join(constants.WORKING_DIR, 'index_average_price_distr.png'))

    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(np.fromiter(ses.values(), dtype=float).min(), np.fromiter(ses.values(), dtype=float).max(), 20)
    ax.hist(covered_grid_prices, alpha=0.5, density=True, bins=bins, label='covered')
    ax.hist(non_covered_grid_prices, alpha=0.5, density=True, bins=bins, label='not-covered')
    ax.legend()
    fig.suptitle('Xi’an, China - Distribution of average house price (RMB) \n generated line', fontsize=10)
    fig.savefig(os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'index_average_price_distr_by_coverage_2.png'))

