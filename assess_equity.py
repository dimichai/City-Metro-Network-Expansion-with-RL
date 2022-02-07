#%%
import argparse
import os
import constants
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


def v_to_g(index, grid_x_max, grid_y_max):
    grid_x = index // grid_x_max
    grid_y = index % grid_y_max

    return list(zip(grid_x, grid_y))

def g_to_v(gx, gy, grid_x_max):
    index = int(gx) * grid_x_max + int(gy)

    return index

def build_od_matrix(grid_num, od_index_path):
    
    od_matrix = np.zeros((grid_num, grid_num))

    f = open(od_index_path, 'r')
    for line in f:
        index1, index2, weight = line.rstrip().split('\t')
        index1 = int(index1)
        index2 = int(index2)
        weight = float(weight)

        od_matrix[index1][index2] = weight
    f.close()

    od_matrix =  od_matrix / od_matrix.max()   

    return od_matrix

def build_group_od_mask(origin_vs, grid_x_max, grid_y_max):
    """Creates mask with the OD dimensions (grid_num = grid_x_max * grid_y_max).
    Assigns 0 to the cells for which the origin square does not belong to the given group.
    Assigns 1 to the cels for which the origin square belongs to the given group. 
    This mask will be later used to filter satisfied OD by groups.
    Args:
        grid_x_max/grid_y_max ([int]): total number of cells on the x/y axis 
        origin_vs (list): the vector indices of the origin cells that belong to the group for which the mask is being created.

    """
    mask = np.zeros((grid_x_max * grid_y_max, grid_x_max * grid_y_max))

    for i in origin_vs:
        mask[i][:] = 1

    return mask


def satisfied_od_mask(tour_idx, grid_x_max, grid_y_mask):
    # output--satisfied_od_pair:   [[1,2],[2,3]]

    satisfied_od_pair = []

    for i in range(len(tour_idx) - 1):
        for j in range(i + 1, len(tour_idx)):
            per_od_pair = []
            per_od_pair.append(tour_idx[i])
            per_od_pair.append(tour_idx[j])
            satisfied_od_pair.append(per_od_pair)

    satisfied_od_mask = np.zeros((grid_x_max * grid_y_mask, grid_x_max * grid_y_mask))

    for pair in satisfied_od_pair:
        i, j = pair

        satisfied_od_mask[i][j] = 1

    return satisfied_od_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assess coverage of generated line')

    # parser.add_argument('--model_folder', required=True, type=str)
    parser.add_argument('--model_folder', default="21_15_26.123481", type=str)
    parser.add_argument('--od_index_path', default="od_index_masked.txt", type=str)
    parser.add_argument('--grid_x_max', default=29, type=int)
    parser.add_argument('--grid_y_max', default=29, type=int)

    args = parser.parse_args()

    # read the created line from the given model, which is saved as a list of indices in file tour_idx.txt
    # output_loc = os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'tour_idx.txt')
    output_loc = os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'tour_idx_multiple.txt')
    with open(output_loc, 'r') as f:
        tour_idx = f.readline()
    
    # convert the list of indices into a list of (x, y) pairs where x and y are the grid coordinates.
    tour_idx = np.array(tour_idx.split(','), dtype=np.int64)
    # tour_idx can be read as a series of created lines (e.g. if we run the model multiple times to account for randomness)
    # but we need the unique squares as well for calculating covered OD flows.
    tour_idx_unique = np.unique(tour_idx)
    # tour_g_idx = v_to_g(tour_idx, args.grid_x_max, args.grid_y_max)

    avg_price_loc = os.path.join(constants.WORKING_DIR, 'index_average_price.txt')
    ses = defaultdict(list)
    with open(avg_price_loc, 'r') as f:
        for line in f:
            g, s = line.rstrip().split('\t')
            # convert grid index string to tuple
            gx = g.split(',')[0]
            gy = g.split(',')[1]

            v_idx = g_to_v(gx, gy, args.grid_x_max)

            # ses[v_idx].append(float(s))
            ses['v'].append(v_idx)
            ses['gx'].append(gx)
            ses['gy'].append(gy)
            ses['ses'].append(float(s))
    
    # Create ses dataframe and create bins.
    df_ses = pd.DataFrame(ses).set_index('v')
    # Splits SES into 10 bins of equal size.
    df_ses['ses_bin'] = pd.qcut(df_ses['ses'], 5, labels=False)

    # Create overall origin destination flow matrix.
    od_mx = build_od_matrix(args.grid_x_max * args.grid_y_max, args.od_index_path)

    # Create a mask [0, 1] for each SES bin, which will be used to filter in only the OD pairs that start from each group.
    # We do this because we need to calculate total and satisfied OD demand for each bin.
    group_masks = []
    for g in np.sort(df_ses['ses_bin'].unique()):
        g_mask = build_group_od_mask(list(df_ses[df_ses['ses_bin'] == g].index), args.grid_x_max, args.grid_y_max)
        group_masks.append(g_mask)

    # Multiply the overall OD marix with each group's mask, to create group-specific OD matrices.
    group_od = []
    for g in np.sort(df_ses['ses_bin'].unique()):
        g_od = group_masks[g] * od_mx
        group_od.append(g_od)

    sat_od_mask = satisfied_od_mask(tour_idx_unique, args.grid_x_max, args.grid_y_max)
    
    group_satisfied_od = []
    for g in np.sort(df_ses['ses_bin'].unique()):
        g_od = sat_od_mask * group_od[g]
        group_satisfied_od.append(g_od)

    group_satisfied_od_total = []
    for g in np.sort(df_ses['ses_bin'].unique()):
        g_od_total = np.round(group_satisfied_od[g].sum() / group_od[g].sum(), 3)
        print(f'Group {g}: Total OD: {group_od[g].sum()} - Satisfied OD: {group_satisfied_od[g].sum()} - Fraction: {np.round(group_satisfied_od[g].sum() / group_od[g].sum(), 3)}')
        group_satisfied_od_total.append(g_od_total)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(range(5), group_satisfied_od_total)
    fig.suptitle(f'Xi’an, China - Satisfied OD Percentage by income bin \n New Line generated from {args.model_folder}')
    fig.savefig(os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'satisfied_od_by_group.png'))

    # price_values = np.fromiter(ses.values(), dtype=float)
    # price_normalised =  (price_values - price_values.mean()) / price_values.std()
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.hist(price_normalised, bins=20)
    # fig.suptitle('Xi’an, China - Distribution of average house price (RMB) - Normalised')
    # fig.savefig(os.path.join(constants.WORKING_DIR, 'index_average_price_distr_norm.png'))

    

    # covered_grid_prices = np.array([ses[v] for v in tour_idx if v in ses])
    # non_covered_grid_prices = np.array([ses[v] for v in ses.keys() if v not in tour_idx])

    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.hist(ses.values(), bins=20)
    # fig.suptitle('Xi’an, China - Distribution of average house price (RMB)')
    # fig.savefig(os.path.join(constants.WORKING_DIR, 'index_average_price_distr.png'))

    # fig, ax = plt.subplots(figsize=(5, 5))
    # bins = np.linspace(np.fromiter(ses.values(), dtype=float).min(), np.fromiter(ses.values(), dtype=float).max(), 20)
    # ax.hist(covered_grid_prices, alpha=0.5, density=True, bins=bins, label='covered')
    # ax.hist(non_covered_grid_prices, alpha=0.5, density=True, bins=bins, label='not-covered')
    # ax.axvline(covered_grid_prices.mean(), color='C0', linestyle='dashed', linewidth=1)
    # ax.axvline(non_covered_grid_prices.mean(), color='C1', linestyle='dashed', linewidth=1)
    # ax.legend()
    # fig.suptitle(f'Xi’an, China - Distribution of average house price (RMB) \n generated line from {args.model_folder}', fontsize=10)
    # fig.savefig(os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'index_average_price_distr_by_coverage.png'))


# %% Diagnostic Plots
# grid_x_max = 29
# grid_y_max = 29
# od_index_path = './od_index_masked.txt'

# # Plot Origin Destination Matrix
# od = build_od_matrix(grid_x_max * grid_y_max, od_index_path)
# plt.figure(figsize=(10,10))
# im = plt.spy(od)
# plt.colorbar()


# %%
