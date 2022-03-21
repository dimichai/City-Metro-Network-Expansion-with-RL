#%%
import argparse
import os
import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcolors
import pandas as pd
from collections import defaultdict
import csv
import torch

exist_line_num = 2
line_full_tensor = [
                    torch.tensor([[234], [293],[294],[295],[325],[326],[357],[358],[359],[360],[361],[362],[363],[364],[365],[366],[367],[368],[341],[342],[343],[344   ]]), 
                    torch.tensor([[ 13],[ 43],[ 72],[101],[130],[159],[188],[217],[246],[275],[304],[333],[362],[391],[420],[449],[478],[507],[536],[565],[594],[623],[652],[681],[710],[739],[768]])
                ]

line_station_list = [[234, 293, 295, 325, 326, 357, 359, 360, 361, 362, 363, 364, 365, 366, 368, 341, 342, 343, 344], 
                    [13, 43, 101, 130, 159, 188, 246, 275, 304, 362, 391, 420, 449, 478, 507, 536, 594, 623, 681, 710, 768]]


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

def agent_exist_line_pair1(tour_idx_cpu, agent_grid_list, per_line_full_tensor, per_line_station_list):
    satisfied_od_pair = []

    agent_line = (tour_idx_cpu - per_line_full_tensor)

    intersection_need = (agent_line == 0).nonzero()

    if intersection_need.size()[0] == 0:
        pass # there is no interaction

    else:
        interaction_index_mult = intersection_need[:, 1]
        interaction_index_list = []
        for i in interaction_index_mult:
            interaction_index_list.append(agent_grid_list[i])

        for i in agent_grid_list:
            if i not in interaction_index_list:
                for j in per_line_station_list:
                    if j not in interaction_index_list:
                        per_od_pair = []
                        per_od_pair.append(i)
                        per_od_pair.append(j)
                        satisfied_od_pair.append(per_od_pair)

    return satisfied_od_pair # for each element: the agent station is the first


def satisfied_od_mask(tour_idx, grid_x_max, grid_y_mask):
    # output--satisfied_od_pair:   [[1,2],[2,3]]

    # agent_grid_list = tour_idx[0].tolist()

    satisfied_od_pair1 = []

    for i in range(len(tour_idx) - 1):
        for j in range(i + 1, len(tour_idx)):
            per_od_pair = []
            per_od_pair.append(tour_idx[i])
            per_od_pair.append(tour_idx[j])
            satisfied_od_pair1.append(per_od_pair)

    satisfied_od_pair2 = []
    for i in range(exist_line_num):
        per_line_full_tensor = line_full_tensor[i]
        per_line_station_list = line_station_list[i]

        per_satisfied_od_pair2 = agent_exist_line_pair1(torch.tensor([[tour_idx]]), tour_idx, per_line_full_tensor, per_line_station_list)
        
        satisfied_od_pair2 = satisfied_od_pair2 + per_satisfied_od_pair2

    satisfied_od_pair = satisfied_od_pair1 + satisfied_od_pair2
    satisfied_od_mask = np.zeros((grid_x_max * grid_y_mask, grid_x_max * grid_y_mask))

    for pair in satisfied_od_pair:
        i, j = pair

        satisfied_od_mask[i][j] = 1

    return satisfied_od_mask

def sum_of_diffs(x):
    """Returns the sum of absolute differences between all elements in the given array

    Args:
        x (np.array): input array

    Returns:
        int: sum of absolute differences
    """
    return np.absolute(sum(x - np.reshape(x, (len(x), 1)))).sum()

def ggi(x, w_denom):
    weights = np.array([1/(w_denom**i) for i in range(x.shape[0])])
    # "Normalize" weights to sum to 1
    weights = weights/weights.sum()

    return np.sum(np.sort(x) * weights)

def gen_line_plot_grid(lines, grid_x_max, grid_y_max):
    """Generates a grid_x_max * grid_y_max grid where each grid is valued by the frequency it appears in the generated lines.
    Essentially creates a grid of the given line to plot later on.

    Args:
        line (list): list of generated lines of the model
        grid_x_max (_type_): _description_
        grid_y_mask (_type_): _description_
    """
    data = np.zeros((grid_x_max, grid_y_max))

    for line in lines:
        line_g = v_to_g(np.array(line), grid_x_max, grid_y_max)
        for g in line_g:
            data[g] += 1
    
    data = data/len(lines)

    return data

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

    # read generated lines from output file
    gen_lines = []
    # get all covered squares from the generated lines
    unique_squares = []
    with open(output_loc, 'r') as f:
        for row in csv.reader(f):
            line = []
            for s in row:
                s = int(s)
                line.append(s)
                if s not in unique_squares:
                    unique_squares.append(s)
            
            gen_lines.append(line)

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

    # Create ses dataframe.
    df_ses = pd.DataFrame(ses).set_index('v')

    # Plot the distribution of House Prices
    price_values = np.fromiter(ses['ses'], dtype=float)
    price_normalised =  (price_values - price_values.mean()) / price_values.std()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(price_normalised, bins=20)
    fig.suptitle('Xi’an, China - Distribution of average house price (RMB) - Normalised')
    fig.savefig(os.path.join(constants.WORKING_DIR, 'index_average_price_distr_norm.png'))

    # Plot the average generated line (from the multiple sample generated lines)
    plot_grid = gen_line_plot_grid(gen_lines, args.grid_x_max, args.grid_y_max)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(plot_grid)
    fig.suptitle(f'Xi’an, China - Average Generated line \n from {args.model_folder}')
    fig.savefig(os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'average_generated_line.png'))
    
    # Plot the distribution of covered (by the generated lines) vs non-covered squares by house prices.
    covered_grid_prices = df_ses.loc[np.isin(df_ses.index, unique_squares)]['ses'].values
    non_covered_grid_prices = df_ses.loc[~np.isin(df_ses.index, unique_squares)]['ses'].values

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(df_ses['ses'].values, bins=20)
    fig.suptitle('Xi’an, China - Distribution of average house price (RMB)')
    fig.savefig(os.path.join(constants.WORKING_DIR, 'index_average_price_distr.png'))

    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(df_ses['ses'].min(), df_ses['ses'].max(), 20)
    ax.hist(covered_grid_prices, alpha=0.5, density=True, bins=bins, label='covered')
    ax.hist(non_covered_grid_prices, alpha=0.5, density=True, bins=bins, label='not-covered')
    ax.axvline(covered_grid_prices.mean(), color='C0', linestyle='dashed', linewidth=1)
    ax.axvline(non_covered_grid_prices.mean(), color='C1', linestyle='dashed', linewidth=1)
    ax.legend()
    fig.suptitle(f'Xi’an, China - Distribution of average house price (RMB) \n generated line from {args.model_folder}', fontsize=10)
    fig.savefig(os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'index_average_price_distr_by_coverage.png'))
    
    # Plot the equity of the generated line
    # Splits SES into 5 bins of equal size.
    df_ses['ses_bin'] = pd.qcut(df_ses['ses'], 5, labels=False)

    # Create overall origin destination flow matrix.
    # Note we already load the masked OD here so no need to mask out satisfied OD from current lines.
    od_mx = build_od_matrix(args.grid_x_max * args.grid_y_max, args.od_index_path)

    # Create group specific masks and OD matrices:
    group_masks, group_od = [], []
    for g in np.sort(df_ses['ses_bin'].unique()):
        # Create a mask [0, 1] for each SES bin, which will be used to filter in only the OD pairs that start from each group.
        # We do this because we need to calculate total and satisfied OD demand for each bin.
        g_mask = build_group_od_mask(list(df_ses[df_ses['ses_bin'] == g].index), args.grid_x_max, args.grid_y_max)
        group_masks.append(g_mask)
        # Multiply the overall OD marix with each group's mask, to create group-specific OD matrices.
        g_od = g_mask * od_mx
        # OD matrix is symmetric - therefore the total demand is divided by 2.
        g_od = g_od/2
        group_od.append(g_od)

    # group_sat_ods: total satisfied ODs that belong to either group
    # total_sat_ods: total satisfied ODs (some squares do not have a group)
    group_sat_ods, total_sat_ods = [], []
    sat_ods_by_group = []
    sat_ods_by_group_pct = []
    for line in gen_lines:
        # Mask to filter the OD matrix for satisfied OD flows of the generated line.
        sat_od_mask = satisfied_od_mask(line, args.grid_x_max, args.grid_y_max)
        group_satisfied_od, group_satisfied_od_pct = [], []

        for g in np.sort(df_ses['ses_bin'].unique()):
            # Multiply the satisfied OD mask with the group's OD matrix to get satisfied ODs per group.    
            g_sat_od = sat_od_mask * group_od[g]
            group_satisfied_od.append(g_sat_od)
            # Calculate the satisfied OD percentage per group.
            g_od_pct = np.round(g_sat_od.sum() / group_od[g].sum(), 3)
            # print(f'Group {g}: Total OD: {g_od.sum()} - Satisfied OD: {g_sat_od.sum()} - Fraction: {g_od_pct}')
            group_satisfied_od_pct.append(g_od_pct)
        
        g_total_sat_od = sum([g.sum() for g in group_satisfied_od])

        group_sat_ods.append(g_total_sat_od)
        total_sat_ods.append((sat_od_mask * od_mx).sum())
        sat_ods_by_group.append([g.sum() for g in group_satisfied_od])
        sat_ods_by_group_pct.append(group_satisfied_od_pct)

        # print(f'Total satisfied OD: {(sat_od_mask * od_mx).sum()} - Total satisfied group OD: {g_total_sat_od}')

    mean_sat_od_by_group = np.array(sat_ods_by_group).mean(axis=0)
    mean_sat_od_by_group_pct = np.array(sat_ods_by_group_pct).mean(axis=0)

    ggi_od = ggi(mean_sat_od_by_group, 2)
    ggi_od_pct = ggi(mean_sat_od_by_group_pct, 2)
    
    # mean_diff_1 = np.array([sum_of_diffs(g) for g in sat_ods_by_group]).mean()
    # for i in range(sat_ods_by_group[0]):
    #     j = i + 1
    #     try:

    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.bar(range(5), mean_sat_od_by_group_pct)
    # fig.suptitle(f'Xi’an, China - Mean Satisfied OD % by house price bin \n Total OD: {round(100*sum(total_sat_ods)/128/od_mx.sum(), 2)}% - Total group OD: {round(100*sum(group_sat_ods)/128/sum([g.sum() for g in group_od]), 2)}% (GGI:{round(ggi_od_pct, 6)}) \n New Line generated from {args.model_folder}')
    # fig.savefig(os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'satisfied_od_by_group_pct.png'))

    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.bar(range(5), mean_sat_od_by_group)
    # fig.suptitle(f'Xi’an, China - Mean Satisfied OD by house price bin \n Total OD: {round(sum(total_sat_ods)/128, 2)} - Total group OD: {round(sum(group_sat_ods)/128, 2)} (GGI:{round(ggi_od, 2)}) \n New Line generated from {args.model_folder}')
    # fig.savefig(os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'satisfied_od_by_group.png'))

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].bar(range(5), mean_sat_od_by_group)
    axs[0].title.set_text(f'Xi’an, China - Mean Satisfied OD by house price bin \n Total OD: {round(sum(total_sat_ods)/128, 2)} - Total group OD: {round(sum(group_sat_ods)/128, 2)} (GGI:{round(ggi_od, 2)}) \n New Line generated from {args.model_folder}')
    axs[1].bar(range(5), mean_sat_od_by_group_pct)
    axs[1].title.set_text(f'Xi’an, China - Mean Satisfied OD % by house price bin \n Total OD: {round(100*sum(total_sat_ods)/128/od_mx.sum(), 2)}% - Total group OD: {round(100*sum(group_sat_ods)/128/sum([g.sum() for g in group_od]), 2)}% (GGI:{round(ggi_od_pct, 6)}) \n New Line generated from {args.model_folder}')

    fig.savefig(os.path.join(constants.WORKING_DIR, 'result', args.model_folder, 'satisfied_od_by_group_joint.png'))
    # Plot the distribution of house prices for multiple models together.
    # model_folders = ['16_22_50.580720', '21_15_26.123481']

    # fig, ax = plt.subplots(figsize=(5, 5))

    # for i, m in enumerate(model_folders):
    #     with open(os.path.join(constants.WORKING_DIR, 'result', m, 'tour_idx_multiple.txt'), 'r') as f:
    #         tour_idx = f.readline()

    #     tour_idx = np.array(tour_idx.split(','), dtype=np.int64)
    #     covered_grid_prices = df_ses.loc[np.isin(df_ses.index, tour_idx)]['ses'].values

    #     bins = np.linspace(df_ses['ses'].min(), df_ses['ses'].max(), 20)
    #     ax.hist(covered_grid_prices, alpha=0.3, density=True, bins=bins, label=m, color=mcolors.tab10(i))
    #     ax.axvline(covered_grid_prices.mean(), linestyle='dashed', linewidth=1, color=mcolors.tab10(i))
    #     ax.legend()
    #     fig.suptitle(f'Xi’an, China - Distribution of average house price (RMB) \n generated line from {args.model_folder}', fontsize=10)
    # plt.show()




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
