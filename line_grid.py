#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import constants


grid_x_max = 29
grid_y_max = 29
od_index_path = './od_index_masked.txt'

def v_to_g(index):
    grid_x = index // grid_x_max
    grid_y = index % grid_y_max

    return list(zip(grid_x, grid_y))

#%% Plot average house price in the grid.
data = np.zeros((grid_x_max, grid_y_max))

avg_price_loc = os.path.join(constants.WORKING_DIR, 'index_average_price.txt')
with open(avg_price_loc, 'r') as f:
    for line in f:
        g, s = line.rstrip().split('\t')
        # convert grid index string to tuple
        gx = int(g.split(',')[0])
        gy = int(g.split(',')[1])
        s = float(s)

        data[gx][gy] = s

plt.figure(figsize=(10,10))
im = plt.imshow(data, cmap='coolwarm')
plt.clim(0, 12000)
plt.colorbar()
plt.title("Xi'an - Average house price")
plt.savefig(os.path.join(constants.WORKING_DIR, 'index_average_price_grid.png'))

#%% Plot existing and generated lines in the grid
data = np.zeros((grid_x_max, grid_y_max))

line_0 = [(8, 2), (10, 3), (10, 5), (11, 6), (11, 7), (12, 9), (12, 11), (12, 12), (12, 13),
            (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 20), (11, 22), (11, 23), (11, 24),
            (11, 25)]
line_1 = [(0, 13), (1, 14), (3, 14), (4, 14), (5, 14), (6, 14), (8, 14), (9, 14), (10, 14),
            (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14), (20, 14), (21, 14),
            (23, 14), (24, 14), (26, 14)]

# only OD reward - no equity
line_utility = v_to_g(np.array([287,315,314,313,312,311,310,339,338,367,396,395,394,393,392,391,390,419,448,
            477,476,505,534,563,562,561,590,589,588,587,616,645,644,643,672,671,670,699,
            757,785,812]))

# equal weights on OD and equity
line_equal = v_to_g(np.array([641,613,584,585,586,587,588,559,530,531,502,503,504,505,476,447,418,419,390,391,
            392,393,394,365,336,307,308,279,250,221,222,223,224,195,196,197,198,199,200,171,
            172,173,144,115,86]))

# only equity reward - no OD
line_equity = v_to_g(np.array([732,733,704,675,646,617,588,589,590,591,592,593,594,595,566,567,538,509,510,481,452,453,424,395,396,367,338,339,310,281,252,223,224,225,196,197,198,199,201,202,144,115,86,57,28]))
# line_equity_dislim = v_to_g(np.array([760,704,675,646,617,588,589,590,591,592,593,564,565,566,537,538,509,510,481,482,453,424,395,396,367,338,339,310,281,252,223,224,225,196,197,168,169,171,173,144,115,86,57,28,789]))
line_equity_270 = v_to_g(np.array([87,116,145,174,203,232,233,234,263,264,265,294,323,324,353,354,383,384,413,414,443,444,473,474,503,504,505,506,507,508,537,538,567,568,597,598,599,628,629,658,659,688,689,718,719]))
line_equity_ac = v_to_g(np.array([835,808,810,781,782,753,724,695,666,637,608,550,492,434,405,347,318,260,231,202,173,144,86,57,28]))
line_od_5var = v_to_g(np.array([289,345,374,373,372,371,370,369,368,397,426,425,424,423,422,421,420,419,448,447,446,445,444,443,472,501,530,559,558,557,586,615,614,612,611,638,667,725,754,812]))

for g in line_0:
    data[g] = 1

for g in line_1:
    data[g] = 1

for g in line_utility:
    data[g] = 2

for g in line_equal:
    data[g] = 3

for g in line_equity:
    data[g] = 4

for g in line_od_5var:
    data[g] = 5

# for g in line_equity_dislim:
#     data[g] = 5

# create colormap and legend
values = np.unique(data[data != 0].ravel())

plt.figure(figsize=(5,5))
im = plt.imshow(data)

colors = [ im.cmap(im.norm(value)) for value in values]
labels = ["Current Lines", "New Line Util", "New Line Util + Equity", "New Line Equity", "New Line OD - 5*Var(OD)"]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(values)) ]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.title("Xi'an China - Existing & Generated Metro Lines")

plt.show()

# %% Exclude areas in the original OD matrix.
# Neighbor squares of the existing lines, whose OD should be considered satisfied because they are right next to the line squares. 
# line0_nei1 = [(8, 1), (9, 2), (10, 2), (11, 3), (11, 4), (11, 5), (12, 6), (12, 7), (12, 8), (13, 9), (13, 10),
#                 (13, 11), (13, 12), (13, 13)]
# line0_nei2 = [(13, 15), (13, 16), (13, 17), (13, 18),(13,19),(13, 20), (12, 21), (12, 22), (12, 23), (12, 24), (12, 25)]
# line0_nei3 = [(8, 3), (9, 4), (9, 5), (10, 6), (10, 7), (10, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13)]
# line0_nei4 = [(11,15), (11,16), (11,17), (11,18),(11, 19), (11, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25)]

# exclude_area = line0_nei1 + line0_nei2 + line0_nei3 + line0_nei4

# for g in exclude_area:
#     data[g] = 5
# %%
