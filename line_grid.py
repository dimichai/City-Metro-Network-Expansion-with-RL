#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


grid_x_max = 29
grid_y_max = 29

def v_to_g(index):
    grid_x = index // grid_x_max
    grid_y = index % grid_y_max

    return list(zip(grid_x, grid_y))

data = [[0 for j in range(grid_x_max)] for i in range(grid_y_max)]

data = np.array(data)

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


# create colormap and legend
values = np.unique(data[data != 0].ravel())

plt.figure(figsize=(5,5))
im = plt.imshow(data)

colors = [ im.cmap(im.norm(value)) for value in values]
labels = ["Current Lines", "New Line Util", "New Line Util + Equity", "New Line Equity"]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(values)) ]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.title("Xi'an China - Existing & Generated Metro Lines")

plt.show()

# %%
