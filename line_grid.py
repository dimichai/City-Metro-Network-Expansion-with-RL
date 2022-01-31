#%%
import numpy as np
import matplotlib.pyplot as plt
import random

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

line_new = [0, 1, 4, 5]
line_new = np.array(line_new)
line_new = v_to_g(line_new)

for g in line_0:
    data[g] = 1

for g in line_1:
    data[g] = 2

for g in line_new:
    data[g] = 3

plt.figure(figsize=(5,5))
plt.imshow(data)
plt.show()

# %%
