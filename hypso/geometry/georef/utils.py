import math as m
import numpy as np
from typing import Tuple


def gen_resample_grid(grid_res_x, grid_res_y, bbox_coord_limits) -> Tuple[np.ndarray, list]:
    """
    Generate resample grid to the bbox coordinate limits

    :param grid_res_x:
    :param grid_res_y:
    :param bbox_coord_limits: is expected to be a list or tuple with 4 elements as follows: [xmin, xmax, ymin, ymax]

    :return:
    """

    grid_N_x = int((bbox_coord_limits[1] - bbox_coord_limits[0]) / grid_res_x)
    grid_N_y = int((bbox_coord_limits[3] - bbox_coord_limits[2]) / grid_res_y)

    x = np.linspace(bbox_coord_limits[0] + grid_res_x / 2, bbox_coord_limits[1] - grid_res_x / 2, grid_N_x)
    y = np.linspace(bbox_coord_limits[2] + grid_res_y / 2, bbox_coord_limits[3] - grid_res_y / 2, grid_N_y)

    # np.mgrid[] works only with pure integer arguments/indices
    # grid_x, grid_y = np.mgrid[0:1:10j, 0:1:20j] # works
    # grid_x, grid_y = np.mgrid[x_min:x_max:grid_N_x, y_min:y_max:grid_N_y] # does not

    grid_x, grid_y = np.meshgrid(x, y)
    grid = np.array([grid_x, grid_y]).transpose([1, 2, 0])

    grid_dims = [grid_N_x, grid_N_y]
    return grid, grid_dims

    # Old, manual, slower version (for loops bad)
    # grid_N_x = m.floor((bbox_coord_limits[1] - bbox_coord_limits[0])/grid_res_x)
    # grid_N_y = m.floor((bbox_coord_limits[3] - bbox_coord_limits[2])/grid_res_y)
    # grid = np.zeros((grid_N_y, grid_N_x, 2))
    # for i in range(grid_N_y):
    #    for j in range(grid_N_x):
    #        grid[i,j,0] = grid_res_x/2 + bbox_coord_limits[0] + j*grid_res_x
    #        grid[i,j,1] = grid_res_y/2 + bbox_coord_limits[2] + i*grid_res_y
    # grid_dims = [grid_N_x, grid_N_y]
    # return grid, grid_dims


def gen_resample_grid_bbox_min(grid_res_x, grid_res_y, bbox_minimal) -> Tuple[np.ndarray, list]:
    """
    Generate resample grid bbox minimum

    :param grid_res_x:
    :param grid_res_y:
    :param bbox_minimal: shall have has shape [4,2] and consists of four points describing the corner of the minimal
        bounding box.\n
        The width of the box is taken as the distance between point [0,:] and [3,:].\n
        The height of the box is taken as the distance between point [2,:] and [3,:].\n
        The x direction is taken as the direction going from point [3,:] to point [0,:].\n
        The y direction is taken as the direction going from point [3,:] to point [2,:].\n

    :return:
    """

    tr = bbox_minimal[0, :]  # tr = top right
    tl = bbox_minimal[3, :]  # tl = top left
    bl = bbox_minimal[2, :]  # bl = bot left

    # print('top right', tr)
    # print( 'top left', tl)
    # print( 'bot left', bl)

    bbox_width = m.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    bbox_height = m.sqrt((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2)

    # print(' width:', bbox_width)
    # print('height:', bbox_height)

    grid_x = m.floor(bbox_width / grid_res_x)
    grid_y = m.floor(bbox_height / grid_res_y)

    x_dir = (tr - tl) / bbox_width
    y_dir = (bl - tl) / bbox_height

    # print('x dir:', x_dir)
    # print('y dir:', y_dir)

    grid = np.zeros((grid_y, grid_x, 2))

    for i in range(grid_y):
        grid_point_x = tl[0] + y_dir[0] * i * grid_res_y
        grid_point_y = tl[1] + y_dir[1] * i * grid_res_y
        for j in range(grid_x):
            grid_point_x_ = grid_point_x + x_dir[0] * j * grid_res_x
            grid_point_y_ = grid_point_y + x_dir[1] * j * grid_res_x
            grid[i, j, 0] = grid_point_x_
            grid[i, j, 1] = grid_point_y_

    grid_dims = [grid_x, grid_y]

    return grid, grid_dims
