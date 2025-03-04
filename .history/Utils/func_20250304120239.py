import numpy as np

#  This file contains the functions that are used in the main file

'''
is

'''

def is_within_ellipsoid_rework_2(points, center, search_radii):
    if points.shape[1] == 3:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        x_c, y_c, z_c = center
        x_dist = (x - x_c)**2
        y_dist = (y - y_c)**2
        z_dist = (z - z_c)**2
        squared_distances = x_dist + y_dist + z_dist
        ellipsoid_distances = x_dist/(search_radii[0]**2) + y_dist/(search_radii[1]**2) + z_dist/(search_radii[2]**2)
    else:
        x, y = points[:, 0], points[:, 1]
        x_c, y_c = center
        x_dist = (x - x_c)**2
        y_dist = (y - y_c)**2
        squared_distances = x_dist + y_dist
        ellipsoid_distances = x_dist/(search_radii[0]**2) + y_dist/(search_radii[1]**2)

    within_ellipsoid_mask = ellipsoid_distances <= 1
    points_within_ellipsoid = points[within_ellipsoid_mask]
    ellipsoid_distances = ellipsoid_distances[within_ellipsoid_mask]
    euclidean_distances = np.sqrt(squared_distances[within_ellipsoid_mask])
    sorted_indices = np.argsort(ellipsoid_distances)
    sorted_points = points_within_ellipsoid[sorted_indices]
    closest_distances = euclidean_distances[sorted_indices]
    
    return sorted_indices, closest_distances, sorted_points, within_ellipsoid_mask

def inverse_distance_weighting(data=np.nan, grid_points=np.nan, data_x_col=np.nan, data_y_col=np.nan, data_z_col=np.nan,
                                  grid_x_col=np.nan, grid_y_col=np.nan, grid_z_col=np.nan, o_col=np.nan, 
                                  power=np.nan, max_points=np.nan, min_points=np.nan, search_radii=np.nan):

    data = data.dropna(subset=[o_col])
    
    if data_z_col in data.columns and grid_z_col in grid_points.columns:

        data_full = data[[data_x_col, data_y_col, data_z_col, o_col]].values
        data_coords = data[[data_x_col, data_y_col, data_z_col]].values
        grid_coords = grid_points[[grid_x_col, grid_y_col, grid_z_col]].values
        is_3d = True

    else:

        data_full = data[[data_x_col, data_y_col, o_col]].values
        data_coords = data[[data_x_col, data_y_col]].values
        grid_coords = grid_points[[grid_x_col, grid_y_col]].values
        is_3d = False

    p_list = []

    for i in range(len(grid_coords)):

        grid_point = grid_coords[i]

        if is_3d:
            sorted_indices, closest_distances, sorted_points, within_ellipsoid_mask = is_within_ellipsoid_rework_2(
                data_coords,
                grid_point,
                search_radii)

        else:

            sorted_indices, closest_distances, sorted_points, within_ellipsoid_mask = is_within_ellipsoid_rework_2(
                data_coords,
                grid_point,
                search_radii[:2])

        near_data = sorted_points
        o_values_ellips = data_full[within_ellipsoid_mask, -1]
        o_values = o_values_ellips[sorted_indices]
        n_near_data = len(near_data)
        
        if n_near_data < min_points:

            p_list.append(np.nan)
            print("warning: insufficient points, a nan value was returned")

        else:

            selected_indices = list(range(min(sorted_points.shape[0], max_points)))
            closest_distances = closest_distances[selected_indices]
            o_values = o_values[selected_indices]

            inv_powered_distances = np.where(closest_distances != 0, 1 / (closest_distances ** power), 0)
            sum_weights = np.nansum(inv_powered_distances)

            if np.isnan(sum_weights):

                print("Warning: A NaN weight was assigned")

            normalized_weights = inv_powered_distances / sum_weights
            weighted_average = np.ma.average(o_values, weights=normalized_weights)

            if not isinstance(weighted_average, float):

                weighted_average = np.nan
                print("Warning: A non-float value was averaged out")

            p_list.append(weighted_average)

            if np.isnan(weighted_average):

                print("Warning: A NaN value was averaged out")

    grid_points['idw' + str(power) + o_col] = p_list
    grid_points['idw' + str(power) + o_col].replace('--', np.nan, inplace=True)
    print('The function has finished. The predictions are in the grid_points DataFrame in the column idw' + str(power) + o_col)