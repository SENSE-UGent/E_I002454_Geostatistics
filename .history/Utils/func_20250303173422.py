import numpy as np

#  This file contains the functions that are used in the main file

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

def inverse_distance_weighting3D_(data, grid_points, power, max_points, min_points, search_radii):
    
    data = data.dropna(subset='o')
    data_full = data[['x', 'y', 'z', 'o']].values
    data_coords = data[['x', 'y', 'z']].values
    grid_coords = grid_points[['x', 'y', 'z']].values

    p_list = []

    for i in range(len(grid_coords)):

        grid_point = grid_coords[i]
        sorted_indices, closest_distances, sorted_points, within_ellipsoid_mask = is_within_ellipsoid_rework_2(
                data_coords,
                grid_point,
                search_radii
            )
        
        # print("valid neighbours mask", valid_neighbors_mask)

        near_data = sorted_points
        o_values_ellips = data_full[within_ellipsoid_mask, 3]
        o_values = o_values_ellips[sorted_indices]
        # print("near_data", near_data)
        n_near_data = len(near_data)
        # print("n_near_data", n_near_data)
        
        if n_near_data < min_points:
            
            # If no sufficient valid neighbors found, append NaN value
            p_list.append(np.nan)
            print("warning: insufficient points, a nan value was returned")

        else:
            selected_indices = list(range(min(sorted_points.shape[0], max_points)))

            closest_distances = closest_distances[selected_indices]
             #extract the o_values from data using the valid_neighbors_indices
            o_values = o_values[selected_indices] # should be good
            # print("o_values", o_values)

            # print("power", power)

            # Calculate the inverse squared distances with angle-dependent power
            inv_powered_distances = np.where(closest_distances != 0, 1 / (closest_distances ** power), 0)
            # print("inv_powered_distances", inv_powered_distances)

            # Normalize the weights
            sum_weights = np.nansum(inv_powered_distances)

            if np.isnan(sum_weights):
                print("Warning: A NaN weight was assigned")

            normalized_weights = inv_powered_distances / sum_weights

            # Calculate the weighted average
            weighted_average = np.ma.average(o_values, weights=normalized_weights)

            # Handle non-float results
            if not isinstance(weighted_average, float):

                weighted_average = np.nan
                print("Warning: A non-float value was averaged out")

            p_list.append(weighted_average)

            if np.isnan(weighted_average):
                print("Warning: A NaN value was averaged out")


    grid_points['p'] = p_list

    grid_points['p'].replace('--', np.nan, inplace=True)


def inverse_distance_weighting2D(data, grid_points, 
                                 data_x_col, data_y_col, data_o_col, 
                                 grid_x_col, grid_y_col, 
                                 power, max_points, min_points, search_radius):
    
    # Iterate over each grid point
    for row in grid_points.iterrows():
        # Calculate distances from data points to the current grid point
        distances = np.sqrt((data[data_x_col] - row[1][grid_x_col])**2 
                            + (data[data_y_col] - row[1][grid_y_col])**2)
        
        # Check for NaN distances
        if np.isnan(distances).any():
            print("Warning: Distance value is nan.")
            
        # Create a DataFrame with distances and original data values
        data_dist = data.copy()
        data_dist['dist'] = distances
        print('dist', data_dist['dist'].describe()) 
        
        # Check for NaN values in distances
        num_nan_values = data_dist['dist'].isnull().sum()
        if num_nan_values > 0:

            print('Number of nan values in column dist:', num_nan_values)
        
        # Sort by distances and select the closest points
        data_dist = data_dist.sort_values(by=['dist']).iloc[:max_points]
        
        # If the minimum distance is zero, assign the value directly
        if data_dist['dist'].min() == 0:

            grid_points.at[row[0], 'p'] = data_dist[data_o_col].min()

        else:
            # Check if the minimum points requirement is satisfied
            if len(data_dist.loc[data_dist['dist'] < search_radius]) < min_points:

                grid_points.at[row[0], 'p'] = np.nan
                print('Warning: a nan value was returned because the min_points was not satisfied. ' 
                      + str(len(data_dist.loc[data_dist['dist'] < search_radius])))
                
            else:

                # Calculate the sum of distances
                dsum = data_dist['dist'].sum()
                
                # Calculate weights and normalized weights
                data_dist['weight'] = 1 / (data_dist['dist']**power)
                data_dist['norm_weight'] = data_dist['weight'] / (dsum**power)
                
                # Calculate the weighted average
                weighted_avg = np.average(data_dist[data_o_col], weights=data_dist['norm_weight'])
                
                # Round the weighted average to two decimal places
                num_decimal_places = 2
                rounded_avg = round(weighted_avg, num_decimal_places)
                
                # Update the grid point with the rounded average
                grid_points.at[row[0], 'p'] = rounded_avg

                # Inform the user that the function has finished and where results are
    print('The function has finished. The predictions are in the grid_points DataFrame in the column p.')
                                       
    return #row[1]['p'] np.isnan()

def inverse_distance_weighting2D_beta(data, grid_points, power, max_points, min_points, search_radius):
        # Calculate distance of data points from prediction point and normalised distances for radii
    for row in grid_points.iterrows():

        distances = np.sqrt((data.x - row[1]['x'])**2 
                            + (data.y - row[1]['y'])**2
                            )
        
        if np.isnan(distances).any():
            print("Warning: Distance value is nan.")
            
        # Create dataframe containing distances and o
        data_dist = data.copy()
        data_dist['dist'] = distances
        #Find the number of nan values in column dist
        num_nan_values = data_dist['dist'].isnull().sum()
        #Print the number of nan values if it is larger than 0
        
        if num_nan_values > 0:
            print('Number of nan values in column dist:', num_nan_values)
            
        # # Find the number of nan values in column dist
        # num_nan_values = data_dist['norm_dist'].isnull().sum()
        # #Print the number of nan values if it is larger than 0
        
        # if num_nan_values > 0:
        #     print('Number of nan values in column norm_dist:', num_nan_values)
        
        # Sort dataframe by distances and select only the closest ones
        data_dist = data_dist.sort_values(by=['dist']).iloc[:max_points]
        #start iterating
        # Assign data value from 'data' at 'grid_df' datapoints if the distance is equal to zero
        
        if data_dist['dist'].min() == 0:
            grid_points.at[row[0], 'p'] = data_dist['o'].min()
            
        else:
            #Check if min_points is satisfied and set z_grid to nan if it is not
            
            if len(data_dist.loc[data_dist['dist'] < search_radius]) < min_points:

                grid_points.at[row[0], 'p'] = np.nan
                print('Warning: a nan value was returned because the min_points was not satisfied. ' 
                      + str(len(data_dist.loc[data_dist['dist'] < search_radius])))
                
            else:
                # calculate sum of distances
                dsum = data_dist['dist'].sum()
            # Calculate weighted average of o
                # Calculate weights
                data_dist['weight'] = 1 / (data_dist['dist']**power)
                data_dist['norm_weight'] = data_dist['weight'] / (dsum**power)

                weighted_avg = np.average(data_dist['o'], weights=data_dist['norm_weight'])

                # Round the output float to the same precision as data_dist['o']
                num_decimal_places = 2
                rounded_avg = round(weighted_avg, num_decimal_places)

                # Update the grid_points dataframe with the rounded average
                grid_points.at[row[0], 'p'] = rounded_avg
                                       
    return #row[1]['p'] np.isnan()