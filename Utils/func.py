import numpy as np
import scipy.spatial as sp  # for fast nearest neighbor search
from geostatspy.geostats import setup_rotmat, cova2, ksol_numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import xml.etree.ElementTree as ET


#  This file contains the functions that are used in the main file

'''
is_within_ellipsoid_rework_2: 
    This function is used to determine if a point is within an ellipsoid. It is a rework of the original function

inverse_distance_weighting:
    This function is used to calculate the inverse distance weighting of a point based on the surrounding points
'''

def is_within_ellipsoid_rework_2(points, center, search_radii):
    """
    Check if points are within an ellipsoid defined by the center and search radii. Written by Pablo De Weerdt.
    points: np.ndarray
        Array of points to check, shape (n_points, n_dimensions).
    center: tuple
        Center of the ellipsoid, shape (n_dimensions,).
    search_radii: tuple
        Radii of the ellipsoid, shape (n_dimensions,).
    Returns:
        sorted_indices: np.ndarray
            Indices of the points sorted by distance to the center of the ellipsoid.
        closest_distances: np.ndarray
            Distances of the points to the center of the ellipsoid.
        sorted_points: np.ndarray
            Points sorted by distance to the center of the ellipsoid.
        within_ellipsoid_mask: np.ndarray
            Boolean mask indicating which points are within the ellipsoid.
    """
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
    """
    Inverse distance weighting function. This function is used to calculate the inverse distance weighting of a point based on the surrounding points.
    Written by Pablo De Weerdt.
    :param data: pandas DataFrame with the spatial data
    :param grid_points: pandas DataFrame with the locations for idw
    :param data_x_col: name of the x coordinate column
    :param data_y_col: name of the y coordinate column
    :param data_z_col: name of the z coordinate column
    :param grid_x_col: name of the x coordinate column for locations for idw
    :param grid_y_col: name of the y coordinate column for locations for idw
    :param grid_z_col: name of the z coordinate column for locations for idw
    :param o_col: name of the property column
    :param power: power of the inverse distance weighting
    :param max_points: maximum number of points to use for idw 
    :param min_points: minimum number of points to use for idw
    :param search_radii: maximum isotropic search radius
    :return: None
    grid_points['idw' + str(power) + o_col]: pandas DataFrame with the idw values
    """
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
    return None

def kb2d_locations_v2(df, xcol, ycol, vcol,
                        tmin, tmax,
                        df_loc, xcol_loc, ycol_loc,
                        ndmin, ndmax, radius,
                        ktype, skmean, vario):
    """GSLIB's KB2D program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Jan, 2019).  Version for kriging at a set of spatial locations. Modified by Pablo De Weerdt.
    :param df: pandas DataFrame with the spatial data
    :param xcol: name of the x coordinate column
    :param ycol: name of the y coordinate column
    :param vcol: name of the property column
    :param tmin: property trimming limit
    :param tmax: property trimming limit
    :param df_loc: pandas DataFrame with the locations to krige
    :param xcol_loc: name of the x coordinate column for locations to krige
    :param ycol_loc: name of the y coordinate column for locations to krige
    :param ndmin: minimum number of data points to use for kriging a block
    :param ndmax: maximum number of data points to use for kriging a block
    :param radius: maximum isotropic search radius
    :param ktype:
    :param skmean:
    :param vario:
    :return:
    """
# Constants
    UNEST = -999.
    EPSLON = 1.0e-10
    VERSION = 2.907
    first = True
    PMX = 9999.0    
    MAXSAM = ndmax + 1
    MAXKD = MAXSAM + 1
    MAXKRG = MAXKD * MAXKD
    
# load the variogram
    nst = vario['nst']
    cc = np.zeros(nst); aa = np.zeros(nst); it = np.zeros(nst)
    ang = np.full(nst, np.nan); anis = np.full(nst, np.nan)
    
    c0 = vario['nug']; 
    cc[0] = vario['cc1']; it[0] = vario['it1']; ang[0] = vario['azi1']
    aa[0] = vario['hmaj1']; anis[0] = vario['hmin1']/vario['hmaj1']
    if nst == 2:
        cc[1] = vario['cc2']; it[1] = vario['it2']; ang[1] = vario['azi2']
        aa[1] = vario['hmaj2']; anis[1] = vario['hmin2']/vario['hmaj2']
    
# Allocate the needed memory:   
    xa = np.zeros(MAXSAM)
    ya = np.zeros(MAXSAM)
    vra = np.zeros(MAXSAM)
    dist = np.zeros(MAXSAM)
    nums = np.zeros(MAXSAM)
    r = np.zeros(MAXKD)
    rr = np.zeros(MAXKD)
    s = np.zeros(MAXKD)
    a = np.zeros(MAXKRG)
    klist = np.zeros(len(df_loc))       # list of kriged estimates
    vlist = np.zeros(len(df_loc))

# Load the data
    df_extract = df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]    # trim values outside tmin and tmax
    nd = len(df_extract)
    ndmax = min(ndmax,nd)
    x = df_extract[xcol].values
    y = df_extract[ycol].values
    vr = df_extract[vcol].values
    
# Load the estimation loactions
    nd_loc = len(df_loc)
    x_loc = df_loc[xcol_loc].values # Fixed from original code
    y_loc = df_loc[ycol_loc].values # Fixed from original code
    # vr_loc = df_loc[vcol].values
    
# Make a KDTree for fast search of nearest neighbours   
    dp = list((y[i], x[i]) for i in range(0,nd))
    data_locs = np.column_stack((y,x))
    tree = sp.cKDTree(data_locs, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)

# Summary statistics for the data after trimming
    avg = vr.mean()
    stdev = vr.std()
    ss = stdev**2.0
    vrmin = vr.min()
    vrmax = vr.max()

# Initialize accumulators:
    cbb  = 0.0
    rad2 = radius*radius

# Calculate Block Covariance. Check for point kriging.
    rotmat, maxcov = setup_rotmat(c0,nst,it,cc,ang,PMX)
    cov = cova2(0.0,0.0,0.0,0.0,nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
# Keep this value to use for the unbiasedness constraint:
    unbias = cov
    cbb = cov
    first  = False

# MAIN LOOP OVER ALL THE BLOCKS IN THE GRID:
    nk = 0
    ak = 0.0
    vk = 0.0
    
    for idata in range(len(df_loc)):
        print('Working on location ' + str(idata))
        xloc = x_loc[idata]
        yloc = y_loc[idata] 
        current_node = (yloc,xloc)
        
# Find the nearest samples within each octant: First initialize
# the counter arrays:
        na = -1   # accounting for 0 as first index
        dist.fill(1.0e+20)
        nums.fill(-1)
        dist, nums = tree.query(current_node,ndmax) # use kd tree for fast nearest data search
        # remove any data outside search radius
        na = len(dist)
        nums = nums[dist<radius]
        dist = dist[dist<radius] 
        na = len(dist)        

# Is there enough samples?
        if na + 1 < ndmin:   # accounting for min index of 0
            est  = UNEST
            estv = UNEST
            print('UNEST for Data ' + str(idata) + ', at ' + str(xloc) + ',' + str(yloc))
        else:

# Put coordinates and values of neighborhood samples into xa,ya,vra:
            for ia in range(0,na):
                jj = int(nums[ia])
                xa[ia]  = x[jj]
                ya[ia]  = y[jj]
                vra[ia] = vr[jj]
                    
# Handle the situation of only one sample:
            if na == 0:  # accounting for min index of 0 - one sample case na = 0
                cb1 = cova2(xa[0],ya[0],xa[0],ya[0],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
                xx  = xa[0] - xloc
                yy  = ya[0] - yloc

# Establish Right Hand Side Covariance:
                cb = cova2(xx,yy,0.0,0.0,nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)

                if ktype == 0:
                    s[0] = cb/cbb
                    est  = s[0]*vra[0] + (1.0-s[0])*skmean
                    estv = cbb - s[0] * cb
                else:
                    est  = vra[0]
                    estv = cbb - 2.0*cb + cb1
            else:

# Solve the Kriging System with more than one sample:
                neq = na + ktype # accounting for first index of 0
#                print('NEQ' + str(neq))
                nn  = (neq + 1)*neq/2

# Set up kriging matrices:
                iin=-1 # accounting for first index of 0
                for j in range(0,na):

# Establish Left Hand Side Covariance Matrix:
                    for i in range(0,na):  # was j - want full matrix                    
                        iin = iin + 1
                        a[iin] = cova2(xa[i],ya[i],xa[j],ya[j],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov) 
                    if ktype == 1:
                        iin = iin + 1
                        a[iin] = unbias
                    xx = xa[j] - xloc
                    yy = ya[j] - yloc

# Establish Right Hand Side Covariance:
                    cb = cova2(xx,yy,0.0,0.0,nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
                    r[j]  = cb
                    rr[j] = r[j]

# Set the unbiasedness constraint:
                if ktype == 1:
                    for i in range(0,na):
                        iin = iin + 1
                        a[iin] = unbias
                    iin      = iin + 1
                    a[iin]   = 0.0
                    r[neq-1]  = unbias
                    rr[neq-1] = r[neq]

# Solve the Kriging System:
#                print('NDB' + str(ndb))
#                print('NEQ' + str(neq) + ' Left' + str(a) + ' Right' + str(r))
#                stop
                s = ksol_numpy(neq,a,r)
                ising = 0 # need to figure this out
#                print('weights' + str(s))
#                stop
                
            
# Write a warning if the matrix is singular:
                if ising != 0:
                    print('WARNING KB2D: singular matrix')
                    print('              for block' + str(ix) + ',' + str(iy)+ ' ')
                    est  = UNEST
                    estv = UNEST
                else:

# Compute the estimate and the kriging variance:
                    est  = 0.0
                    estv = cbb
                    sumw = 0.0
                    if ktype == 1: 
                        estv = estv - (s[na])*unbias
                    for i in range(0,na):                          
                        sumw = sumw + s[i]
                        est  = est  + s[i]*vra[i]
                        estv = estv - s[i]*rr[i]
                    if ktype == 0: 
                        est = est + (1.0-sumw)*skmean
        klist[idata] = est
        vlist[idata] = estv
        if est > UNEST:
            nk = nk + 1
            ak = ak + est
            vk = vk + est*est

# END OF MAIN LOOP OVER ALL THE BLOCKS:

    if nk >= 1:
        ak = ak / float(nk)
        vk = vk/float(nk) - ak*ak
        print('  Estimated   ' + str(nk) + ' blocks ')
        print('      average   ' + str(ak) + '  variance  ' + str(vk))

    return klist, vlist

def add_grid():
    plt.gca().grid(True, which='major',linewidth = 1.0); plt.gca().grid(True, which='minor',linewidth = 0.2) # add y grids
    plt.gca().tick_params(which='major',length=7); plt.gca().tick_params(which='minor', length=4)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator()); plt.gca().yaxis.set_minor_locator(AutoMinorLocator()) # turn on minor ticks 

def add_grid2(sub_plot):
    sub_plot.grid(True, which='major', linewidth = 1.0); sub_plot.grid(True, which='minor',linewidth = 0.2) # add y grids
    sub_plot.tick_params(which='major', length=7); sub_plot.tick_params(which='minor', length=4)
    sub_plot.xaxis.set_minor_locator(AutoMinorLocator()); sub_plot.yaxis.set_minor_locator(AutoMinorLocator()) # turn on minor ticks

def read_mod_file(file_path):
    """
    Reads a .mod file containing variogram parameters and returns the output in the same format as GSLIB.make_variogram.
    Conceptualised by Pablo De Weerdt.
    :param file_path: Path to the .mod file
    :return: Dictionary containing variogram parameters
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    variogram = {
        'nug': float(root.attrib['nugget']),
        'nst': int(root.attrib['structures_count']),
        'cc1': 0,
        'it1': 1,
        'azi1': 0,
        'hmaj1': 0,
        'hmin1': 0,
        'cc2': 0,
        'it2': 1,
        'azi2': 0,
        'hmaj2': 0,
        'hmin2': 0
    }

    for i in range(1, variogram['nst'] + 1):
        structure = root.find(f'structure_{i}')
        if structure is not None:
            variogram[f'cc{i}'] = float(structure.attrib['contribution'])
            variogram[f'it{i}'] = 1 if structure.attrib['type'] == 'Exponential' else 2  # Assuming 1=Exponential, 2=Spherical

            ranges = structure.find('ranges')
            if ranges is not None:
                variogram[f'hmaj{i}'] = float(ranges.attrib['max'])
                variogram[f'hmin{i}'] = float(ranges.attrib['medium'])

            angles = structure.find('angles')
            if angles is not None:
                variogram[f'azi{i}'] = float(angles.attrib['x'])

    return variogram

