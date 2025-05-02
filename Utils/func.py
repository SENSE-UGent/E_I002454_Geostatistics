# import dependencies
import numpy as np
import scipy.spatial as sp  # for fast nearest neighbor search
from geostatspy.geostats import setup_rotmat, cova2, ksol_numpy, locate, powint
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import xml.etree.ElementTree as ET
import pandas as pd
import math




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

def ik2d_v2(df,xcol,ycol,vcol,ivtype,koption,ncut,thresh,gcdf,trend,tmin,tmax,nx,xmn,xsiz,ny,ymn,ysiz,ndmin,ndmax,radius,ktype,vario):
          
    """A 2D version of GSLIB's IK3D Indicator Kriging program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (March, 2019). Modified by Pablo De Weerdt.
    :param df: pandas DataFrame with the spatial data
    :param xcol: name of the x coordinate column
    :param ycol: name of the y coordinate column
    :param vcol: name of the property column (cateogorical or continuous - note continuous is untested)
    :param ivtype: variable type, 0 - categorical, 1 - continuous
    :param koption: kriging option, 0 - estimation, 1 - cross validation (under construction)
    :param ncut: number of categories or continuous thresholds
    :param thresh: an ndarray with the category labels or continuous thresholds
    :param gcdf: global CDF, not used if trend is present
    :param trend: an ndarray [ny,ny,ncut] with the local trend proportions or cumulative CDF values
    :param tmin: property trimming limit
    :param tmax: property trimming limit
    :param nx: definition of the grid system (x axis)
    :param xmn: definition of the grid system (x axis)
    :param xsiz: definition of the grid system (x axis)
    :param ny: definition of the grid system (y axis)
    :param ymn: definition of the grid system (y axis)
    :param ysiz: definition of the grid system (y axis)
    :param nxdis: number of discretization points for a block
    :param nydis: number of discretization points for a block
    :param ndmin: minimum number of data points to use for kriging a block
    :param ndmax: maximum number of data points to use for kriging a block
    :param radius: maximum isotropic search radius
    :param ktype: kriging type, 0 - simple kriging and 1 - ordinary kriging
    :param vario: list with all of the indicator variograms (sill of 1.0) in consistent order with above parameters
    :return:
    """
        
# Find the needed paramters:
    PMX = 9999.9
    UNEST = -999
    MAXSAM = ndmax + 1
    MAXEQ = MAXSAM + 1
    mik = 0  # full indicator kriging
    use_trend = False
    if trend.shape[0] == nx and trend.shape[1] == ny and trend.shape[2] == ncut: use_trend = True
    
# load the variogram
    MAXNST = 2
    nst = np.zeros(ncut,dtype=int); c0 = np.zeros(ncut); cc = np.zeros((MAXNST,ncut)) 
    aa = np.zeros((MAXNST,ncut),dtype=int); it = np.zeros((MAXNST,ncut),dtype=int) 
    ang = np.zeros((MAXNST,ncut)); 
    # fill anis with 1.0 for isotropic variograms
    anis = np.ones((MAXNST,ncut))

    for icut in range(0,ncut):
        nst[icut] = int(vario[icut]['nst'])
        c0[icut] = vario[icut]['nug']; cc[0,icut] = vario[icut]['cc1']; it[0,icut] = vario[icut]['it1']; 
        ang[0,icut] = vario[icut]['azi1']; 
        aa[0,icut] = vario[icut]['hmaj1']; anis[0,icut] = vario[icut]['hmin1']/vario[icut]['hmaj1'];
        if nst[icut] == 2:
            print('two structures detected')
            cc[1,icut] = vario[icut]['cc2']; it[1,icut] = vario[icut]['it2']; ang[1,icut] = vario[icut]['azi2']; 
            aa[1,icut] = vario[icut]['hmaj2']; anis[1,icut] = vario[icut]['hmin2']/vario[icut]['hmaj2'];

# Load the data
    df_extract = df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]    # trim values outside tmin and tmax
    MAXDAT = len(df_extract)
    MAXCUT = ncut
    MAXNST = 2
    MAXROT = MAXNST*MAXCUT+ 1
    ikout = np.zeros((nx,ny,ncut))
    maxcov = np.zeros(ncut)
            
    # Allocate the needed memory:   
    xa = np.zeros(MAXSAM)
    ya = np.zeros(MAXSAM)
    vra = np.zeros(MAXSAM)
    dist = np.zeros(MAXSAM)
    nums = np.zeros(MAXSAM)
    r = np.zeros(MAXEQ)
    rr = np.zeros(MAXEQ)
    s = np.zeros(MAXEQ)
    a = np.zeros(MAXEQ*MAXEQ)
    ikmap = np.zeros((nx,ny,ncut))
    vr = np.zeros((MAXDAT,MAXCUT+1))
    
    nviol = np.zeros(MAXCUT)
    aviol = np.zeros(MAXCUT)
    xviol = np.zeros(MAXCUT)
    
    ccdf = np.zeros(ncut)
    ccdfo = np.zeros(ncut)
    ikout = np.zeros((ny,nx,ncut))
    
    x = df_extract[xcol].values
    y = df_extract[ycol].values
    v = df_extract[vcol].values
    
# The indicator data are constructed knowing the thresholds and the
# data value.
    
    if ivtype == 0:
        for icut in range(0,ncut): 
            vr[:,icut] = np.where((v <= thresh[icut] + 0.5) & (v > thresh[icut] - 0.5), '1', '0')
    else:
        for icut in range(0,ncut): 
            vr[:,icut] = np.where(v <= thresh[icut], '1', '0')
    vr[:,ncut] = v

# Make a KDTree for fast search of nearest neighbours   
    dp = list((y[i], x[i]) for i in range(0,MAXDAT))
    data_locs = np.column_stack((y,x))
    tree = sp.cKDTree(data_locs, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)
    
# Summary statistics of the input data
    
    avg = vr[:,ncut].mean()
    stdev = vr[:,ncut].std()
    ss = stdev**2.0
    vrmin = vr[:,ncut].min()
    vrmax = vr[:,ncut].max()
    # print('Data for ik2d: Variable column ' + str(vcol))
    # print('  Number   = ' + str(MAXDAT))
    ndh = MAXDAT
    
    actloc = np.zeros(MAXDAT, dtype = int)
    for i in range(1,MAXDAT):
        actloc[i] = i
    
# Set up the rotation/anisotropy matrices that are needed for the
# variogram and search:

    # print('Setting up rotation matrices for variogram and search')
    radsqd = radius * radius
    rotmat = []
    for ic in range(0,ncut):  
        rotmat_temp, maxcov[ic] = setup_rotmat(c0[ic],int(nst[ic]),it[:,ic],cc[:,ic],ang[:,ic],9999.9)
        rotmat.append(rotmat_temp)    
# Initialize accumulators:  # not setup yet
    nk = 0
    xk = 0.0
    vk = 0.0
    for icut in range (0,ncut):
        nviol[icut] =  0
        aviol[icut] =  0.0
        xviol[icut] = -1.0
    nxy   = nx*ny
    # print('Working on the kriging')

# Report on progress from time to time:
    if koption == 0: 
        nxy   = nx*ny
        nloop = nxy
        irepo = max(1,min((nxy/10),10000))
    else:
        nloop = 10000000
        irepo = max(1,min((nd/10),10000))
    ddh = 0.0
    
# MAIN LOOP OVER ALL THE BLOCKS IN THE GRID:
    for index in range(0,nloop):
      
        # if (int(index/irepo)*irepo) == index: print('   currently on estimate ' + str(index))
    
        if koption == 0:
            
            iy   = int((index)/nx) 
            ix   = index - (iy)*nx
            xloc = xmn + (ix)*xsiz
            yloc = ymn + (iy)*ysiz
        else:
            ddh = 0.0
            # TODO: pass the cross validation value

# Find the nearest samples within each octant: First initialize the counter arrays:
        na = -1   # accounting for 0 as first index
        dist.fill(1.0e+20)
        nums.fill(-1)
        current_node = (yloc,xloc)
        dist, close = tree.query(current_node,ndmax) # use kd tree for fast nearest data search
        # remove any data outside search radius
        close = close[dist<radius]
        dist = dist[dist<radius] 
        nclose = len(dist) 

# Is there enough samples?

        if nclose < ndmin:   # accounting for min index of 0
            for i in range(0,ncut):
                ccdfo[i] = UNEST
            print('UNEST at ' + str(ix) + ',' + str(iy))
        else:         

# Loop over all the thresholds/categories:
            for ic in range(0,ncut):
                krig = True
                if mik == 1 and ic >= 1: krig = False

# Identify the close data (there may be a different number of data at
# each threshold because of constraint intervals); however, if
# there are no constraint intervals then this step can be avoided.
                nca = -1
                for ia in range(0,nclose):
                    j  = int(close[ia]+0.5)
                    ii = actloc[j]
                    accept = True
                    if koption != 0 and (abs(x[j]-xloc) + abs(y[j]-yloc)).lt.EPSLON: accept = False
                    if accept:
                        nca = nca + 1
                        vra[nca] = vr[ii,ic]
                        xa[nca]  = x[j]
                        ya[nca]  = y[j]

# If there are no samples at this threshold then use the global cdf:
                if nca == -1:
                    if use_trend:
                        ccdf[ic] = trend[ny-iy-1,ix,ic]
                    else:
                        ccdf[ic] = gcdf[ic]
                else:
            
# Now, only load the variogram, build the matrix,... if kriging:
                    neq = nclose + ktype
                    na = nclose

# Set up kriging matrices:
                    iin=-1 # accounting for first index of 0
                    for j in range(0,na):
# Establish Left Hand Side Covariance Matrix:
                        for i in range(0,na):  # was j - want full matrix                    
                            iin = iin + 1
                            a[iin] = cova2(xa[i],ya[i],xa[j],ya[j],nst[ic],c0[ic],PMX,cc[:,ic],aa[:,ic],it[:,ic],ang[:,ic],anis[:,ic],rotmat[ic],maxcov[ic]) 
                        if ktype == 1:
                            iin = iin + 1
                            a[iin] = maxcov[ic]            
                        r[j] = cova2(xloc,yloc,xa[j],ya[j],nst[ic],c0[ic],PMX,cc[:,ic],aa[:,ic],it[:,ic],ang[:,ic],anis[:,ic],rotmat[ic],maxcov[ic]) 
    
# Set the unbiasedness constraint:
                    if ktype == 1:
                        for i in range(0,na):
                            iin = iin + 1
                            a[iin] = maxcov[ic]
                        iin      = iin + 1
                        a[iin]   = 0.0
                        r[neq-1]  = maxcov[ic]
                        rr[neq-1] = r[neq]
# Solve the system:
                    if neq == 1:
                        ising = 0.0
                        s[0]  = r[0] / a[0]
                    else:
                        s = ksol_numpy(neq,a,r)

# Finished kriging (if it was necessary):

# Compute Kriged estimate of cumulative probability:
                    sumwts   = 0.0
                    ccdf[ic] = 0.0
                    for i in range(0,nclose):
                        ccdf[ic] = ccdf[ic] + vra[i]*s[i]
                        sumwts   = sumwts   + s[i]
                    if ktype == 0: 
                        if use_trend == True:
                            ccdf[ic] = ccdf[ic] + (1.0-sumwts)*trend[ny-iy-1,ix,ic]
                        else:
                            ccdf[ic] = ccdf[ic] + (1.0-sumwts)*gcdf[ic]

# Keep looping until all the thresholds are estimated:
 
# Correct and write the distribution to the output file:
            nk = nk + 1
            ccdfo = ordrel2(ivtype,ncut,ccdf)

            # check if the resulting ccdfo is in increasing order
            if np.any(np.diff(ccdfo) < 0):
                print('Warning: ccdfo is not in increasing order.')
                print('ccdf: ' + str(ccdf))
                # print('ccdfo: ' + str(ccdfo))
                print("location " + str(ny-iy-1) + ',' + str(ix) + '  ' + str(ccdfo))
        
# Write the IK CCDF for this grid node:
            if koption == 0:
                # print("location " + str(ny-iy-1) + ',' + str(ix) + '  ' + str(ccdfo))
                ikout[ny-iy-1,ix,:] = ccdfo
            else:
                 print('TBD')
    return ikout

def ordrel2(ivtype,ncut,ccdf):
    """Correct a indicator based CDF for order relations.
    :param ivtype: variable type, 0 - categorical and 1 - continuous
    :param ncut: number of categories or thresholds
    :param ccdf: input cumulative distribution function
    :return: cumulative distribution function correct for order relations
    """
#    print('input ordering relations'); print(ccdf)
    ccdfo = np.zeros(ncut)
    ccdf1 = np.zeros(ncut)
    ccdf2 = np.zeros(ncut) # do we need MAXCUT = 100 for these 2?

# Make sure conditional cdf is within [0,1]:
    for i in range(0,ncut):
        if ccdf[i] < 0.0:
            ccdf1[i] = 0.0
            ccdf2[i] = 0.0
        elif ccdf[i] > 1.0:
            ccdf1[i] = 1.0
            ccdf2[i] = 1.0
        else:
            ccdf1[i] = ccdf[i]
            ccdf2[i] = ccdf[i]
#    print('ordering relations'); print(ccdf1,ccdf2)

# Correct sequentially up, then down, and then average:
    if ivtype == 0:
        sumcdf = 0.0
        for i in range(0,ncut):
            sumcdf = sumcdf + ccdf1[i]
        if sumcdf <= 0.0: sumcdf = 1.0
        for i in range(0,ncut):
            ccdfo[i] = ccdf1[i] / sumcdf
    else:
        for i in range(1,ncut):
            if ccdf1[i] < ccdf1[i-1]: ccdf1[i] = ccdf1[i-1]
        for i in range(ncut-2,-1,-1):
            if ccdf2[i] > ccdf2[i+1]: ccdf2[i] = ccdf2[i+1]
        for i in range(0,ncut):
            ccdfo[i] = 0.5*(ccdf1[i]+ccdf2[i])

# Return with corrected CDF:
    return ccdfo

def beyond2(ivtype,nccut,ccut,ccdf,ncut,cut,cdf,zmin,zmax,ltail,ltpar,middle,mpar,utail,utpar,zval,cdfval):
#-----------------------------------------------------------------------
#
#                     Go Beyond a Discrete CDF
#                     ************************
#
# This subroutine is a general purpose subroutine to interpolate within
# and extrapolate beyond discrete points on a conditional CDF.  If the
# Z value "zval" is specified then the corresponding CDF value "cdfval"
# will be computed, if the CDF value "cdfval" is specified the
# corresponding Z value "zval" will be computed. Modified from GeostatsPy by Pablo De Weerdt.
#
#
#
# INPUT/OUTPUT VARIABLES:
#
#   ivtype           variable type (1=continuous, 0=categorical)
#   nccut            number of cutoffs defining the conditional CDF
#   ccut()           real array of the nccut cutoffs
#   ccdf()           real array of the conditional cdf values
#   ncut             number of cutoffs defining the global CDF
#   cut()            real array of the ncut cutoffs
#   cdf()            real array of the global cdf values
#
#   zmin,zmax        minimum and maximum allowable data values
#   ltail            option to handle values in lower tail
#   ltpar            parameter required for option ltail
#   middle           option to handle values in the middle
#   mpar             parameter required for option middle
#   utail            option to handle values in upper tail
#   utpar            parameter required for option utail
#
#   zval             interesting cutoff (if -1 then it is calculated)
#   cdfval           interesting CDF (if -1 then it is calculated)
#
#
#-----------------------------------------------------------------------
    EPSLON = 1.0e-20; UNEST=-1.0

# Check for both "zval" and "cdfval" defined or undefined:
    ierr  = 1; 
    if zval > UNEST and cdfval > UNEST: 
        return -1
    if zval <= UNEST and cdfval <= UNEST: 
        return - 1
    
# Handle the case of a categorical variable:
    if ivtype == 0:
        cum = 0
        for i in range(0,nccut):
            cum = cum + ccdf[i]
            if cdfval <= cum:
                zval = ccut[i]
                return zval
        return zval
    
# Figure out what part of distribution: ipart = 0 - lower tail
#                                       ipart = 1 - middle
#                                       ipart = 2 - upper tail
    ierr  = 0
    ipart = 1
    if zval > UNEST:
        if zval <= ccut[0]:       
            ipart = 0
        if zval >= ccut[nccut-1]:
            ipart = 2
    else:
        if cdfval <= ccdf[0]:
            ipart = 0
        if cdfval >= ccdf[nccut-1]:
            ipart = 2
      
# ARE WE IN THE LOWER TAIL?

    if ipart == 0: 
        if ltail ==1:
# Straight Linear Interpolation:
            powr = 1.0
            if zval > UNEST:
                cdfval = powint(zmin,ccut[0],0.0,ccdf[0],zval,powr)
            else:
                zval = powint(0.0,ccdf[0],zmin,ccut[0],cdfval,powr)
        elif ltail == 2:

# Power Model interpolation to lower limit "zmin"?
                if zval > UNEST: 
                    cdfval = powint(zmin,ccut[0],0.0,ccdf[0],zval,ltpar)
                else:
                    powr = 1.0 / ltpar
                    zval = powint(0.0,ccdf[0],zmin,ccut[0],cdfval,powr)
                
# Linear interpolation between the rescaled global cdf?
        elif ltail == 3:
            if zval > UNEST:
# Computing the cdf value. Locate the point and the class bound:
                idat = locate(cut,1,ncut,zval)
                iupp = locate(cut,ncut,1,ncut,ccut[0])

# Straight linear interpolation if no data; otherwise, linear:
                if idat <= -1 or idat >= ncut -1 or iupp <= -1 or iupp >= ncut-1: # modfity for 0 index
                    cdfval = powint(zmin,cut[0],0.0,cdf[0],zval,1.)
                else:
                    temp = powint(cut[idat],cut[idat+1],cdf[idat],cdf[idat+1],zval,1.)
                    cdfval = temp*ccdf[0]/cdf[iupp]
            else:

# Computing Z value: Are there any data out in the tail?

                iupp = locate(cut,ncut,1,ncut,ccut[0])

# Straight linear interpolation if no data; otherwise, local linear
# interpolation:
                if iupp <= 0 or iupp >= ncut:
                    zval = powint(0.0,cdf[0],zmin,cut[0],cdfval,1.)
                else:
                    temp = cdfval*cdf[iupp]/ccdf[1]
                    idat = locate(cdf,ncut,1,ncut,temp)
                    if idat <= -1 or idat >= ncut-1:  # adjusted for 0 origin
                        zval = powint(0.0,cdf[0],zmin,cut[0],cdfval,1.)
                    else:
                        zval = powint(cdf[idat],cdf[idat+1],cut[dat],cut[idat+1],temp,1.)
        else:

# Error situation - unacceptable option:
           ierr = 2
           return -1
            
# FINISHED THE LOWER TAIL,  ARE WE IN THE MIDDLE?
    if ipart == 1:

# Establish the lower and upper limits:
        if zval > UNEST: 
            cclow = locate(ccut,1,nccut,zval)
            cchigh = cclow + 1
        else:
            cclow = locate(ccdf,1,nccut,cdfval)
            cchigh = cclow + 1
        if middle == 1:

# Straight Linear Interpolation:
            powr = 1.0
            if zval > UNEST:
                cdfval = powint(ccut[cclow],ccut[cchigh],ccdf[cclow],ccdf[cchigh],zval,powr)
            else:
                zval = powint(ccdf[cclow],ccdf[cchigh],ccut[cclow],ccut[cchigh],cdfval,powr)
                  
# Power interpolation between class bounds?
        elif middle == 2:
                if zval > UNEST:
                    cdfval = powint(ccut[cclow],ccut[cchigh],ccdf[cclow],ccdf[cchigh],zval,mpar)
                else:
                    powr = 1.0 / mpar
                    zval = powint(ccdf[cclow],ccdf[cchigh],ccut[cclow],ccut[cchigh],cdfval,powr)
                  
# Linear interpolation between the rescaled global cdf?
        elif middle == 3:
            ilow = locate(cut,ncut,1,ncut,ccut[cclow])
            iupp = locate(cut,ncut,1,ncut,ccut[cchigh])
            if cut[ilow] < ccut[cclow]:  
                ilow = ilow + 1
            if cut[iupp]  > ccut[cchigh]:  
                iupp = iupp - 1
            if zval > UNEST:
                idat = locate(cut,1,ncut,zval)

# Straight linear interpolation if no data; otherwise, local linear
# interpolation:
                if idat <= -1 or idat >= ncut-1 or ilow <= -1 or ilow >= ncut-1 or iupp <= -1 or iupp >= ncut-1 or iupp <= ilow:
                    cdfval=powint(ccut[cclow],ccut[cchigh],ccdf[cclow],ccdf[cchigh],zval,1.)
                else:
                    temp = powint(cut[idat],cut[idat+1],cdf[idat],cdf[idat+1],zval,1.)
                    cdfval=powint(cdf[ilow],cdf[iupp],ccdf[cclow],ccdf[cchigh],temp,1.)
            else:

# Straight linear interpolation if no data; otherwise, local linear
# interpolation:
                if ilow <= -1 or ilow >= ncut-1 or iup <= -1 or iupp >= ncut-1 or iupp < ilow:
                    zval=powint(ccdf[cclow],ccdf[cchigh],ccut[cclow],ccut[cchigh],cdfval,1.)
                else:
                    temp=powint(ccdf[cclow],ccdf[cchigh],cdf[ilow],cdf[iupp],cdfval,1.)
                    idat = locate(cdf,1,ncut,temp)
                    if cut[idat] < ccut[cclow]: 
                        idat=idat+1
                    if idat <= -1 or idat >= ncut-1 or cut[idat+1] > ccut[cchigh]:
                        zval = powint(ccdf[cclow],ccdf[cchigh],ccut[cclow],ccut[cchigh],cdfval,1.)
                    else:
                        zval = powint(cdf[idat],cdf[idat+1],cut[idat],cut[idat+1],temp,1.)
                    zval = powint(cdf[idat],cdf[idat+1],cut[idat],cut[idat+1],temp,1.)

        else:

# Error situation - unacceptable option:
            ierr = 2
            return -1

# FINISHED THE MIDDLE,  ARE WE IN THE UPPER TAIL?
    if ipart == 2: 
        if utail == 1: 
            powr = 1.0
            if zval > UNEST:
                cdfval = powint(ccut(nccut),zmax,ccdf(nccut),1.0,zval,powr)
            else:
                zval   = powint(ccdf(nccut),1.0,ccut(nccut),zmax,cdfval,powr)        
        elif utail == 2:

# Power interpolation to upper limit "utpar"?
            if zval > UNEST:
                cdfval = powint(ccut(nccut),zmax,ccdf(nccut),1.0,zval,utpar)
            else:
                powr = 1.0 / utpar
                zval   = powint(ccdf(nccut),1.0,ccut(nccut),zmax,cdfval,powr)

# Linear interpolation between the rescaled global cdf?
        elif utail == 3:
            if zval > UNEST:

# Approximately Locate the point and the class bound:
                idat = locate(cut,1,ncut,zval,idat)
                ilow = locate(cut,1,ncut,ccut(nccut),ilow)
                if cut[idat] < zval:
                    idat = idat + 1
                if cut[ilow] < ccut[nccut-1]: 
                    ilow = ilow + 1

# Straight linear interpolation if no data; otherwise, local linear
# interpolation:
                if idat < -1 or idat >= ncut-1 or ilow <= -1 or ilow >= ncut-1:
                    cdfval = powint(ccut(nccut),zmax,ccdf(nccut),1.0,zval,1.)
                else:
                    temp   = powint(cut(idat),cut(idat+1),cdf(idat),cdf(idat+1),zval,1.)
                    cdfval = powint(cdf(ilow),1.0,ccdf(nccut),1.0,temp,1.)

            else:

# Computing Z value: Are there any data out in the tail?
                ilow = locate(cut,ncut,1,ncut,ccut(nccut),ilow)
                if cut[ilow] < ccut[nccut-1]: 
                    ilow = ilow + 1

# Straight linear interpolation if no data; otherwise, local linear
# interpolation:
                if ilow <= -1 or ilow >= ncut-1:
                    zval   = powint(ccdf(nccut),1.0,ccut(nccut),zmax,cdfval,1.)
                else:
                    temp = powint(ccdf(nccut),1.0,cdf(ilow),1.0,cdfval,1.)
                    idat = locate(cdf,ncut,1,ncut,temp)
                    if cut[idat] < ccut[nccut-1]: 
                        idat=idat+1
                    if idat >= ncut-1:
                        zval   = powint(ccdf[nccut-1],1.0,ccut[nccut-1],zmax,cdfval,1.)
                    else:
                        zval = powint(cdf[idat],cdf[idat+1],cut[idat],cut[idat+1],temp,1.)

# Fit a Hyperbolic Distribution?
        elif utail == 4:

# Figure out "lambda" and required info:
            lambd = math.pow(ccut[nccut-1],utpar)*(1.0-ccdf[nccut-1])
            if zval > UNEST: 
                cdfval = 1.0 - (lambd/(math.pow(zval,utpar)))
            else:
                zval = (lambd/math.pow((1.0-cdfval),(1.0/utpar)))          
        else:

# Error situation - unacceptable option:
            ierr = 2
            return -1
        

    if zval < zmin:
        zval = zmin
    if zval > zmax: 
        zval = zmax

# All finished - return:

    return zval, cdfval

def calculate_etype_and_conditional_variance(ccdf, ccut, maxdis, zmin, zmax, ltail, ltpar, middle, mpar, utail, utpar):
    """
    Calculate the e-type and conditional variance based on ccdf and ccut arrays using beyond2 for CCDF reconstruction.
    Code based on POSTIK Fortran code from GSLIB. Adapted by Pablo De Weerdt.
    Parameters:
        ccdf (list of float): Cumulative distribution function values.
        ccut (list of float): Cutoff values corresponding to ccdf.
        maxdis (int): Maximum discretization for calculations.
        zmin (float): Minimum Z value.
        zmax (float): Maximum Z value.
        ltail (int): Option to handle values in the lower tail.
        ltpar (float): Parameter for the lower tail option.
        middle (int): Option to handle values in the middle.
        mpar (float): Parameter for the middle option.
        utail (int): Option to handle values in the upper tail.
        utpar (float): Parameter for the upper tail option.

    Returns:
        tuple: A tuple containing e-type (float) and conditional variance (float).
    """
    dis = 1.0 / maxdis
    cdfval = -0.5 * dis
    etype = 0.0
    ecv = 0.0

    for _ in range(maxdis):
        cdfval += dis
        zval = -1.0

        # Use beyond2 for CCDF reconstruction
        zval = beyond2(1, len(ccut), ccut, ccdf, 0, [], [], zmin, zmax, ltail, ltpar, middle, mpar, utail, utpar, zval, cdfval)[0]

        etype += zval
        ecv += zval * zval

    etype /= maxdis
    ecv = max((ecv / maxdis - etype * etype), 0.0)

    return etype, ecv

def ik2d_v2_loc(df, xcol, ycol, vcol, ivtype, ncut, thresh, gcdf, tmin, tmax, df_loc, xcol_loc, ycol_loc, ndmin, ndmax, radius, ktype, vario):
    """
    A 2D version of GSLIB's IK3D Indicator Kriging program for a set of spatial locations.
    This function uses the same steps and logic as ik2d_v2, but the input locations for prediction are provided as a pandas DataFrame.
    Modified from GeostatsPy's ik2d_v2 function by Pablo De Weerdt.
    :param df: pandas DataFrame with the spatial data
    :param xcol: name of the x coordinate column
    :param ycol: name of the y coordinate column
    :param vcol: name of the property column (categorical or continuous)
    :param ivtype: variable type, 0 - categorical, 1 - continuous
    :param ncut: number of categories or continuous thresholds
    :param thresh: an ndarray with the category labels or continuous thresholds
    :param gcdf: global CDF, not used if trend is present
    :param tmin: property trimming limit
    :param tmax: property trimming limit
    :param df_loc: pandas DataFrame with the locations to krige
    :param xcol_loc: name of the x coordinate column for locations to krige
    :param ycol_loc: name of the y coordinate column for locations to krige
    :param ndmin: minimum number of data points to use for kriging a block
    :param ndmax: maximum number of data points to use for kriging a block
    :param radius: maximum isotropic search radius
    :param ktype: kriging type, 0 - simple kriging and 1 - ordinary kriging
    :param vario: list with all of the indicator variograms (sill of 1.0) in consistent order with above parameters
    :return: DataFrame with kriged estimates and variances for the specified locations
    """
    PMX = 9999.9
    UNEST = -999

    # Load the variogram parameters
    MAXNST = 2
    nst = np.zeros(ncut, dtype=int)
    c0 = np.zeros(ncut)
    cc = np.zeros((MAXNST, ncut))
    aa = np.zeros((MAXNST, ncut), dtype=int)
    it = np.zeros((MAXNST, ncut), dtype=int)
    ang = np.zeros((MAXNST, ncut))
    anis = np.ones((MAXNST, ncut))  # Fill anisotropy with 1.0 for isotropic variograms

    for icut in range(ncut):
        nst[icut] = int(vario[icut]['nst'])
        c0[icut] = vario[icut]['nug']
        cc[0, icut] = vario[icut]['cc1']
        it[0, icut] = vario[icut]['it1']
        ang[0, icut] = vario[icut]['azi1']
        aa[0, icut] = vario[icut]['hmaj1']
        anis[0, icut] = vario[icut]['hmin1'] / vario[icut]['hmaj1']
        if nst[icut] == 2:
            cc[1, icut] = vario[icut]['cc2']
            it[1, icut] = vario[icut]['it2']
            ang[1, icut] = vario[icut]['azi2']
            aa[1, icut] = vario[icut]['hmaj2']
            anis[1, icut] = vario[icut]['hmin2'] / vario[icut]['hmaj2']

    # Load the data
    df_extract = df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]
    x = df_extract[xcol].values
    y = df_extract[ycol].values
    v = df_extract[vcol].values

    # Indicator data
    vr = np.zeros((len(df_extract), ncut + 1))
    if ivtype == 0:
        for icut in range(ncut):
            vr[:, icut] = np.where((v <= thresh[icut] + 0.5) & (v > thresh[icut] - 0.5), 1, 0)
    else:
        for icut in range(ncut):
            vr[:, icut] = np.where(v <= thresh[icut], 1, 0)
    vr[:, ncut] = v

    # KDTree for fast nearest neighbor search
    data_locs = np.column_stack((y, x))
    tree = sp.cKDTree(data_locs, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)

    # Prepare output DataFrame
    results = []

    # Set up the rotation/anisotropy matrices for the variogram
    rotmat = []
    maxcov = np.zeros(ncut)
    for ic in range(ncut):
        rotmat_temp, maxcov[ic] = setup_rotmat(c0[ic], nst[ic], it[:, ic], cc[:, ic], ang[:, ic], PMX)
        rotmat.append(rotmat_temp)

    # Loop over locations
    for idx, loc in df_loc.iterrows():
        xloc = loc[xcol_loc]
        yloc = loc[ycol_loc]
        current_node = (yloc, xloc)

        # Find nearest samples
        dist, close = tree.query(current_node, ndmax)
        close = close[dist < radius]
        dist = dist[dist < radius]
        nclose = len(dist)

        if nclose < ndmin:
            results.append({
                xcol_loc: xloc,
                ycol_loc: yloc,
                'estimate': UNEST,
                'variance': UNEST
            })
            continue

        # Kriging for each threshold
        ccdf = np.zeros(ncut)
        for ic in range(ncut):
            neq = nclose + ktype
            a = np.zeros((neq, neq))
            r = np.zeros(neq)

            for j in range(nclose):
                for i in range(nclose):
                    a[i, j] = cova2(x[close[i]], y[close[i]], x[close[j]], y[close[j]],
                                     nst[ic], c0[ic], PMX, cc[:, ic], aa[:, ic], it[:, ic],
                                     ang[:, ic], anis[:, ic], rotmat[ic], maxcov[ic])
                r[j] = cova2(xloc, yloc, x[close[j]], y[close[j]],
                             nst[ic], c0[ic], PMX, cc[:, ic], aa[:, ic], it[:, ic],
                             ang[:, ic], anis[:, ic], rotmat[ic], maxcov[ic])

            if ktype == 1:
                a[-1, :-1] = 1
                a[:-1, -1] = 1
                a[-1, -1] = 0
                r[-1] = 1

            # Solve the system
            if neq == 1:
                ising = 0.0
                s = np.zeros(neq)
                s[0] = r[0] / a[0, 0]
            else:
                s = ksol_numpy(neq, a.flatten(), r)

            # Compute kriged estimate
            ccdf[ic] = np.sum(s[:nclose] * vr[close, ic])

        # Correct and write the distribution to the output file
        ccdfo = ordrel2(ivtype, ncut, ccdf)

        # Store results per threshold into separate columns
        result = {xcol_loc: xloc, ycol_loc: yloc}
        for ic in range(ncut):
            result[f'estimate_thresh_{ic+1}'] = ccdfo[ic]
        results.append(result)

    return pd.DataFrame(results)