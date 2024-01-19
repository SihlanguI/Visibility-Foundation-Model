import math
import katdal
import numpy as np
import pandas as pd


def get_corrprods(vis):
    bl = vis.corr_products
    bl_idx = []
    for i in range(len(bl)):
        bl_idx.append((bl[i][0][0:-1]+bl[i][1][0:-1]))
    return np.array(bl_idx)


def get_bl_idx(vis, nant):
    """
    Get the indices of the correlation products.

    Parameters:
    -----------
    vis : katdal.visdatav4.VisibilityDataV4
       katdal data object
    nant : int
       number of antennas

    Returns:
    --------
    output : numpy array
       array of baseline indices
    """
    nant = nant
    A1, A2 = np.triu_indices(nant, 1)
   #print(A1)
    # Creating baseline antenna combinations
    corr_products = np.array(['m{:03d}m{:03d}'.format(A1[i], A2[i]) for i in range(len(A1))])
    df = pd.DataFrame(data=np.arange(len(A1)), index=corr_products).T
    corr_prods = get_corrprods(vis)
    bl_idx = df[corr_prods].values[0].astype(np.int32)

    return bl_idx 
