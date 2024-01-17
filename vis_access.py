import katdal
from katdal.lazy_indexer import DaskLazyIndexer
from katdal.lazy_indexer import LazyTransform

import katdal
import numpy as np
import pandas as pd

def read_rdb(path):
    """
    Read in the RDB file.

    Parameters
    ----------
    path : str
        RDB file

    Returns
    -------
    output_file : katdal.visdatav4.VisibilityDataV4
       katdal data object
    """
    data = katdal.open(path)
    return data
    

def load(dataset, indices, vis, weights, flags):
    """Load data from lazy indexers into existing storage.

    This is optimised for the MVF v4 case where we can use dask directly
    to eliminate one copy, and also load vis, flags and weights in parallel.
    In older formats it causes an extra copy.

    Parameters
    ----------
    dataset : :class:`katdal.DataSetw
        Input dataset, possibly with an existing selection
    indices : tuple
        Index expression for subsetting the dataset
    vis, weights, flags : array-like
        Outputs, which must have the correct shape and type
    """
    if isinstance(dataset.vis, DaskLazyIndexer):
        DaskLazyIndexer.get([dataset.vis, dataset.weights, dataset.flags], indices,
                            out=[vis, weights, flags])
    else:
        vis[:] = dataset.vis[indices]
        weights[:] = dataset.weights[indices]
        flags[:] = dataset.flags[indices]
        
        
def vis_per_scan(data):
    """Load visibilities per scan into memory.
    
    Parameters:
    dataset : :class:`katdal.DataSetw
        Input dataset, possibly with an existing selection
    
    """
    for scan in data.scans():
        n_time, n_chans, n_bl = data.shape
        vis_chunk = np.empty((int(n_time), int(n_chans), int(n_bl)), dtype=np.complex64)
        weight_chunk = np.zeros_like(vis_chunk, dtype=np.float32)
        flag_chunk = np.zeros_like(vis_chunk, dtype=np.bool)
        load(data, np.s_[:, :, :], vis_chunk, weight_chunk, flag_chunk)
    return vis_chunk, flag_chunk


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
    bl_idx : numpy array of baseline indices
    """
    nant = nant
    A1, A2 = np.triu_indices(nant, 1)
    # Create baseline antenna combinations
    corr_products = np.array(['m{:03d}m{:03d}'.format(A1[i], A2[i]) for i in range(len(A1))])
    df = pd.DataFrame(data=np.arange(len(A1)), index=corr_products).T
    corr_prods = get_corrprods(vis)
    bl_idx = df[corr_prods].values[0].astype(np.int32)
    return bl_idx