import katdal
from katdal.lazy_indexer import DaskLazyIndexer
from katdal.lazy_indexer import LazyTransform

import numpy as np

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
