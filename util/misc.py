import numpy as np



# I've started using indexes instead of indices because it bothers me when people use "indice" as the singular.
# Between "index and indexes" and "indices and indices" I much prefer the former, so I decided to start using
# indexes, Latin plurals be damned.
def create_stencil_indexes(stencil_size, num_stencils, offset=0):
    """
    Calculate the indexes for every stencil.

    TODO: figure out a function to do this with multiple dimensions

    Parameters
    ----------
    stencil_size : int
        Size of an individual stencil.
    num_stencils : int
        Number of stencils. In multiple dimensions, this should be a tuple
        of the number of stencils along each dimension.
    offset : int
        Constant offset to add to every index. Equivalently, the value of the
        smallest index in the leftmost stencil.
        Useful to index into an array with ghost cells.
    
    Returns
    -------
    ndarray
        The array of stencil indexes.
    """

    return offset + np.arange(stencil_size)[None, ...] + np.arange(num_stencils)[..., None]
