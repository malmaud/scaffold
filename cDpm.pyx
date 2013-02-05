import numpy as np
cimport numpy as np

cpdef int test(int x):
    y=x+1
    return y

cpdef int sample_c(int i,
                   np.ndarray[int, ndim=1] c,
                   params,
                   np.ndarray[bool, ndim=2] data,
                   double dp_alpha,
                   double beta,
                   rng):
    c_diff = np.delete(c, i)
    cdef np.ndarray[int, ndim=1] cluster_ids = np.unique(c_diff)
    cdef int n_clusters = len(cluster_ids)
    cdef np.ndarray[double, ndim=1] p = np.zeros(n_clusters+1)
    cdef np.ndarray[int, ndim=1] x = data[i].astype(int)
    cdef int j, count
    cdef double prior, lh
    for j in range(n_clusters):
        count = sum(c_diff==cluster_id)
        prior = log(count)

