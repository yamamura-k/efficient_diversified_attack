cimport numpy as np
cdef extern from "CPP/_cluster_coef_para_nodeweight.hpp":
    cdef void compute_cluster_coef_batch(int bs, int n, double step, float D[], double Coef[], float weights[])

cdef void _compute_cluster_coef_from_distance_matrix_batch(int bs, int n, double step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C, np.ndarray[float, ndim=1] weights):
    cdef float* _D = <float*> D.data
    cdef float* _weights = <float*> weights.data
    cdef double* _C = <double*> C.data
    compute_cluster_coef_batch(bs, n, step, _D, _C, _weights)
    return 


def compute_cluster_coef_from_distance_matrix_batch(int bs, int n, double step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C, np.ndarray[float, ndim=1] weights):
    ans = _compute_cluster_coef_from_distance_matrix_batch(bs, n, step, D, C, weights)
    return C