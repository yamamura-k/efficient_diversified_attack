import numpy as np
from cython import boundscheck, wraparound
cimport cython
cimport numpy as cnp

cpdef cnp.ndarray[int, ndim=1] _get_target_classes(cnp.ndarray[int, ndim=3] target_class):
        cdef int im
        cdef int _size
        cpdef cnp.ndarray[int, ndim=1] n_classes
        with  boundscheck(False), wraparound(False):
            _size = target_class.shape[1]
            n_classes = np.ones((_size, ), dtype=np.int32)
            for im in range(_size):
                n_classes[im] = len(
                    set(target_class[:, im, :].reshape(-1).tolist())
                )
        return n_classes