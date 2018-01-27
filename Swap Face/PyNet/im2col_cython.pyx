# Author: Xiao Zhang
# Credit: Heavily referenced from Stanford CS231N

import numpy as np
cimport numpy as np
cimport cython

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def im2col_cython(np.ndarray[DTYPE_t, ndim=4] x, np.ndarray[np.int32_t, ndim=1] layer_config):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    
    cdef int field_height = layer_config[0]
    cdef int field_width = layer_config[1]
    cdef int padding_h = layer_config[2]
    cdef int padding_w = layer_config[3]
    cdef int stride_h = layer_config[4]
    cdef int stride_w = layer_config[5]

    cdef int HH = (H + 2 * padding_h - field_height) / stride_h + 1
    cdef int WW = (W + 2 * padding_w - field_width) / stride_w + 1

   
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.pad(x,
            ((0, 0), (0, 0), (padding_h, padding_w), (padding_h, padding_w)), mode='constant')

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (C * field_height * field_width, N * HH * WW),
            dtype=x.dtype)

    # Moving the inner loop to a C function with no bounds checking works, but does
    # not seem to help performance in any measurable way.

    im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW,
                        field_height, field_width, stride_h, stride_w)
    return cols


@cython.boundscheck(False)
cdef int im2col_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int stride_h, int stride_w) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride_h * yy + ii, stride_w * xx + jj]



def col2im_cython(np.ndarray[DTYPE_t, ndim=2] cols, np.ndarray[np.int64_t, ndim=1] input_shape,  np.ndarray[np.int32_t, ndim=1] layer_config):

    cdef int N = input_shape[0]
    cdef int C = input_shape[1]
    cdef int H = input_shape[2]
    cdef int W = input_shape[3]

    cdef int field_height = layer_config[0]
    cdef int field_width = layer_config[1]
    cdef int padding_h = layer_config[2]
    cdef int padding_w = layer_config[3]
    cdef int stride_h = layer_config[4]
    cdef int stride_w = layer_config[5]

    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int HH = (H + 2 * padding_h - field_height) / stride_h + 1
    cdef int WW = (W + 2 * padding_w - field_width) / stride_w + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * padding_h, W + 2 * padding_w),
                                        dtype=cols.dtype)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_cython_inner(cols, x_padded, N, C, H, W, HH, WW, 
                        field_height, field_width, stride_h, stride_w)
    if padding_h == 0 and padding_w == 0:
        return x_padded
    elif  padding_h == 0:
        return x_padded[:, :, :, padding_w:-padding_w]
    else:
        return x_padded[:, :, padding_h:-padding_h, padding_w:-padding_w]


@cython.boundscheck(False)
cdef int col2im_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int stride_h, int stride_w) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride_h * yy + ii, stride_w * xx + jj] += cols[row, col]
