import cython
import numpy as np
cimport numpy as np

ctypedef fused float:
	np.float32_t
	np.float64_t

cdef extern void bilinear_forward_kernel(float* input, 
								  float* output, 
								  int output_h, 
								  int output_w, 
								  int input_h, 
								  int input_w,
								  int nchannel,
								  int nbatch
								  );

cdef extern void bilinear_backward_kernel(float* grad_output, 
								   float* grad_input, 
								   int output_h, 
								   int output_w, 
								   int input_h, 
								   int input_w,
								   int nchannel,
								   int nbatch
								   );

def bilinear_forward(np.ndarray[float, ndim=4, mode="c"] input,
					np.ndarray[float, ndim=4, mode="c"] output):
	cdef int N = input.shape[0]
	cdef int C = input.shape[1]
	cdef int input_h = input.shape[2]
	cdef int input_w = input.shape[3]
	cdef int output_h = output.shape[2]
	cdef int output_w = output.shape[3]

	bilinear_forward_kernel(&input[0,0,0,0], &output[0,0,0,0], output_h, output_w, input_h, input_w, C, N)
	return output

def bilinear_backward(np.ndarray[float, ndim=4, mode="c"] grad_output,
					np.ndarray[float, ndim=4, mode="c"] grad_input):
	cdef int N = grad_input.shape[0]
	cdef int C = grad_input.shape[1]
	cdef int input_h = grad_input.shape[2]
	cdef int input_w = grad_input.shape[3]
	cdef int output_h = grad_output.shape[2]
	cdef int output_w = grad_output.shape[3]
	bilinear_backward_kernel(&grad_output[0,0,0,0], &grad_input[0,0,0,0], output_h, output_w, input_h, input_w, C, N)
	return grad_input






