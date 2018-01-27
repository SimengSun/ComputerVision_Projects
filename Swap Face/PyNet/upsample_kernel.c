/*Credit: Heavily referenced from torch spatialbilinearupsample implementation"*/

void bilinear_forward_kernel(float* input, 
					  float* output, 
					  int output_h, 
					  int output_w, 
					  int input_h, 
					  int input_w,
					  int nchannel,
					  int nbatch
					  )
{
	const int channel = nbatch * nchannel;
	const float h_ratio = (output_h > 1) ? (float)(input_h - 1)/(output_h - 1) : 0.f;
	const float w_ratio = (output_w > 1) ? (float)(input_w - 1)/(output_w - 1) : 0.f;
	int h2 = 0;
	for (h2 = 0; h2 < output_h; ++h2) {
	const float h1r = h_ratio * h2;
	const int h1 = h1r;
	const int h1p = (h1 < input_h - 1) ? 1 : 0;
	const float h1lambda = h1r - h1;
	const float h0lambda = (float)1. - h1lambda;
	int w2 = 0;
	for (w2 = 0; w2 < output_w; ++w2) {
	  const float w1r = w_ratio * w2;
	  const int w1 = w1r;
	  const int w1p = (w1 < input_w - 1) ? 1 : 0;
	  const float w1lambda = w1r - w1;
	  const float w0lambda = (float)1. - w1lambda;
	  const float* pos1 = &input[h1 * input_w + w1];
	  float* pos2 = &output[h2 * output_w + w2];
	  int c = 0;
	  for (c = 0; c < channel; ++c) {
		pos2[0] = h0lambda * (w0lambda * pos1[0]+ w1lambda * pos1[w1p])
				  + h1lambda * (w0lambda * pos1[h1p * input_w]
				  + w1lambda * pos1[h1p * input_w + w1p]);
		pos1 += input_w * input_h;
		pos2 += output_w * output_h;
	  }
	}
}
};

void bilinear_backward_kernel(float* grad_output, 
					   float* grad_input, 
					  int output_h, 
					  int output_w, 
					  int input_h, 
					  int input_w,
					  int nchannel,
					  int nbatch
					  )
{ 
  const int channel = nbatch * nchannel;
  const float rheight =(output_h > 1) ? (float)(input_h - 1)/(output_h - 1) : 0.f;
  const float rwidth = (output_w > 1) ? (float)(input_w - 1)/(output_w - 1) : 0.f;
  int h2 = 0;
  for (h2 = 0; h2 < output_h; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < input_h - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = (float)1. - h1lambda;
    int w2 = 0;
    for (w2 = 0; w2 < output_w; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < input_w - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = (float)1. - w1lambda;
      float* pos1 = &grad_input[h1 * input_w + w1];
      const float* pos2 = &grad_output[h2 * output_w + w2];
      int c = 0;
      for (c = 0; c < channel; ++c) {
        pos1[0] += h0lambda * w0lambda * pos2[0];
        pos1[w1p] += h0lambda * w1lambda * pos2[0];
        pos1[h1p * input_w] += h1lambda * w0lambda * pos2[0];
        pos1[h1p * input_w + w1p] += h1lambda * w1lambda * pos2[0];
        pos1 += input_w * input_h;
        pos2 += output_w * output_h;
      }
    }
}
};