#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/L1Cost.cu"
#else

void THNN_(L1Cost_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output)
{
  THCUNN_check_dim_size(state, output, 1, 0, 1);
  THCUNN_assertSameGPU(state, 1, input);
  accntype sum;
  ptrdiff_t size = THCTensor_(nElement)(state, input);
  input = THCTensor_(newContiguous)(state, input);
  thrust::device_ptr<ntype> input_data(THCTensor_(data)(state, input));
  sum = thrust::transform_reduce(input_data, input_data+size, l1cost_functor<ntype, accntype>(), accntype(0), thrust::plus<accntype>());

  THCTensor_(free)(state, input);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accntype, ntype>::to(sum));
}

void THNN_(L1Cost_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 2, input, gradInput);
  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  THCTensor_(resizeAs)(state, gradInput, input);

  thrust::device_ptr<ntype> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<ntype> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(input_data, input_data+size, gradInput_data, l1cost_updateGradInput_functor<ntype>());

  THCTensor_(free)(state, input);
}

#endif
