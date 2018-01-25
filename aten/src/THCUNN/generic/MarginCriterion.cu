#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MarginCriterion.cu"
#else

void THNN_(MarginCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           accntype margin_)
{
  ntype margin = ScalarConvert<accntype, ntype>::to(margin_);
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_dim_size(state, output, 1, 0, 1);
  THCUNN_assertSameGPU(state, 2, input, target);

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<ntype> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<ntype> target_data(THCTensor_(data)(state, target));
  accntype sum = thrust::inner_product(input_data, input_data+size, target_data, (accntype) 0, thrust::plus<accntype>(),
      margin_functor<ntype, accntype>(ScalarConvert<ntype, accntype>::to(margin)));

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accntype, ntype>::to(sum));
}


void THNN_(MarginCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradInput,
           bool sizeAverage,
           accntype margin_)
{
  ntype margin = ScalarConvert<accntype, ntype>::to(margin_);

  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  accntype norm = sizeAverage ? 1.f/size : 1;

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  THCTensor_(resizeAs)(state, gradInput, input);

  thrust::device_ptr<ntype> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<ntype> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<ntype> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data,
      margin_updateGradInput_functor<ntype, accntype>(ScalarConvert<ntype, accntype>::to(margin), norm));

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
