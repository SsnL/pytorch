#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/DistKLDivCriterion.cu"
#else

#include "THCApply.cuh"

void THNN_(DistKLDivCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           bool reduce)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 2, input, target);

  THArgCheck(THCTensor_(nElement)(state, input) == THCTensor_(nElement)(state, target), 2,
             "input and target need to have the same number of elements");

  if (!reduce) {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply3(state, input, target, output,
                        kl_updateOutput_no_reduce_functor<ntype>());
    return;
  }

  THCTensor_(resize1d)(state, output, 1);

  accntype sum;

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<ntype> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<ntype> target_data(THCTensor_(data)(state, target));
  sum = thrust::inner_product(input_data, input_data+size, target_data, (accntype) 0, thrust::plus<accntype>(), kl_functor<ntype, accntype>());

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accntype, ntype>::to(sum));
}

void THNN_(DistKLDivCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           bool sizeAverage,
           bool reduce)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 4, input, target, gradInput, gradOutput);

  THArgCheck(THCTensor_(nElement)(state, input) == THCTensor_(nElement)(state, target), 2,
             "input and target need to have the same number of elements");

  THCTensor_(resizeAs)(state, gradInput, input);

  if (!reduce) {
    THCUNN_check_nElement(state, gradOutput, input);
    THC_pointwiseApply3(state, target, gradOutput, gradInput,
                        kl_updateGradInput_no_reduce_functor<ntype>());
    return;
  }

  THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  ntype norm = (sizeAverage ? ScalarConvert<accntype, ntype>::to(accntype(1)/size) : ScalarConvert<int, ntype>::to(1));

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<ntype> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<ntype> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<ntype> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data,
                    kl_updateGradInput_functor<ntype>(norm, THCTensor_(get1d)(state, gradOutput, 0)));

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
