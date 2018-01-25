#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/BCECriterion.cu"
#else

#include "THCApply.cuh"

void THNN_(BCECriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           THCTensor *weights,
           bool reduce)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_nElement(state, input, weights);
  THCUNN_assertSameGPU(state, 3, input, target, weights);

  if (!reduce) {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply3(state, input, target, output,
        bce_updateOutput_no_reduce_functor<ntype, accntype>());
    if (weights) {
      THCTensor_(cmul)(state, output, output, weights);
    }
    return;
  }

  THCTensor_(resize1d)(state, output, 1);
  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<ntype> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<ntype> target_data(THCTensor_(data)(state, target));

  accntype sum;
  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    thrust::device_ptr<ntype> weights_data(THCTensor_(data)(state, weights));
    sum = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      bce_functor_weights<ntype, accntype>(),
      (accntype) 0,
      thrust::plus<accntype>()
    );
    THCTensor_(free)(state, weights);
  } else {
    sum = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      bce_functor<ntype, accntype>(),
      (accntype) 0,
      thrust::plus<accntype>()
    );
  }

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accntype, ntype>::to(sum));
}

void THNN_(BCECriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           bool sizeAverage,
           THCTensor *weights,
           bool reduce)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_nElement(state, input, weights);
  THCUNN_assertSameGPU(state, 4, input, target, gradInput, weights);

  THCTensor_(resizeAs)(state, gradInput, input);

  if (!reduce) {
    THCUNN_check_nElement(state, gradOutput, input);
    THC_pointwiseApply3(state, input, target, gradInput,
        bce_updateGradInput_no_reduce_functor<ntype, accntype>());
    THCTensor_(cmul)(state, gradInput, gradInput, gradOutput);
    if (weights) {
      THCTensor_(cmul)(state, gradInput, gradInput, weights);
    }
    return;
  }

  THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  ntype norm = ScalarConvert<accntype, ntype>::to((sizeAverage ? accntype(1)/size : accntype(1)) * THCTensor_(get1d)(state, gradOutput, 0));

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<ntype> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<ntype> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<ntype> gradInput_data(THCTensor_(data)(state, gradInput));

  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    thrust::device_ptr<ntype> weights_data(THCTensor_(data)(state, weights));
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      gradInput_data,
      bce_updateGradInput_functor_weights<ntype, accntype>(norm)
    );
    THCTensor_(free)(state, weights);
  } else {
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      gradInput_data,
      bce_updateGradInput_functor<ntype, accntype>(norm)
    );
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
