#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/HardTanh.cu"
#else

#include "../common.h"

void THNN_(HardTanh_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accntype min_val_,
           accntype max_val_,
           bool inplace)
{
  ntype min_val = ScalarConvert<accntype, ntype>::to(min_val_);
  ntype max_val = ScalarConvert<accntype, ntype>::to(max_val_);

  THCUNN_assertSameGPU(state, 2, input, output);
  if(inplace)
  {
    THCTensor_(set)(state, output, input);
    THC_pointwiseApply1(state, output, hardtanhupdateOutput_functor<ntype>(min_val, max_val));
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2(state, output, input,
                               hardtanhupdateOutput_functor<ntype>(min_val, max_val));
  }
}

void THNN_(HardTanh_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accntype min_val_,
           accntype max_val_,
           bool inplace)
{
  ntype min_val = ScalarConvert<accntype, ntype>::to(min_val_);
  ntype max_val = ScalarConvert<accntype, ntype>::to(max_val_);

  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  if (inplace)
  {
    THCTensor_(set)(state, gradInput, gradOutput);
    THC_pointwiseApply2(state, gradInput, input,
                                 hardtanhupdateGradInput_functor<ntype>(min_val, max_val));
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3(state, gradInput, input, gradOutput,
                                 hardtanhupdateGradInput_functor<ntype>(min_val, max_val));
  }
}

#endif
