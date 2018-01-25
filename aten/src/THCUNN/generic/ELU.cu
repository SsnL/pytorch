#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/ELU.cu"
#else

#include "../common.h"


void THNN_(ELU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accntype alpha,
           accntype scale,
           bool inplace)
{
  ntype negcoef = ScalarConvert<accntype, ntype>::to(alpha * scale);
  ntype poscoef = ScalarConvert<accntype, ntype>::to(scale);
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input, ELUupdateOutputIP_functor<ntype>(negcoef, poscoef));
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2(state, output, input, ELUupdateOutput_functor<ntype>(negcoef, poscoef));
  }
}


void THNN_(ELU_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           accntype alpha,
           accntype scale)
{
  ntype negcoef = ScalarConvert<accntype, ntype>::to(alpha * scale);
  ntype poscoef = ScalarConvert<accntype, ntype>::to(scale);
  THCUNN_check_nElement(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, ELUupdateGradInput_functor<ntype>(negcoef, poscoef));
}

#endif
