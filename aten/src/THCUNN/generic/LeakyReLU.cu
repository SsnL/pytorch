#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LeakyReLU.cu"
#else

#include "../common.h"

void THNN_(LeakyReLU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accntype negval_,
           bool inplace)
{
  ntype negval = ScalarConvert<accntype, ntype>::to(negval_);

  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input, LeakyReLUUpdateOutputIP<ntype>(negval));
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2(state, output, input, LeakyReLUUpdateOutput<ntype>(negval));
  }

  THCudaCheck(cudaGetLastError());
}

void THNN_(LeakyReLU_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accntype negval_,
           bool inplace)
{
  ntype negval = ScalarConvert<accntype, ntype>::to(negval_);

  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2(state, gradOutput, input, LeakyReLUUpdateGradInputIP<ntype>(negval));
    THCTensor_(set)(state, gradInput, gradOutput);
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3(state, gradInput, input, gradOutput, LeakyReLUUpdateGradInput<ntype>(negval));
  }

  THCudaCheck(cudaGetLastError());
}

#endif
