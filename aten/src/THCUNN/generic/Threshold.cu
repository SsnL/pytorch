#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Threshold.cu"
#else

#include "../common.h"

void THNN_(Threshold_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accntype threshold_,
           accntype val_,
           bool inplace)
{
  ntype threshold = ScalarConvert<accntype, ntype>::to(threshold_);
  ntype val = ScalarConvert<accntype, ntype>::to(val_);
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input,
      ThresholdUpdateOutputIP<ntype>(threshold, val)
    );
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2(state, output, input,
      ThresholdUpdateOutput<ntype>(threshold, val)
    );
  }

  THCudaCheck(cudaGetLastError());
}

void THNN_(Threshold_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accntype threshold_,
           accntype val_,
           bool inplace)
{
  ntype threshold = ScalarConvert<accntype, ntype>::to(threshold_);
  ntype val = ScalarConvert<accntype, ntype>::to(val_);
  (void) val;
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2(state, gradOutput, input,
      ThresholdUpdateGradInputIP<ntype>(threshold)
    );
    THCTensor_(set)(state, gradInput, gradOutput);
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3(state, gradInput, input, gradOutput,
       ThresholdUpdateGradInput<ntype>(threshold)
    );
  }

  THCudaCheck(cudaGetLastError());
}

#endif
