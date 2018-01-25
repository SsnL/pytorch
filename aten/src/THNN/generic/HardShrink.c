#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/HardShrink.c"
#else

void THNN_(HardShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accntype lambda_)
{
  ntype lambda = TH_CONVERT_ACCNTYPE_TO_NTYPE(lambda_);
  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(ntype, output, ntype, input,
    if (*input_data > lambda)
      *output_data = *input_data;
    else if (*input_data < -lambda)
      *output_data = *input_data;
    else
      *output_data = 0;
  );
}

void THNN_(HardShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accntype lambda_)
{
  ntype lambda = TH_CONVERT_ACCNTYPE_TO_NTYPE(lambda_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(ntype, gradInput, ntype, gradOutput, ntype, input,
    if (*input_data > lambda || *input_data < -lambda)
      *gradInput_data = *gradOutput_data;
    else
      *gradInput_data = 0;
  );
}

#endif
