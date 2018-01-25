#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LeakyReLU.c"
#else

void THNN_(LeakyReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accntype negval_,
          bool inplace)
{
  ntype negval = TH_CONVERT_ACCNTYPE_TO_NTYPE(negval_);
  if (inplace)
  {
    TH_TENSOR_APPLY(ntype, input,
      if (*input_data <= 0)
        *input_data *= negval;
    );
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(ntype, output, ntype, input,
      *output_data = *input_data > 0 ? *input_data : *input_data * negval;
    );
  }
}

void THNN_(LeakyReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accntype negval_,
          bool inplace)
{
  ntype negval = TH_CONVERT_ACCNTYPE_TO_NTYPE(negval_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (inplace)
  {
    TH_TENSOR_APPLY2(ntype, gradOutput, ntype, input,
      if (*input_data <= 0)
        *gradOutput_data *= negval;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(ntype, gradInput, ntype, gradOutput, ntype, input,
      *gradInput_data = *input_data > 0 ? *gradOutput_data : *gradOutput_data * negval;
    );
  }
}

#endif
