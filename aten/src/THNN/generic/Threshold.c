#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Threshold.c"
#else

void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accntype threshold_,
          accntype val_,
          bool inplace)
{
  ntype threshold = TH_CONVERT_ACCNTYPE_TO_NTYPE(threshold_);
  ntype val = TH_CONVERT_ACCNTYPE_TO_NTYPE(val_);
  if (inplace)
  {
    TH_TENSOR_APPLY(ntype, input,
      if (*input_data <= threshold)
        *input_data = val;
    );
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(ntype, output, ntype, input,
      *output_data = (*input_data > threshold) ? *input_data : val;
    );
  }
}

void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accntype threshold_,
          accntype val_,
          bool inplace)
{
  ntype threshold = TH_CONVERT_ACCNTYPE_TO_NTYPE(threshold_);
  ntype val = TH_CONVERT_ACCNTYPE_TO_NTYPE(val_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (inplace)
  {
    TH_TENSOR_APPLY2(ntype, gradOutput, ntype, input,
      if ((*input_data) <= threshold)
        *gradOutput_data = 0;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(ntype, gradInput, ntype, gradOutput, ntype, input,
      if ((*input_data) > threshold)
        *gradInput_data = *gradOutput_data;
      else
        *gradInput_data = 0;
    );
  }
}

#endif
