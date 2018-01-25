#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ELU.c"
#else

void THNN_(ELU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accntype alpha_,
          accntype scale,
          bool inplace)
{
  ntype negcoef = TH_CONVERT_ACCNTYPE_TO_NTYPE(alpha_ * scale);
  ntype poscoef = TH_CONVERT_ACCNTYPE_TO_NTYPE(scale);
  if (inplace) {
    TH_TENSOR_APPLY(ntype, input,
      *input_data = *input_data <= 0 ? (exp(*input_data)-1) * negcoef : *input_data * poscoef;
    );
    THTensor_(set)(output, input);
  } else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(ntype, input, ntype, output,
      *output_data = *input_data <= 0 ? (exp(*input_data)-1) * negcoef : *input_data * poscoef;
    );
  }
}

void THNN_(ELU_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          accntype alpha_,
          accntype scale)
{
  ntype negcoef = TH_CONVERT_ACCNTYPE_TO_NTYPE(alpha_ * scale);
  ntype poscoef = TH_CONVERT_ACCNTYPE_TO_NTYPE(scale);
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(ntype, gradInput, ntype, gradOutput, ntype, output,
    *gradInput_data = *output_data <= 0 ? *gradOutput_data * (*output_data + negcoef) : *gradOutput_data * poscoef;
  );
}

#endif
