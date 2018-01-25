#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sqrt.c"
#else

void THNN_(Sqrt_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accntype eps_)
{
  ntype eps = TH_CONVERT_ACCNTYPE_TO_NTYPE(eps_);
  THTensor_(resizeAs)(output, input);
  THTensor_(sqrt)(output, input);
}

void THNN_(Sqrt_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_SHAPE(output, gradOutput);
  THTensor_(resizeAs)(gradInput, input);

  if (output->nDimension == 1 ||
      !THTensor_(isContiguous)(output) ||
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(ntype, gradInput, ntype, gradOutput, ntype, output,
      *gradInput_data = (*output_data == 0.0) ? 0.0 : (0.5 * (*gradOutput_data / *output_data));
    );
  }
  else
  {
    ntype *gradOutput_data = THTensor_(data)(gradOutput);
    ntype *gradInput_data  = THTensor_(data)(gradInput);
    ntype *output_data     = THTensor_(data)(output);
    int64_t i;
#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(output); i++)
    {
      if (output_data[i] == 0.0)
        gradInput_data[i] = 0.0;
      else
        gradInput_data[i] = 0.5 * (gradOutput_data[i] / output_data[i]);
    }
  }
}

#endif
