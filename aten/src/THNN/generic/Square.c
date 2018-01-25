#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Square.c"
#else

void THNN_(Square_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  
  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(ntype, output, ntype, input,
      *output_data = (*input_data) * (*input_data);
    );
  }
  else
  {
    ntype *output_data = THTensor_(data)(output);
    ntype *input_data  = THTensor_(data)(input);
    int64_t i;
#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(input); i++)
      output_data[i] = input_data[i]*input_data[i];
  }
}

void THNN_(Square_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  THNN_CHECK_SHAPE(input, gradOutput);
  THTensor_(resizeAs)(gradInput, input);

  if (input->nDimension == 1 || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(ntype, gradInput, ntype, gradOutput, ntype, input,
      *gradInput_data  = 2.0 * (*gradOutput_data) * (*input_data);
    );
  }
  else
  {
    ntype *gradOutput_data = THTensor_(data)(gradOutput);
    ntype *gradInput_data  = THTensor_(data)(gradInput);
    ntype *input_data  = THTensor_(data)(input);
    int64_t i;
#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(gradInput); i++)
      gradInput_data[i] = 2.0 * gradOutput_data[i] * input_data[i];
  }
}

#endif
