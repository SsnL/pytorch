#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMarginCriterion.c"
#else

void THNN_(SoftMarginCriterion_updateOutput)(
  THNNState *state,
  THTensor *input,
  THTensor *target,
  THTensor *output,
  bool sizeAverage)
{
  THNN_CHECK_NELEMENT(input, target);
  THTensor_(resize1d)(output, 1);

  ntype sum;

  sum = 0;
  TH_TENSOR_APPLY2(ntype, input, ntype, target,
                   ntype z = log(1. + exp(-*input_data* *target_data));
                   sum += z;)

  if(sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(SoftMarginCriterion_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *target,
  THTensor *gradInput,
  bool sizeAverage)
{
  THNN_CHECK_NELEMENT(input, target);
  ntype norm = (sizeAverage ? 1./((ntype)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(ntype, gradInput, ntype, input, ntype, target,
                   ntype z = exp(-*target_data * *input_data);
                   *gradInput_data = -norm*(*target_data)*z/(1. + z);)
}

#endif
