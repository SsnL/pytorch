#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MarginCriterion.c"
#else

void THNN_(MarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          accntype margin_)
{
  ntype margin = TH_CONVERT_ACCNTYPE_TO_NTYPE(margin_);
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  ntype sum = 0;

  TH_TENSOR_APPLY2(ntype, input, ntype, target,
    ntype z = (margin - *input_data * *target_data);
    sum += z>0 ? z : 0;
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(MarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          accntype margin_)
{
  ntype margin = TH_CONVERT_ACCNTYPE_TO_NTYPE(margin_);
  THNN_CHECK_NELEMENT(input, target);
  ntype norm = (sizeAverage ? 1./((ntype)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(ntype, gradInput, ntype, input, ntype, target,
    *gradInput_data = (*input_data * *target_data) < margin ? -norm * *target_data : 0;
  );
}

#endif
