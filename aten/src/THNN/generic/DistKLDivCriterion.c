#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/DistKLDivCriterion.c"
#else

void THNN_(DistKLDivCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          bool reduce)
{
  THNN_CHECK_NELEMENT(input, target);

  if (!reduce) {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY3(ntype, input, ntype, target, ntype, output,
      *output_data = *target_data > 0 ? *target_data * (log(*target_data) - *input_data) : 0;
    );
    return;
  }

  THTensor_(resize1d)(output, 1);

  ntype sum = 0;

  TH_TENSOR_APPLY2(ntype, input, ntype, target,
    sum += *target_data > 0 ? *target_data * (log(*target_data) - *input_data) : 0;
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(DistKLDivCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          bool sizeAverage,
          bool reduce)
{
  THNN_CHECK_NELEMENT(input, target);
  THTensor_(resizeAs)(gradInput, input);

  if (!reduce) {
    THNN_CHECK_NELEMENT(input, gradOutput);
    TH_TENSOR_APPLY3(ntype, gradInput, ntype, gradOutput, ntype, target,
      *gradInput_data = *target_data > 0 ? (-*target_data) * *gradOutput_data : 0;
    );
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);

  ntype norm = (sizeAverage ? 1./((ntype)THTensor_(nElement)(input)) : 1.);

  TH_TENSOR_APPLY3(ntype, gradInput, ntype, input, ntype, target,
    *gradInput_data = *target_data > 0 ? norm * (-*target_data) * THTensor_fastGet1d(gradOutput, 0) : 0;
  );
}

#endif
