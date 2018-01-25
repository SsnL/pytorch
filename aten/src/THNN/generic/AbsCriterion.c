#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/AbsCriterion.c"
#else

void THNN_(AbsCriterion_updateOutput)(
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
      *output_data = fabs(*input_data - *target_data);
    );
    return;
  }

  ntype sum = 0;
  THTensor_(resize1d)(output, 1);
  TH_TENSOR_APPLY2(ntype, input, ntype, target,
    sum += fabs(*input_data - *target_data);
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(AbsCriterion_updateGradInput)(
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
    THNN_CHECK_NELEMENT(gradOutput, input);
    TH_TENSOR_APPLY3(ntype, gradInput, ntype, input, ntype, target,
      *gradInput_data = ((*input_data - *target_data) >= 0 ? 1 : -1);
    );
    TH_TENSOR_APPLY2(ntype, gradInput, ntype, gradOutput,
      *gradInput_data *= *gradOutput_data;
    );
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
  ntype norm = (sizeAverage ? 1./((ntype)THTensor_(nElement)(input)) : 1.) * THTensor_fastGet1d(gradOutput, 0);

  TH_TENSOR_APPLY3(ntype, gradInput, ntype, input, ntype, target,
    *gradInput_data = (*input_data - *target_data) >= 0 ? norm : -norm;
  );
}

#endif
