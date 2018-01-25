#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MSECriterion.c"
#else

void THNN_(MSECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          bool reduce)
{
  THNN_CHECK_NELEMENT(input, target);

  if (reduce) {
    THTensor_(resize1d)(output, 1);

    ntype sum = 0;

    TH_TENSOR_APPLY2(ntype, input, ntype, target,
      ntype z = (*input_data - *target_data);
      sum += z*z;
    );

    if (sizeAverage)
      sum /= THTensor_(nElement)(input);

    THTensor_(set1d)(output, 0, sum);
    return;
  }

  THTensor_(resizeAs)(output, input);
  TH_TENSOR_APPLY3(ntype, input, ntype, target, ntype, output,
      ntype z = (*input_data - *target_data);
      *output_data = z*z;
  );
}

void THNN_(MSECriterion_updateGradInput)(
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

  if (reduce) {
    THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
    ntype norm = sizeAverage ? 2./((ntype)THTensor_(nElement)(input)) : 2.;
    norm *= THTensor_(get1d)(gradOutput, 0);
    TH_TENSOR_APPLY3(ntype, gradInput, ntype, input, ntype, target,
      *gradInput_data = norm * (*input_data - *target_data);
    );
    return;
  }

  THNN_CHECK_NELEMENT(input, gradOutput);
  TH_TENSOR_APPLY3(ntype, gradInput, ntype, input, ntype, target,
    *gradInput_data = 2. * (*input_data - *target_data);
  );
  TH_TENSOR_APPLY2(ntype, gradInput, ntype, gradOutput,
    *gradInput_data *= *gradOutput_data;
  );
}

#endif
