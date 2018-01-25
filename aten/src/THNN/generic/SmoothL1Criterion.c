#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SmoothL1Criterion.c"
#else

void THNN_(SmoothL1Criterion_updateOutput)(
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
      ntype z = fabs(*input_data - *target_data);
      *output_data = z < 1 ? 0.5 * z * z : z - 0.5;
    );
    return;
  }

  THTensor_(resize1d)(output, 1);

  ntype sum = 0;
  TH_TENSOR_APPLY2(ntype, input, ntype, target,
    ntype z = fabs(*input_data - *target_data);
    sum += z < 1 ? 0.5*z*z : z - 0.5;
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(SmoothL1Criterion_updateGradInput)(
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
      ntype x = *input_data - *target_data;
      if (x < -1.) {
        *gradInput_data = -1.;
      } else if (x > 1.) {
        *gradInput_data = 1.;
      } else {
        *gradInput_data = x;
      }
    );
    TH_TENSOR_APPLY2(ntype, gradInput, ntype, gradOutput,
      *gradInput_data *= *gradOutput_data;
    );
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
  ntype norm = (sizeAverage ? 1./((ntype)THTensor_(nElement)(input)) : 1.) * THTensor_fastGet1d(gradOutput, 0);

  TH_TENSOR_APPLY3(ntype, gradInput, ntype, input, ntype, target,
    ntype x = *input_data - *target_data;
    if (x < -1.)
     *gradInput_data = - norm;
    else if (x > 1.)
     *gradInput_data = norm;
    else
     *gradInput_data = norm * x;
  );
}

#endif
