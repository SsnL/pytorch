#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RReLU.c"
#else

void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          accntype lower_,
          accntype upper_,
          bool train,
          bool inplace,
          THGenerator *generator)
{
  ntype lower = TH_CONVERT_ACCNTYPE_TO_NTYPE(lower_);
  ntype upper = TH_CONVERT_ACCNTYPE_TO_NTYPE(upper_);
  if (train)
  {
    // get default random generator
    THTensor_(resizeAs)(noise, input);
    if (inplace)
    {
      TH_TENSOR_APPLY2(ntype, input, ntype, noise,
        if (*input_data <= 0)
        {
          const ntype r = (ntype)THRandom_uniform(generator, lower, upper);
          *input_data = (*input_data) * r;
          *noise_data = r;
        }
        else
        {
          *noise_data = 1;
        }
      );
      THTensor_(set)(output, input);
    }
    else
    {
      THTensor_(resizeAs)(output, input);
      TH_TENSOR_APPLY3(ntype, input, ntype, output, ntype, noise,
        if (*input_data <= 0)
        {
          const ntype r = (ntype)THRandom_uniform(generator, lower, upper);
          *output_data = (*input_data) * r;
          *noise_data = r;
        }
        else
        {
          *output_data = *input_data;
          *noise_data = 1;
        }
      );
    }
  }
  else
  {
    const ntype negSlope = (lower + upper) / 2;
    if (inplace)
    {
      TH_TENSOR_APPLY(ntype, input,
        if (*input_data <= 0)
        {
          *input_data = *input_data * negSlope;
        }
      );
      THTensor_(set)(output, input);
    }
    else
    {
      THTensor_(resizeAs)(output, input);
      TH_TENSOR_APPLY2(ntype, input, ntype, output,
        const ntype r = (*input_data) <= 0 ? negSlope : 1;
        *output_data = *input_data * r;
      );
    }
  }
}

void THNN_(RReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          accntype lower_,
          accntype upper_,
          bool train,
          bool inplace)
{
  ntype lower = TH_CONVERT_ACCNTYPE_TO_NTYPE(lower_);
  ntype upper = TH_CONVERT_ACCNTYPE_TO_NTYPE(upper_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (train && upper - lower > 1E-6)    // e.g. if upper == lower, RReLU behaves like LeakyReLU
  {
    // multiply the gradient by the noise tensor
    if (inplace)
    {
      THTensor_(cmul)(gradOutput, gradOutput, noise);
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
      THTensor_(cmul)(gradInput, gradOutput, noise);
    }
  }
  else
  {
    // use constant factor for negative input values
    const ntype negSlope = (lower + upper) / 2;
    if (inplace)
    {
      TH_TENSOR_APPLY2(ntype, gradOutput, ntype, input,
        if (*input_data <= 0)
        {
          *gradOutput_data = (*gradOutput_data) * negSlope;
        }
      );
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
      TH_TENSOR_APPLY3(ntype, gradInput, ntype, gradOutput, ntype, input,
        *gradInput_data = (*input_data) <= 0 ? (*gradOutput_data) * negSlope : (*gradOutput_data);
      );
    }
  }
}

#endif
