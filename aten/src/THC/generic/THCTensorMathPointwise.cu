#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPointwise.cu"
#else

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_(NAME, CFUNC, NTYPE)           \
  struct Tensor_##NAME##_##NTYPE##_Op {                                 \
    __device__ __forceinline__ void operator()(ntype* out, ntype* in) const { \
      *out = CFUNC(*in);                                                \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(ntype* v) const {         \
      *v = CFUNC(*v);                                                   \
    }                                                                   \
  };                                                                    \
                                                                        \
  void THCTensor_(NAME)(THCState* state, THCTensor* self_, THCTensor* src) { \
    THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));               \
    if (self_ == src) {                                                 \
      if (!THC_pointwiseApply1(state, self_, Tensor_##NAME##_##NTYPE##_Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);                      \
      }                                                                 \
    } else {                                                            \
      THCTensor_(resizeAs)(state, self_, src);                          \
                                                                        \
      if (!THC_pointwiseApply2(state, self_, src, Tensor_##NAME##_##NTYPE##_Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);                      \
      }                                                                 \
    }                                                                   \
                                                                        \
    THCudaCheck(cudaGetLastError());                                    \
  }

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC, NTYPE) \
  IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_(NAME, CFUNC, NTYPE)

#if defined(THC_NTYPE_IS_FLOAT) || defined(THC_NTYPE_IS_DOUBLE) || defined(THC_NTYPE_IS_HALF)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  log, THCNumerics<ntype>::log,   Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(lgamma, THCNumerics<ntype>::lgamma, Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, THCNumerics<ntype>::log1p, Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  exp, THCNumerics<ntype>::exp,   Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(expm1, THCNumerics<ntype>::expm1, Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  cos, THCNumerics<ntype>::cos,   Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  sin, THCNumerics<ntype>::sin,   Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( sqrt, THCNumerics<ntype>::sqrt,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(rsqrt, THCNumerics<ntype>::rsqrt, Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( ceil, THCNumerics<ntype>::ceil,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, THCNumerics<ntype>::floor, Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(trunc, THCNumerics<ntype>::trunc, Ntype)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  acos, THCNumerics<ntype>::acos,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  cosh, THCNumerics<ntype>::cosh,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  asin, THCNumerics<ntype>::asin,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  sinh, THCNumerics<ntype>::sinh,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(   tan, THCNumerics<ntype>::tan,   Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  atan, THCNumerics<ntype>::atan,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  tanh, THCNumerics<ntype>::tanh,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(   erf, THCNumerics<ntype>::erf,   Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(erfinv, THCNumerics<ntype>::erfinv,Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( round, THCNumerics<ntype>::round, Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  frac, THCNumerics<ntype>::frac,  Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  cinv, THCNumerics<ntype>::cinv,  Ntype)

#endif

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  neg, THCNumerics<ntype>::neg,   Ntype)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  abs, THCNumerics<ntype>::abs,   Ntype)

#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_
#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC

void THCTensor_(sign)(THCState* state, THCTensor* self_, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorSignOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorSignOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(clamp)(THCState *state, THCTensor *self_, THCTensor *src, ntype min_value,
  ntype max_value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorClampOp<ntype>(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorClampOp<ntype>(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cross)(THCState *state, THCTensor *self, THCTensor *x, THCTensor *y, int dimension)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, x, y));

  int i;
  int nd = THCTensor_(nDimension)(state, x);
  ptrdiff_t nelem = THCTensor_(nElement)(state, x);
  THArgCheck(nd == THCTensor_(nDimension)(state, y), 1, "tensors must have same number of dimensions");
  for (i = 0; i < nd; i++) {
    THArgCheck(THCTensor_(size)(state, x, i) == THCTensor_(size)(state, y, i), 1, "dimension %i of x and y does not match", i);
    if (dimension < 0 && THCTensor_(size)(state, x, i) == 3) {
      dimension = i;
    }
  }

  THArgCheck(dimension >= 0 && dimension < nd, 3, "dimension %d out of range", dimension+1);
  THArgCheck(THCTensor_(size)(state, x, dimension) == 3, 3,
      "dimension %d does not have size 3", dimension+1);
  THCTensor_(resizeAs)(state, self, x);

  int64_t sx = THCTensor_(stride)(state, x, dimension);
  int64_t sy = THCTensor_(stride)(state, y, dimension);
  int64_t so = THCTensor_(stride)(state, self, dimension);
  THCTensor *nx = THCTensor_(newNarrow)(state, x, dimension, 0, 1);
  THCTensor *ny = THCTensor_(newNarrow)(state, y, dimension, 0, 1);
  THCTensor *nself = THCTensor_(newNarrow)(state, self, dimension, 0, 1);
  if (!THC_pointwiseApply3(state, nself, nx, ny, TensorCrossOp<ntype>(sx, sy, so))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
  THCTensor_(free)(state, nx);
  THCTensor_(free)(state, ny);
  THCTensor_(free)(state, nself);
}

#if defined(THC_NTYPE_IS_FLOAT) || defined(THC_NTYPE_IS_DOUBLE) || defined(THC_NTYPE_IS_HALF)

void THCTensor_(atan2)(THCState *state, THCTensor *self_, THCTensor *tx, THCTensor *ty)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, tx, ty));
  THArgCheck(THCTensor_(nElement)(state, tx) ==
             THCTensor_(nElement)(state, ty), 3, "sizes do not match");
  THCTensor_(resizeAs)(state, self_, tx);

  if (!THC_pointwiseApply3(state, self_, tx, ty, TensorATan2Op<ntype>())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(sigmoid)(THCState* state, THCTensor* self_, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorSigmoidOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorSigmoidOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(digamma)(THCState* state, THCTensor* self_, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ != src) {
    THCTensor_(resizeAs)(state, self_, src);
  }
  if (!THC_pointwiseApply2(state, self_, src, TensorDigammaOp<ntype, accntype>())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(polygamma)(THCState* state, THCTensor* self_, int64_t n, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ != src) {
    THCTensor_(resizeAs)(state, self_, src);
  }
  switch (n) {
    case 0:
      if (!THC_pointwiseApply2(state, self_, src, TensorDigammaOp<ntype, accntype>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
      break;
    case 1:
      if (!THC_pointwiseApply2(state, self_, src, TensorTrigammaOp<ntype, accntype>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
      break;
    default:
      THError("polygamma(n,x) is not implemented for n>=2");
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(pow)(THCState *state, THCTensor *self_, THCTensor *src, ntype value) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(1))) {
      if (!THC_pointwiseApply1(state, self_, TensorPowOp<ntype, 1>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(2))) {
      if (!THC_pointwiseApply1(state, self_, TensorPowOp<ntype, 2>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(3))) {
      if (!THC_pointwiseApply1(state, self_, TensorPowOp<ntype, 3>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(-1))) {
      if (!THC_pointwiseApply1(state, self_, TensorPowOp<ntype, -1>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(-2))) {
      if (!THC_pointwiseApply1(state, self_, TensorPowOp<ntype, -2>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // fallback implementation using pow
      if (!THC_pointwiseApply1(state, self_, TensorPowOp<ntype, -3>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(1))) {
      if (!THC_pointwiseApply2(state, self_, src, TensorPowOp<ntype, 1>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(2))) {
      if (!THC_pointwiseApply2(state, self_, src, TensorPowOp<ntype, 2>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(3))) {
      if (!THC_pointwiseApply2(state, self_, src, TensorPowOp<ntype, 3>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(-1))) {
      if (!THC_pointwiseApply2(state, self_, src, TensorPowOp<ntype, -1>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<ntype>::eq(value, ScalarConvert<int, ntype>::to(-2))) {
      if (!THC_pointwiseApply2(state, self_, src, TensorPowOp<ntype, -2>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // fallback implementation using pow
      if (!THC_pointwiseApply2(state, self_, src, TensorPowOp<ntype, -3>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(tpow)(THCState *state, THCTensor *self_, ntype value, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorTPowOp<ntype>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorTPowOp<ntype>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(lerp)(THCState *state, THCTensor *result, THCTensor *a, THCTensor *b, ntype w)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, result, a, b));
  THArgCheck(THCTensor_(nElement)(state, a) ==
             THCTensor_(nElement)(state, b), 3, "sizes do not match");
  THCTensor_(resizeAs)(state, result, a);

  if (!THC_pointwiseApply3(state, result, a, b, TensorLerpOp<ntype>(w))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif

THC_API void
THCTensor_(cadd)(THCState *state, THCTensor *self_, THCTensor* src1, ntype value, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == ScalarConvert<int, ntype>::to(1)) {
      // self += src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorAddOp<ntype>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self += value * src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorCAddOp<ntype>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    if (value == ScalarConvert<int, ntype>::to(1)) {
      // self = src1 + src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddOp<ntype>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 + value * src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorCAddOp<ntype>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(csub)(THCState *state, THCTensor *self_, THCTensor* src1, ntype value, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == ScalarConvert<int, ntype>::to(1)) {
      // self -= src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorSubOp<ntype>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self += -value * src2
      if (!THC_pointwiseApply2(state, self_, src2,
                                   TensorCAddOp<ntype>(
                                     ScalarNegate<ntype>::to(value)))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    if (value == ScalarConvert<int, ntype>::to(1)) {
      // self = src1 - src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorSubOp<ntype>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 - value * src2
      if (!THC_pointwiseApply3(state, self_, src1, src2,
                                   TensorCAddOp<ntype>(
                                     ScalarNegate<ntype>::to(value)))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cmul)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorMulOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 * src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorMulOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cpow)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self = pow(self, src2)
    if (!THC_pointwiseApply2(state, self_, src2, TensorCPowOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = pow(src1, src2)
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorCPowOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cdiv)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorDivOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorDivOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(clshift)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_NTYPE_IS_HALF)
  return THError("clshift not supported for torch.CudaHalfTensor");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorLShiftOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorLShiftOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(crshift)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_NTYPE_IS_HALF)
  return THError("crshift not supported for torch.CudaHalfTensor");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorRShiftOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorRShiftOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(cmax)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorMaxOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorMaxOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cmin)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorMinOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorMinOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cremainder)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorCRemainderOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorCRemainderOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cfmod)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorCFmodOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorCFmodOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cmaxValue)(THCState *state, THCTensor *self, THCTensor *src, ntype value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1(state, self, TensorMaxValueOp<ntype>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src);
    if (!THC_pointwiseApply2(state, self, src, TensorMaxValueOp<ntype>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cminValue)(THCState *state, THCTensor *self, THCTensor *src, ntype value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1(state, self, TensorMinValueOp<ntype>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src);
    if (!THC_pointwiseApply2(state, self, src, TensorMinValueOp<ntype>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(addcmul)(THCState *state, THCTensor *self_, THCTensor *t, ntype value, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THCTensor_(resizeAs)(state, self_, t);
    THCTensor_(copy)(state, self_, t);
  }
  else
  {
    THArgCheck(THCTensor_(nElement)(state, self_) == THCTensor_(nElement)(state, src1),
               1, "sizes do not match");
  }

  THArgCheck(THCTensor_(nElement)(state, src1) == THCTensor_(nElement)(state, src2),
             3, "sizes do not match");

  if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddCMulOp<ntype>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(addcdiv)(THCState *state, THCTensor *self_, THCTensor *t, ntype value, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THCTensor_(resizeAs)(state, self_, t);
    THCTensor_(copy)(state, self_, t);
  }
  else
  {
    THArgCheck(THCTensor_(nElement)(state, self_) == THCTensor_(nElement)(state, src1),
               1, "sizes do not match");
  }
  THArgCheck(THCTensor_(nElement)(state, src1) == THCTensor_(nElement)(state, src2),
             3, "sizes do not match");

  if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddCDivOp<ntype>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cbitand)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_NTYPE_IS_HALF) || defined(THC_NTYPE_IS_FLOAT) || defined(THC_NTYPE_IS_DOUBLE)
  return THError("cbitand is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorBitAndOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorBitAndOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(cbitor)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_NTYPE_IS_HALF) || defined(THC_NTYPE_IS_FLOAT) || defined(THC_NTYPE_IS_DOUBLE)
  return THError("cbitor is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorBitOrOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorBitOrOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(cbitxor)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_NTYPE_IS_HALF) || defined(THC_NTYPE_IS_FLOAT) || defined(THC_NTYPE_IS_DOUBLE)
  return THError("cbitor is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorBitXorOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorBitXorOp<ntype>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}
#endif
