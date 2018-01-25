#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateHalfType.h"
#endif

#include "THCHalf.h"

#if defined(CUDA_HALF_TENSOR) || defined(FORCE_TH_HALF)

#define ntype half
#define accntype float
#define Ntype Half

// if only here via FORCE_TH_HALF, don't define CReal since
// FORCE_TH_HALF should only be used for TH types
#ifdef CUDA_HALF_TENSOR
#define CReal CudaHalf
#endif

#define THC_NTYPE_IS_HALF
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype

#ifdef CUDA_HALF_TENSOR
#undef CReal
#endif

#undef THC_NTYPE_IS_HALF

#endif // defined(CUDA_HALF_TENSOR) || defined(FORCE_TH_HALF)

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
