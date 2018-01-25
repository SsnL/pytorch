#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define ntype float
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accntype float
#define Ntype Float
#define CReal Cuda
#define THC_NTYPE_IS_FLOAT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THC_NTYPE_IS_FLOAT

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
