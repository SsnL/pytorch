#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define ntype float
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accntype float
#define Ntype Float
#define CReal Cuda
#define THCS_NTYPE_IS_FLOAT
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THCS_NTYPE_IS_FLOAT

#ifndef THCSGenerateAllTypes
#ifndef THCSGenerateFloatTypes
#undef THCS_GENERIC_FILE
#endif
#endif
