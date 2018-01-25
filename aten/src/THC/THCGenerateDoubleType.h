#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateDoubleType.h"
#endif

#define ntype double
#define accntype double
#define Ntype Double
#define CReal CudaDouble
#define THC_NTYPE_IS_DOUBLE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THC_NTYPE_IS_DOUBLE

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
