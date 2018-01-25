#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateDoubleType.h"
#endif

#define ntype double
#define accntype double
#define Ntype Double
#define CReal CudaDouble
#define THCS_NTYPE_IS_DOUBLE
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THCS_NTYPE_IS_DOUBLE

#ifndef THCSGenerateAllTypes
#ifndef THCSGenerateFloatTypes
#undef THCS_GENERIC_FILE
#endif
#endif
