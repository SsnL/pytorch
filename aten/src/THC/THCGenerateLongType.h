#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateLongType.h"
#endif

#define ntype int64_t
#define accntype int64_t
#define Ntype Long
#define CReal CudaLong
#define THC_NTYPE_IS_LONG
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THC_NTYPE_IS_LONG

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
