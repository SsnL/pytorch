#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateIntType.h"
#endif

#define ntype int32_t
#define accntype int64_t
#define Ntype Int
#define CReal CudaInt
#define THC_NTYPE_IS_INT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THC_NTYPE_IS_INT

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
