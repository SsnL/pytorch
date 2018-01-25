#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateShortType.h"
#endif

#define ntype int16_t
#define accntype int64_t
#define Ntype Short
#define CReal CudaShort
#define THC_NTYPE_IS_SHORT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THC_NTYPE_IS_SHORT

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
