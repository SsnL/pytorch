#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateShortType.h"
#endif

#define ntype int16_t
#define accntype int64_t
#define Ntype Short
#define CReal CudaShort
#define THCS_NTYPE_IS_SHORT
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THCS_NTYPE_IS_SHORT

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
