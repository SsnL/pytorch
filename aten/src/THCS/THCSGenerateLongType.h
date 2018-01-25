#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateLongType.h"
#endif

#define ntype int64_t
#define accntype int64_t
#define Ntype Long
#define CReal CudaLong
#define THCS_NTYPE_IS_LONG
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THCS_NTYPE_IS_LONG

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
