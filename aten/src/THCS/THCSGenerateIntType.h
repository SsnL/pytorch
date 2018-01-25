#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateIntType.h"
#endif

#define ntype int32_t
#define accntype int64_t
#define Ntype Int
#define CReal CudaInt
#define THCS_NTYPE_IS_INT
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THCS_NTYPE_IS_INT

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
