#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateCharType.h"
#endif

#define ntype int8_t
#define accntype int64_t
#define Ntype Char
#define CReal CudaChar
#define THCS_NTYPE_IS_CHAR
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THCS_NTYPE_IS_CHAR

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
