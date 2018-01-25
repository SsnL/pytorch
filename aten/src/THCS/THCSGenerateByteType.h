#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define ntype uint8_t
#define accntype int64_t
#define Ntype Byte
#define CReal CudaByte
#define THCS_NTYPE_IS_BYTE
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THCS_NTYPE_IS_BYTE

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
