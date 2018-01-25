#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define ntype uint8_t
#define accntype int64_t
#define Ntype Byte
#define CReal CudaByte
#define THC_NTYPE_IS_BYTE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef CReal
#undef THC_NTYPE_IS_BYTE

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
