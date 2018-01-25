#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateHalfType.h"
#endif

#include "THHalf.h"
#define ntype THHalf
#define accntype float
#define TH_CONVERT_NTYPE_TO_ACCNTYPE(_val) TH_half2float(_val)
#define TH_CONVERT_ACCNTYPE_TO_NTYPE(_val) TH_float2half(_val)
#define Ntype Half
#define THInf TH_HALF_BITS_TO_LITERAL(TH_HALF_INF)
#define TH_NTYPE_IS_HALF
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THInf
#undef TH_NTYPE_IS_HALF
#undef TH_CONVERT_NTYPE_TO_ACCNTYPE
#undef TH_CONVERT_ACCNTYPE_TO_NTYPE

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
