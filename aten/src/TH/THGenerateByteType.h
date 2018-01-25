#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define real uint8_t
#define ureal uint8_t
#define accreal int64_t
#define Real Byte
#define TH_CONVERT_NTYPE_TO_ACCNTYPE(_val) (accreal)(_val)
#define TH_CONVERT_ACCNTYPE_TO_NTYPE(_val) (real)(_val)
#define THInf UCHAR_MAX
#define TH_NTYPE_IS_BYTE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef ureal
#undef accreal
#undef Real
#undef THInf
#undef TH_NTYPE_IS_BYTE
#undef TH_CONVERT_NTYPE_TO_ACCNTYPE
#undef TH_CONVERT_ACCNTYPE_TO_NTYPE

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
