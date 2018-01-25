#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define real float
#define accreal double
#define TH_CONVERT_NTYPE_TO_ACCNTYPE(_val) (accreal)(_val)
#define TH_CONVERT_ACCNTYPE_TO_NTYPE(_val) (real)(_val)
#define Real Float
#define THInf FLT_MAX
#define TH_NTYPE_IS_FLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_NTYPE_IS_FLOAT
#undef TH_CONVERT_NTYPE_TO_ACCNTYPE
#undef TH_CONVERT_ACCNTYPE_TO_NTYPE

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
