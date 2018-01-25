#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateDoubleType.h"
#endif

#define ntype double
#define accntype double
#define TH_CONVERT_NTYPE_TO_ACCNTYPE(_val) (accntype)(_val)
#define TH_CONVERT_ACCNTYPE_TO_NTYPE(_val) (ntype)(_val)
#define Ntype Double
#define THInf DBL_MAX
#define TH_NTYPE_IS_DOUBLE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accntype
#undef ntype
#undef Ntype
#undef THInf
#undef TH_NTYPE_IS_DOUBLE
#undef TH_CONVERT_NTYPE_TO_ACCNTYPE
#undef TH_CONVERT_ACCNTYPE_TO_NTYPE

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
