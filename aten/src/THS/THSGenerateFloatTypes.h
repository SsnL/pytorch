#ifndef THS_GENERIC_FILE
#error "You must define THS_GENERIC_FILE before including THSGenerateAllTypes.h"
#endif

#define ntype float
#define accntype double
#define Ntype Float
#define THSInf FLT_MAX
#define THS_NTYPE_IS_FLOAT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef accntype
#undef ntype
#undef Ntype
#undef THSInf
#undef THS_NTYPE_IS_FLOAT

#define ntype double
#define accntype double
#define Ntype Double
#define THSInf DBL_MAX
#define THS_NTYPE_IS_DOUBLE
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef accntype
#undef ntype
#undef Ntype
#undef THSInf
#undef THS_NTYPE_IS_DOUBLE

#undef THS_GENERIC_FILE
