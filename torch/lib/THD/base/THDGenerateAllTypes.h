#ifndef THD_GENERIC_FILE
#error "You must define THD_GENERIC_FILE before including THDGenerateAllTypes.h"
#endif

#define ntype uint8_t
#define accntype int64_t
#define Ntype Byte
#define THDInf UCHAR_MAX
#define THD_NTYPE_IS_BYTE
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THDInf
#undef THD_NTYPE_IS_BYTE

#define ntype int8_t
#define accntype int64_t
#define Ntype Char
#define THDInf SCHAR_MAX
#define THD_NTYPE_IS_CHAR
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THDInf
#undef THD_NTYPE_IS_CHAR

#define ntype int16_t
#define accntype int64_t
#define Ntype Short
#define THDInf SHRT_MAX
#define THD_NTYPE_IS_SHORT
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THDInf
#undef THD_NTYPE_IS_SHORT

#define ntype int32_t
#define accntype int64_t
#define Ntype Int
#define THDInf INT_MAX
#define THD_NTYPE_IS_INT
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THDInf
#undef THD_NTYPE_IS_INT

#define ntype int64_t
#define accntype int64_t
#define Ntype Long
#define THDInf LONG_MAX
#define THD_NTYPE_IS_LONG
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THDInf
#undef THD_NTYPE_IS_LONG

#define ntype float
#define accntype double
#define Ntype Float
#define THDInf FLT_MAX
#define THD_NTYPE_IS_FLOAT
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THDInf
#undef THD_NTYPE_IS_FLOAT

#define ntype double
#define accntype double
#define Ntype Double
#define THDInf DBL_MAX
#define THD_NTYPE_IS_DOUBLE
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THDInf
#undef THD_NTYPE_IS_DOUBLE

#undef THD_GENERIC_FILE
