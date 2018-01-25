#ifndef THS_GENERIC_FILE
#error "You must define THS_GENERIC_FILE before including THSGenerateIntTypes.h"
#endif

#define ntype uint8_t
#define accntype int64_t
#define Ntype Byte
#define THSInf UINT8_MAX
#define THS_NTYPE_IS_BYTE
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THSInf
#undef THS_NTYPE_IS_BYTE

#define ntype int8_t
#define accntype int64_t
#define Ntype Char
#define THSInf INT8_MAX
#define THS_NTYPE_IS_CHAR
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THSInf
#undef THS_NTYPE_IS_CHAR

#define ntype int16_t
#define accntype int64_t
#define Ntype Short
#define THSInf INT16_MAX
#define THS_NTYPE_IS_SHORT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THSInf
#undef THS_NTYPE_IS_SHORT

#define ntype int32_t
#define accntype int64_t
#define Ntype Int
#define THSInf INT32_MAX
#define THS_NTYPE_IS_INT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THSInf
#undef THS_NTYPE_IS_INT

#define ntype int64_t
#define accntype int64_t
#define Ntype Long
#define THSInf INT64_MAX
#define THS_NTYPE_IS_LONG
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef ntype
#undef accntype
#undef Ntype
#undef THSInf
#undef THS_NTYPE_IS_LONG

#undef THS_GENERIC_FILE
