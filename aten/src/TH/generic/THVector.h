#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVector.h"
#else

struct THGenerator;

TH_API void THVector_(fill)(ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THVector_(cadd)(ntype *z, const ntype *x, const ntype *y, const ntype c, const ptrdiff_t n);
TH_API void THVector_(adds)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THVector_(cmul)(ntype *z, const ntype *x, const ntype *y, const ptrdiff_t n);
TH_API void THVector_(muls)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THVector_(cdiv)(ntype *z, const ntype *x, const ntype *y, const ptrdiff_t n);
TH_API void THVector_(divs)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THVector_(copy)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(neg)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(normal_fill)(ntype *data,
                                   const int64_t size,
                                   struct THGenerator *generator,
                                   const ntype mean,
                                   const ntype stddev);

#if defined(TH_NTYPE_IS_SHORT) || defined(TH_NTYPE_IS_INT) || defined(TH_NTYPE_IS_LONG)
TH_API void THVector_(abs)(ntype *y, const ntype *x, const ptrdiff_t n);
#endif

/* floating point only now */
#if defined(TH_NTYPE_IS_FLOAT) || defined(TH_NTYPE_IS_DOUBLE)

TH_API void THVector_(log)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(lgamma)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(digamma)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(trigamma)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(log1p)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(sigmoid)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(exp)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(expm1)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(erf)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(erfinv)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(cos)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(acos)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(cosh)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(sin)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(asin)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(sinh)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(tan)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(atan)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(tanh)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(pow)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THVector_(sqrt)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(rsqrt)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(ceil)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(floor)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(round)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(abs)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(trunc)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(frac)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THVector_(cinv)(ntype *y, const ntype *x, const ptrdiff_t n);

#endif /* floating point only part */

/* Initialize the dispatch pointers */
TH_API void THVector_(vectorDispatchInit)(void);

#endif
