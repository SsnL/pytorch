#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.c"
#else

#ifdef USE_CBLAS
  #ifdef USE_MKL_BLAS
    #include<mkl_cblas.h>
    #define BLAS_64 1
    #define blas_int int64_t
  #else // USE_MKL
    #include<cblas.h>
    typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE;
  #endif // USE_MKL
#endif // USE_CBLAS

#ifndef BLAS_64
  #define BLAS_64 0
#endif
#ifndef blas_int
  #define blas_int int
#endif

#ifdef TH_NTYPE_IS_DOUBLE
  #define CBLAS_(NAME) cblas_d##NAME
  #define BLAS_(NAME) d##NAME##_
#else
  #define CBLAS_(NAME) cblas_s##NAME
  #define BLAS_(NAME) s##NAME##_
#endif

#if defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT)
#ifndef USE_CBLAS
TH_EXTERNC void BLAS_(swap)(const blas_int *n, ntype *x, const blas_int *incx, ntype *y, const blas_int *incy);
TH_EXTERNC void BLAS_(scal)(const blas_int *n, const ntype *a, ntype *x, const blas_int *incx);
TH_EXTERNC void BLAS_(copy)(const blas_int *n, const ntype *x, const blas_int *incx, ntype *y, const blas_int *incy);
TH_EXTERNC void BLAS_(axpy)(const blas_int *n, const ntype *a, const ntype *x, const blas_int *incx, ntype *y, const blas_int *incy);

#if defined(BLAS_F2C) || defined(TH_NTYPE_IS_DOUBLE)
TH_EXTERNC double BLAS_(dot)(const blas_int *n, const ntype *x, const blas_int *incx, const ntype *y, const blas_int *incy);
#else
TH_EXTERNC float BLAS_(dot)(const blas_int *n, const ntype *x, const blas_int *incx, const ntype *y, const blas_int *incy);
#endif

TH_EXTERNC void BLAS_(gemv)(const char *trans, const blas_int *m, const blas_int *n, const ntype *alpha, const ntype *a, const blas_int *lda,
                           const ntype *x, const blas_int *incx, const ntype *beta, ntype *y, const blas_int *incy);
TH_EXTERNC void BLAS_(ger)(const blas_int *m, const blas_int *n, const ntype *alpha, const ntype *x, const blas_int *incx, const ntype *y,
                          const blas_int *incy, ntype *a, const blas_int *lda);

TH_EXTERNC void BLAS_(gemm)(const char *transa, const char *transb, const blas_int *m, const blas_int *n, const blas_int *k, const ntype *alpha,
                                   const ntype *a, const blas_int *lda, const ntype *b, const blas_int *ldb, const ntype *beta, ntype *c, const blas_int *ldc);
#else // USE_CBLAS

#ifndef _cblas_transpose_trasform
static inline CBLAS_TRANSPOSE toCBlasTranspose(const char fortran_transpose) {
  if (fortran_transpose  == 'n' || fortran_transpose == 'N') {
    return CblasNoTrans;
  } else if (fortran_transpose == 'T' || fortran_transpose == 't') {
    return CblasTrans;
  } else {
    return CblasConjTrans;
  }
}
#define _cblas_transpose_trasform
#endif

// TH_EXTERNC void CBLAS_(swap)(const blas_int n, const ntype *x, const blas_int incx, const ntype *y, const blas_int incy);
// TH_EXTERNC void CBLAS_(scal)(const blas_int n, const ntype *a, const ntype *x, const blas_int incx);
// TH_EXTERNC void CBLAS_(copy)(const blas_int n, const ntype *x, const blas_int incx, const ntype *y, const blas_int incy);
// TH_EXTERNC void CBLAS_(axpy)(const blas_int n, const ntype *a, const ntype *x, const blas_int *incx, const ntype *y, const blas_int incy);
// TH_EXTERNC ntype CBLAS_(dot)(const blas_int n, const ntype *x, const blas_int incx, const ntype *y, const blas_int incy);
// TH_EXTERNC void CBLAS_(gemv)(const CBLAS_ORDER order,
//                              const CBLAS_TRANSPOSE trans, const blas_int m, const blas_int n,
//                              const ntype alpha, const ntype *a, const blas_int lda,
//                              const ntype *x, const blas_int incx, const ntype beta,
//                              const ntype *y, const blas_int incy);
// TH_EXTERNC void CBLAS_(ger)(const CBLAS_ORDER order, const blas_int m, const blas_int n,
//                             const ntype alpha, const ntype *x, const blas_int incx,
//                             const ntype *y, const blas_int incy, const ntype *a, const blas_int lda);
// TH_EXTERNC void CBLAS_(gemm)(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSEtransb,
//                              const blas_int *m, const blas_int n, const blas_int k, const ntype alpha,
//                              const ntype *a, const blas_int lda, const ntype *b, const blas_int ldb, const ntype beta,
//                              const ntype *c, const blas_int ldc);

static inline void BLAS_(swap)(const blas_int *n, ntype *x, const blas_int *incx, ntype *y, const blas_int *incy) {
  return CBLAS_(swap)(*n, x, *incx, y, *incy);
}

static inline void BLAS_(scal)(const blas_int *n, const ntype *a, ntype *x, const blas_int *incx){
  return CBLAS_(scal)(*n, *a, x, *incx);
}

static inline void BLAS_(copy)(const blas_int *n, const ntype *x, const blas_int *incx, ntype *y, const blas_int *incy) {
  return CBLAS_(copy)(*n, x, *incx, y, *incy);
}

static inline void BLAS_(axpy)(const blas_int *n, const ntype *a, const ntype *x, const blas_int *incx, ntype *y, const blas_int *incy) {
  return CBLAS_(axpy)(*n, *a, x, *incx, y, *incy);
}

static inline ntype BLAS_(dot)(const blas_int *n, const ntype *x, const blas_int *incx, const ntype *y, const blas_int *incy) {
  return CBLAS_(dot)(*n, x, *incx, y, *incy);
}

static inline void BLAS_(gemv)(const char *trans, const blas_int *m, const blas_int *n, const ntype *alpha, const ntype *a, const blas_int *lda,
                           const ntype *x, const blas_int *incx, const ntype *beta, ntype *y, const blas_int *incy) {
  return CBLAS_(gemv)(CblasColMajor, toCBlasTranspose(*trans), *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

static inline void BLAS_(ger)(const blas_int *m, const blas_int *n, const ntype *alpha, const ntype *x, const blas_int *incx, const ntype *y,
                          const blas_int *incy, ntype *a, const blas_int *lda) {
  return CBLAS_(ger)(CblasColMajor, *m, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

static inline void BLAS_(gemm)(const char *transa, const char *transb, const blas_int *m, const blas_int *n, const blas_int *k, const ntype *alpha,
                                   const ntype *a, const blas_int *lda, const ntype *b, const blas_int *ldb, const ntype *beta, ntype *c, const blas_int *ldc) {
  return CBLAS_(gemm)(CblasColMajor, toCBlasTranspose(*transa), toCBlasTranspose(*transb), *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}
#endif // USE_CBLAS
#endif // defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT)

void THBlas_(swap)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT))
  if ( BLAS_64 || ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) )
  {
    blas_int i_n = (blas_int) n;
    blas_int i_incx = (blas_int) incx;
    blas_int i_incy = (blas_int) incy;
    BLAS_(swap)(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
    {
      ntype z = x[i*incx];
      x[i*incx] = y[i*incy];
      y[i*incy] = z;
    }
  }
}

void THBlas_(scal)(int64_t n, ntype a, ntype *x, int64_t incx)
{
  if(n == 1)
    incx = 1;

#if defined(USE_BLAS) && (defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT))
  if ( BLAS_64 || ((n <= INT_MAX) && (incx <= INT_MAX)) )
  {
    blas_int i_n = (blas_int) n;
    blas_int i_incx = (blas_int) incx;

    BLAS_(scal)(&i_n, &a, x, &i_incx);
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++) {
      if (a == 0) {
        x[i*incx] = 0;
      } else {
        x[i*incx] *= a;
      }
    }
  }
}

void THBlas_(copy)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT))
  if ( BLAS_64 || ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) )
  {
    blas_int i_n = (blas_int) n;
    blas_int i_incx = (blas_int) incx;
    blas_int i_incy = (blas_int) incy;

    BLAS_(copy)(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
      y[i*incy] = x[i*incx];
  }
}

void THBlas_(axpy)(int64_t n, ntype a, ntype *x, int64_t incx, ntype *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT))
  if ( BLAS_64 || ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) )
  {
    blas_int i_n = (blas_int) n;
    blas_int i_incx = (blas_int) incx;
    blas_int i_incy = (blas_int) incy;

    BLAS_(axpy)(&i_n, &a, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
      y[i*incy] += a*x[i*incx];
  }
}

ntype THBlas_(dot)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT))
  if (BLAS_64 || ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) )
  {
    blas_int i_n = (blas_int) n;
    blas_int i_incx = (blas_int) incx;
    blas_int i_incy = (blas_int) incy;

    return (ntype) BLAS_(dot)(&i_n, x, &i_incx, y, &i_incy);
  }
#endif
  {
    int64_t i;
    ntype sum = 0;
    for(i = 0; i < n; i++)
    sum += x[i*incx]*y[i*incy];
    return sum;
  }
}

void THBlas_(gemv)(char trans, int64_t m, int64_t n, ntype alpha, ntype *a, int64_t lda,
                   ntype *x, int64_t incx, ntype beta, ntype *y, int64_t incy)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT))
  if (incx > 0 && incy > 0) {
    if (BLAS_64 ||
        ((m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)))
    {
      THArgCheck(lda >= THMax(1, m), 6,
        "lda should be at least max(1, m=%d), but have %d", m, lda);
      blas_int i_m = (blas_int) m;
      blas_int i_n = (blas_int) n;
      blas_int i_lda = (blas_int) lda;
      blas_int i_incx = (blas_int) incx;
      blas_int i_incy = (blas_int) incy;

      BLAS_(gemv)(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
      return;
    }
  }
#endif
  {
    int64_t i, j;

    if( (trans == 'T') || (trans == 't') )
    {
      for(i = 0; i < n; i++)
      {
        ntype sum = 0;
        ntype *row_ = a+lda*i;
        for(j = 0; j < m; j++)
          sum += x[j*incx]*row_[j];
	if (beta == 0)
	  y[i*incy] = alpha*sum;
	else
	  y[i*incy] = beta*y[i*incy] + alpha*sum;
      }
    }
    else
    {
      if(beta != 1)
        THBlas_(scal)(m, beta, y, incy);

      for(j = 0; j < n; j++)
      {
        ntype *column_ = a+lda*j;
        ntype z = alpha*x[j*incx];
        for(i = 0; i < m; i++)
          y[i*incy] += z*column_[i];
      }
    }
  }
}

void THBlas_(ger)(int64_t m, int64_t n, ntype alpha, ntype *x, int64_t incx, ntype *y, int64_t incy, ntype *a, int64_t lda)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT))
  if (incx > 0 && incy > 0) {
    if (BLAS_64 ||
        ((m <= INT_MAX) && (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)))
    {
      THArgCheck(lda >= THMax(1, m), 9,
        "lda should be at least max(1, m=%d), but have %d", m, lda);
      blas_int i_m = (blas_int) m;
      blas_int i_n = (blas_int) n;
      blas_int i_lda = (blas_int) lda;
      blas_int i_incx = (blas_int) incx;
      blas_int i_incy = (blas_int) incy;

      BLAS_(ger)(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
      return;
    }
  }
#endif
  {
    int64_t i, j;
    for(j = 0; j < n; j++)
    {
      ntype *column_ = a+j*lda;
      ntype z = alpha*y[j*incy];
      for(i = 0; i < m; i++)
        column_[i] += z*x[i*incx] ;
    }
  }
}

void THBlas_(gemm)(char transa, char transb, int64_t m, int64_t n, int64_t k, ntype alpha,
                   ntype *a, int64_t lda, ntype *b, int64_t ldb, ntype beta, ntype *c, int64_t ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

#if defined(USE_BLAS) && (defined(TH_NTYPE_IS_DOUBLE) || defined(TH_NTYPE_IS_FLOAT))
  if (BLAS_64 ||
      ((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) &&
       (lda <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX)))
  {
    THArgCheck(lda >= THMax(1, (transa_ ? k : m)), 8,
      "lda should be at least max(1, %d), but have %d", (transa_ ? k : m), lda);
    THArgCheck(ldb >= THMax(1, (transb_ ? n : k)), 10,
      "ldb should be at least max(1, %d), but have %d", (transb_ ? n : k), ldb);
    THArgCheck(ldc >= THMax(1, m), 13,
      "ldc should be at least max(1, m=%d), but have %d", m, ldc);
    blas_int i_m = (blas_int) m;
    blas_int i_n = (blas_int) n;
    blas_int i_k = (blas_int) k;
    blas_int i_lda = (blas_int) lda;
    blas_int i_ldb = (blas_int) ldb;
    blas_int i_ldc = (blas_int) ldc;

    BLAS_(gemm)(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
    return;
  }
#endif
  {
    int64_t i, j, l;
    if(!transa_ && !transb_)
    {
      ntype *a_ = a;
      for(i = 0; i < m; i++)
      {
        ntype *b_ = b;
        for(j = 0; j < n; j++)
        {
          ntype sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      ntype *a_ = a;
      for(i = 0; i < m; i++)
      {
        ntype *b_ = b;
        for(j = 0; j < n; j++)
        {
          ntype sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
      ntype *a_ = a;
      for(i = 0; i < m; i++)
      {
        ntype *b_ = b;
        for(j = 0; j < n; j++)
        {
          ntype sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else
    {
      ntype *a_ = a;
      for(i = 0; i < m; i++)
      {
        ntype *b_ = b;
        for(j = 0; j < n; j++)
        {
          ntype sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
  }
}

#undef BLAS_64
#undef BLAS_
#undef CBLAS_
#undef blas_int

#endif
