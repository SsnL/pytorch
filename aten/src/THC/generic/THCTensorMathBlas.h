#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathBlas.h"
#else

THC_API accntype THCTensor_(dot)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(addmv)(THCState *state, THCTensor *self, ntype beta, THCTensor *t, ntype alpha, THCTensor *mat, THCTensor *vec);
THC_API void THCTensor_(addmm)(THCState *state, THCTensor *self, ntype beta, THCTensor *t, ntype alpha, THCTensor *mat1, THCTensor *mat2);
THC_API void THCTensor_(addr)(THCState *state, THCTensor *self, ntype beta, THCTensor *t, ntype alpha, THCTensor *vec1, THCTensor *vec2);
THC_API void THCTensor_(addbmm)(THCState *state, THCTensor *result, ntype beta, THCTensor *t, ntype alpha, THCTensor *batch1, THCTensor *batch2);
THC_API void THCTensor_(baddbmm)(THCState *state, THCTensor *result, ntype beta, THCTensor *t, ntype alpha, THCTensor *batch1, THCTensor *batch2);

THC_API void THCTensor_(btrifact)(THCState *state, THCTensor *ra_, THCudaIntTensor *rpivots_, THCudaIntTensor *rinfo_, int pivot, THCTensor *a);
THC_API void THCTensor_(btrisolve)(THCState *state, THCTensor *rb_, THCTensor *b, THCTensor *atf, THCudaIntTensor *pivots);


#endif
