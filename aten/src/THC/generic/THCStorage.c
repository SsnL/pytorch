#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.c"
#else

ntype* THCStorage_(data)(THCState *state, const THCStorage *self)
{
  return self->data;
}

ptrdiff_t THCStorage_(size)(THCState *state, const THCStorage *self)
{
  return self->size;
}

int THCStorage_(elementSize)(THCState *state)
{
  return sizeof(ntype);
}

void THCStorage_(set)(THCState *state, THCStorage *self, ptrdiff_t index, ntype value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(self->data + index, &value, sizeof(ntype),
                              cudaMemcpyHostToDevice,
                              stream));
  THCudaCheck(cudaStreamSynchronize(stream));
}

ntype THCStorage_(get)(THCState *state, const THCStorage *self, ptrdiff_t index)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  ntype value;
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(&value, self->data + index, sizeof(ntype),
                              cudaMemcpyDeviceToHost, stream));
  THCudaCheck(cudaStreamSynchronize(stream));
  return value;
}

THCStorage* THCStorage_(new)(THCState *state)
{
  return THCStorage_(newWithSize)(state, 0);
}

THCStorage* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size)
{
  return THCStorage_(newWithAllocator)(
    state, size,
    state->cudaDeviceAllocator,
    state->cudaDeviceAllocator->state);
}

THCStorage* THCStorage_(newWithAllocator)(THCState *state, ptrdiff_t size,
                                          THCDeviceAllocator* allocator,
                                          void* allocatorContext)
{
  THArgCheck(size >= 0, 2, "invalid size");
  int device;
  THCudaCheck(cudaGetDevice(&device));

  THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));
  memset(storage, 0, sizeof(THCStorage));
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  storage->size = size;
  storage->device = device;

  if(size > 0)
  {
    // update heap *before* attempting malloc, to free space for the malloc
    cudaError_t err =
      (*allocator->malloc)(allocatorContext,
                           (void**)&(storage->data),
                           size * sizeof(ntype),
                           THCState_getCurrentStream(state));
    if(err != cudaSuccess){
      free(storage);
    }
    THCudaCheck(err);
  } else {
    storage->data = NULL;
  }
  return storage;
}

THCStorage* THCStorage_(newWithSize1)(THCState *state, ntype data0)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 1);
  THCStorage_(set)(state, self, 0, data0);
  return self;
}

THCStorage* THCStorage_(newWithSize2)(THCState *state, ntype data0, ntype data1)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 2);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  return self;
}

THCStorage* THCStorage_(newWithSize3)(THCState *state, ntype data0, ntype data1, ntype data2)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 3);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  return self;
}

THCStorage* THCStorage_(newWithSize4)(THCState *state, ntype data0, ntype data1, ntype data2, ntype data3)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 4);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  THCStorage_(set)(state, self, 3, data3);
  return self;
}

THCStorage* THCStorage_(newWithMapping)(THCState *state, const char *fileName, ptrdiff_t size, int isShared)
{
  THError("not available yet for THCStorage");
  return NULL;
}

THCStorage* THCStorage_(newWithData)(THCState *state, ntype *data, ptrdiff_t size)
{
  return THCStorage_(newWithDataAndAllocator)(state, data, size,
                                              state->cudaDeviceAllocator,
                                              state->cudaDeviceAllocator->state);
}

THCStorage* THCStorage_(newWithDataAndAllocator)(
  THCState *state, ntype *data, ptrdiff_t size,
  THCDeviceAllocator *allocator, void *allocatorContext) {
  THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));
  memset(storage, 0, sizeof(THCStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  int device;
  if (data) {
    struct cudaPointerAttributes attr;
    THCudaCheck(cudaPointerGetAttributes(&attr, data));
    device = attr.device;
  } else {
    THCudaCheck(cudaGetDevice(&device));
  }
  storage->device = device;
  return storage;
}

void THCStorage_(setFlag)(THCState *state, THCStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THCStorage_(clearFlag)(THCState *state, THCStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THCStorage_(retain)(THCState *state, THCStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&self->refcount);
}

void THCStorage_(free)(THCState *state, THCStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (THAtomicDecrementRef(&self->refcount))
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THCudaCheck(
        (*self->allocator->free)(self->allocatorContext, self->data));
    }
    if(self->flag & TH_STORAGE_VIEW) {
      THCStorage_(free)(state, self->view);
    }
    THFree(self);
  }
}
#endif
