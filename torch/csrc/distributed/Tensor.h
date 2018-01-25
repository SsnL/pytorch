#ifndef THDP_TENSOR_INC
#define THDP_TENSOR_INC

#define THDPTensor TH_CONCAT_3(THDP,Ntype,Tensor)
#define THDPTensorStr TH_CONCAT_STRING_3(torch.cuda.,Ntype,Tensor)
#define THDPTensorClass TH_CONCAT_3(THDP,Ntype,TensorClass)
#define THDPTensor_(NAME) TH_CONCAT_4(THDP,Ntype,Tensor_,NAME)

#define THDPDoubleTensor_Check(obj)  PyObject_IsInstance(obj, THDPDoubleTensorClass)
#define THDPFloatTensor_Check(obj)   PyObject_IsInstance(obj, THDPFloatTensorClass)
#define THDPHalfTensor_Check(obj)    PyObject_IsInstance(obj, THDPHalfTensorClass)
#define THDPLongTensor_Check(obj)    PyObject_IsInstance(obj, THDPLongTensorClass)
#define THDPIntTensor_Check(obj)     PyObject_IsInstance(obj, THDPIntTensorClass)
#define THDPShortTensor_Check(obj)   PyObject_IsInstance(obj, THDPShortTensorClass)
#define THDPCharTensor_Check(obj)    PyObject_IsInstance(obj, THDPCharTensorClass)
#define THDPByteTensor_Check(obj)    PyObject_IsInstance(obj, THDPByteTensorClass)

#define THDPDoubleTensor_CData(obj)  (obj)->cdata
#define THDPFloatTensor_CData(obj)   (obj)->cdata
#define THDPLongTensor_CData(obj)    (obj)->cdata
#define THDPIntTensor_CData(obj)     (obj)->cdata
#define THDPShortTensor_CData(obj)   (obj)->cdata
#define THDPCharTensor_CData(obj)    (obj)->cdata
#define THDPByteTensor_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THDPTensorType TH_CONCAT_3(THDP,Ntype,TensorType)
#define THDPTensorBaseStr TH_CONCAT_STRING_3(Distributed,Ntype,TensorBase)
#define THDPTensor_stateless_(NAME) TH_CONCAT_4(THDP,Ntype,Tensor_stateless_,NAME)
#define THDPTensorStatelessType TH_CONCAT_3(THDP,Ntype,TensorStatelessType)
#define THDPTensorStateless TH_CONCAT_3(THDP,Ntype,TensorStateless)
#define THDPTensorStatelessMethods TH_CONCAT_3(THDP,Ntype,TensorStatelessMethods)
#endif

#include "override_macros.h"

#define THD_GENERIC_FILE "torch/csrc/generic/Tensor.h"
#include <THD/base/THDGenerateAllTypes.h>

#endif
