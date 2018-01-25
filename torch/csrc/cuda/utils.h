#ifndef THCP_UTILS_H
#define THCP_UTILS_H

#define THCPUtils_(NAME) TH_CONCAT_4(THCP,Ntype,Utils_,NAME)

#define THCStoragePtr  TH_CONCAT_3(THC,Ntype,StoragePtr)
#define THCTensorPtr   TH_CONCAT_3(THC,Ntype,TensorPtr)
#define THCPStoragePtr TH_CONCAT_3(THCP,Ntype,StoragePtr)
#define THCPTensorPtr  TH_CONCAT_3(THCP,Ntype,TensorPtr)

#define THCSTensorPtr  TH_CONCAT_3(THCS,Ntype,TensorPtr)
#define THCSPTensorPtr TH_CONCAT_3(THCSP,Ntype,TensorPtr)

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/utils.h"
#include <THC/THCGenerateAllTypes.h>

#endif
