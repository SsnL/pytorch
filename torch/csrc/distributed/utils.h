#ifndef THDP_UTILS_H
#define THDP_UTILS_H

#include "THDP.h"

#define THDPUtils_(NAME) TH_CONCAT_4(THDP,Ntype,Utils_,NAME)

#define THDStoragePtr TH_CONCAT_3(THD,Ntype,StoragePtr)
#define THDTensorPtr  TH_CONCAT_3(THD,Ntype,TensorPtr)
#define THDPStoragePtr TH_CONCAT_3(THDP,Ntype,StoragePtr)
#define THDPTensorPtr  TH_CONCAT_3(THDP,Ntype,TensorPtr)

#include "override_macros.h"

#define THD_GENERIC_FILE "torch/csrc/generic/utils.h"
#include <THD/base/THDGenerateAllTypes.h>

typedef THPPointer<THDTensorDescriptor> THDPTensorDesc;

#endif
