#pragma once

#include <TH/TH.h>
#include "../../THD.h"

#define THDStorage         TH_CONCAT_3(THD,Ntype,Storage)
#define THDStorage_(NAME)  TH_CONCAT_4(THD,Ntype,Storage_,NAME)

#include "generic/THDStorage.h"
#include <TH/THGenerateAllTypes.h>
