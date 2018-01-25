
#define THTensor                    TH_CONCAT_3(TH,Ntype,Tensor)
#define THTensor_(NAME)             TH_CONCAT_4(TH,Ntype,Tensor_,NAME)

#define THPTensor                   TH_CONCAT_3(THP,Ntype,Tensor)
#define THPTensorStr                TH_CONCAT_STRING_3(torch.,Ntype,Tensor)
#define THPTensorClass              TH_CONCAT_3(THP,Ntype,TensorClass)
#define THPTensor_(NAME)            TH_CONCAT_4(THP,Ntype,Tensor_,NAME)

#define THPStorage TH_CONCAT_3(THP,Ntype,Storage)
#define THPStorageStr TH_CONCAT_STRING_3(torch.,Ntype,Storage)
#define THPStorageClass TH_CONCAT_3(THP,Ntype,StorageClass)
#define THPStorage_(NAME) TH_CONCAT_4(THP,Ntype,Storage_,NAME)

#ifdef _THP_CORE
#define THStoragePtr TH_CONCAT_3(TH,Ntype,StoragePtr)
#define THTensorPtr  TH_CONCAT_3(TH,Ntype,TensorPtr)
#define THPStoragePtr TH_CONCAT_3(THP,Ntype,StoragePtr)
#define THPTensorPtr  TH_CONCAT_3(THP,Ntype,TensorPtr)
#endif
