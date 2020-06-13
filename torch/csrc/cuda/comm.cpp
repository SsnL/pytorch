#include <torch/csrc/cuda/comm.h>

#include <torch/csrc/cuda/device_set.h>
#include <torch/csrc/utils/tensor_flatten.h>

#ifdef USE_NCCL
#include <torch/csrc/cuda/nccl.h>
#endif

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/WrapDimUtils.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/variable.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace torch { namespace cuda {
using namespace at;
using namespace torch::autograd;

// Some operations can be performed more efficiently if we're handling tensors
// of a single type only. Adding this logic directly in the loop makes it a bit
// ugly, so here's a helper for it.
struct unique_type_checker {
  void show(const at::DeprecatedTypeProperties& t) {
    if (!unique) return;
    if (!type) type = &t;
    unique = (type == &t);
  }

  const at::DeprecatedTypeProperties *type = nullptr;
  bool unique = true;
};

// ***************** Broadcast *******************
//
// Broadcast a source tensor (CPU or CUDA) to a list of CUDA devices, or CUDA
// tensors on one or more devices.

// no checks
static inline
std::vector<Tensor>& _broadcast_out_impl(const Tensor& tensor, std::vector<Tensor> &out_tensors) {
#ifdef USE_NCCL
  std::vector<Tensor> nccl_list;
  nccl_list.reserve(out_tensors.size() + 1);
  nccl_list.push_back(tensor);
  for (auto& out_tensor : out_tensors) {
    nccl_list.push_back(out_tensor);
  }
  if (nccl::is_available(nccl_list)) {
    nccl::broadcast(nccl_list);
  } else {
#else
  {
#endif
    for (auto& out_tensor : out_tensors) {
      out_tensor.copy_(tensor, /*non_blocking=*/true);
    }
  }
  return out_tensors;
}

std::vector<Tensor>& broadcast_out(const Tensor& tensor, std::vector<Tensor> &out_tensors) {
  for (size_t i = 0; i < out_tensors.size(); i++) {
    TORCH_CHECK(
      out_tensors[i].is_cuda(),
      "Expected all output tensors to be CUDA tensors, but output tensor at index ",
      i, " has device '", out_tensors[i].device(), "'");
    TORCH_CHECK(
      out_tensors[i].sizes() == tensor.sizes(),
      "Expected all output tensors to have same shape as the source tensor ",
      tensor.sizes(), ", but output tensor at index ", i, " has shape ",
      out_tensors[i].sizes());
  }
  return _broadcast_out_impl(tensor, out_tensors);
}

std::vector<Tensor> broadcast(const Tensor& tensor, const std::vector<at::Device>& devices) {
  std::vector<Tensor> diff_device_dst_tensors;
  diff_device_dst_tensors.reserve(devices.size());
  for (const auto& device : devices) {
    TORCH_CHECK(device.is_cuda(), "Expected CUDA devices, but got '", device, "'");
    TORCH_CHECK(device.has_index(), "Expected devices with specified indices, but got '", device, "'");
    if (device != tensor.device()) {
      diff_device_dst_tensors.push_back(
        at::empty(tensor.sizes(), tensor.options().device(device)));  // preserve memory format
    }
  }
  _broadcast_out_impl(tensor, diff_device_dst_tensors);
  std::vector<Tensor> dst_tensors;
  dst_tensors.reserve(devices.size());
  auto it = diff_device_dst_tensors.begin();
  for (const auto& device : devices) {
    if (device != tensor.device()) {
      dst_tensors.push_back(*it++);
    } else {
      dst_tensors.push_back(tensor);
    }
  }
  TORCH_INTERNAL_ASSERT(it == diff_device_dst_tensors.end());
  return dst_tensors;
}

// NOTE [ Version Counter in comm.*_coalesced ]
//
// broadcast_coalesced
// ~~~~~~~~~~~~~~~~~~~
//
// In broadcast_coalesced, multiple variables may be coalesced into a single
// large one, broadcast to other devices, and the get split according to the
// original shapes.
//
// When splitting, the view operations will make all Variables broadcast
// together to share a single version counter, because they are all views of the
// large Variable. However, that large Variable is immediately discarded and all
// these Varaibles do not share storage at all.
//
// For example, when two buffers are broadcast together in `DataParallel` and
// one of them is modified in-place during `forward` but the other is needed in
// backward, autograd engine will complain.
//
// We thus re-wrap these Variables after broadcasting (i.e., effectively doing
// what is equivalent to .data in Python), and give them individual version
// counters.
//
// NB: Just calling detach() on the variables is not sufficient
//
// NB: For `device[0]` in broadcast_coalesced, the input Variables are always
//     returned as-is, so **do not** re-wrap them.
//
// reduce_add_coalesced
// ~~~~~~~~~~~~~~~~~~~~
//
// Similarly for reduce_add_coalesced, when the output are newly created
// Variables.
tensor_list2d broadcast_coalesced(const TensorList& tensors, const std::vector<at::Device>& devices, size_t buffer_size) {
  if (tensors.empty()) {
    return tensor_list2d(devices.size());
  }
  auto ref_device = tensors[0].device();
  TORCH_CHECK(
    std::all_of(tensors.begin() + 1, tensors.end(), [&](const at::Tensor& t) { return t.device() == ref_device; }),
    "All tensors must be the same device");
  for (const auto& device : devices) {
    TORCH_CHECK(device.is_cuda(), "Expected CUDA devices, but got '", device, "'");
    TORCH_CHECK(device.has_index(), "Expected devices with specified indices, but got '", device, "'");
  }
#ifdef USE_NCCL
  buffer_size = std::min(torch::cuda::nccl::get_max_count(), buffer_size);
#endif

  tensor_list2d outputs(devices.size());
  outputs[0] = tensors.vec();
  for (auto & o : outputs)
    o.reserve(tensors.size());

  unique_type_checker type_checker;
  at::cuda::CUDAGuard device_guard(devices[0]);
  for (auto & chunk : utils::take_tensors(tensors, buffer_size)) {
    auto & type = chunk.type();
    type_checker.show(type);
    std::vector<at::Tensor> results;
    if (type.is_sparse()) {
      auto flat_tuple = utils::flatten_sparse_tensors(chunk.tensors);
      auto broadcast_indices = broadcast(flat_tuple.first, devices);
      auto broadcast_values = broadcast(flat_tuple.second, devices);
      results.reserve(devices.size());
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(devices[i].index());
        auto & device_outputs = outputs[i];
        auto & inds = broadcast_indices[i];
        auto & vals = broadcast_values[i];
        for (auto & t : utils::unflatten_sparse_tensors(inds, vals, chunk.tensors)) {
          // See NOTE [ Version Counter in comm.*_coalesced ]
          Variable var = t;
          device_outputs.push_back(make_variable(var.tensor_data(), false));
        }
      }
    } else {
      auto results = broadcast(utils::flatten_dense_tensors(chunk.tensors), devices);
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(devices[i].index());
        auto & device_outputs = outputs[i];
        for (auto & t : utils::unflatten_dense_tensors(results[i], chunk.tensors)) {
          // See NOTE [ Version Counter in comm.*_coalesced ]
          Variable var = t;
          device_outputs.push_back(make_variable(var.tensor_data(), false));
        }
      }
    }
  }

  // If we only saw a single tensor type, then we can skip expensive reordering
  if (!type_checker.unique) {
    for (auto & o : outputs)
      utils::reorder_tensors_like(o, tensors);
  }
  return outputs;
}

// ***************** Scatter *******************
//
// Scatter a source tensor (CPU or CUDA) to a list of CUDA tensors on one or
// more devices.

std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors,
    int64_t dim,
    const c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>>& streams) {
  TORCH_CHECK(!out_tensors.empty(), "Expected at least one output tensor to scatter to");
  dim = at::maybe_wrap_dim(dim, tensor);
  int64_t total_size = 0;
  std::vector<int64_t> chunk_sizes;
  chunk_sizes.reserve(out_tensors.size());
  for (size_t i = 0; i < out_tensors.size(); i++) {
    TORCH_CHECK(
      out_tensors[i].is_cuda(),
      "Expected all output tensors to be CUDA tensors, but output tensor at index ",
      i, " has device '", out_tensors[i].device(), "'");
    auto out_sizes = out_tensors[i].sizes().vec();
    bool same_ndim = out_sizes.size() == tensor.dim();
    if (same_ndim) {
      total_size += out_sizes[dim];
      chunk_sizes.push_back(out_sizes[dim]);
      out_sizes[dim] = tensor.size(dim);
    }
    TORCH_CHECK(
      same_ndim && out_sizes == tensor.sizes(),
      "Output tensor at index ", i, " has incorrect shape: ",
      out_tensors[i].sizes(), ". Expected same "
      "shape except for scatter dim ", dim, " as the source tensor: ",
      at::IntArrayRef(tensor.sizes()));
  }
  TORCH_CHECK(
    total_size == tensor.size(dim),
    "Total size for output tensors along scatter dim ", dim, " does not match "
    "the source tensor size at dim ", dim, ". Expected ", tensor.size(dim),
    ", but got total size ", total_size);

  auto chunks = tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);
  at::cuda::OptionalCUDAStreamGuard cuda_guard;
  for (size_t i = 0; i < chunks.size(); i++) {
    if (streams && (*streams)[i]) {
      const auto device_index = static_cast<int16_t>(out_tensors[i].get_device());
      TORCH_CHECK(
          (*streams)[i]->device_index() == device_index,
          "Expected the CUDA device associated with the stream at index ",
          i, " (was ", (*streams)[i]->device_index(), ") ",
          "to match the device supplied at that index ",
          "(expected ", device_index, ")");
      cuda_guard.reset_stream(*(*streams)[i]);
    }
    // NB: We don't detect the case where `out_tensor` is already the correct
    //     view of `tensor` since that would be nontrivial and involve checking
    //     ptr, offset, and strides. So `scatter_out(src, src.chunk(...))` does
    //     more copying than `scatter(src)`.
    out_tensors[i].copy_(chunks[i], /*non_blocking=*/true);
  }
  return out_tensors;
}

std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    const std::vector<at::Device>& devices,
    const c10::optional<std::vector<int64_t>>& chunk_sizes,
    int64_t dim,
    const c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>>& streams) {
  TORCH_CHECK(!devices.empty(), "Expected at least one device to scatter to");
  if (chunk_sizes.has_value()) {
    TORCH_CHECK(chunk_sizes->size() == devices.size(),
        "Expected devices and chunk_sizes to be of same length, but got "
        "len(devices) = ", devices.size(), " and len(chunk_sizes) = ", chunk_sizes->size());
  }
  dim = at::maybe_wrap_dim(dim, tensor);
  std::vector<at::Tensor> chunks =
    chunk_sizes ? tensor.split_with_sizes(/*split_sizes=*/*chunk_sizes, /*dim=*/dim)
                : tensor.chunk(/*chunks=*/devices.size(), /*dim=*/dim);
  at::cuda::OptionalCUDAStreamGuard cuda_guard;
  for (size_t i = 0; i < chunks.size(); ++i) {
    TORCH_CHECK(devices[i].is_cuda(), "Expected CUDA devices, but got '", devices[i], "'");
    TORCH_CHECK(devices[i].has_index(), "Expected devices with specified indices, but got '", devices[i], "'");
    const auto device_index = static_cast<int16_t>(devices[i].index());
    if (device_index != tensor.get_device()) {
      if (streams && (*streams)[i]) {
        TORCH_CHECK(
            (*streams)[i]->device_index() == device_index,
            "Expected the CUDA device associated with the stream at index ",
            i, " (was ", (*streams)[i]->device_index(), ") ",
            "to match the device supplied at that index ",
            "(expected ", device_index, ")");
        cuda_guard.reset_stream(*(*streams)[i]);
      }
      chunks[i] = chunks[i].to(
              {DeviceType::CUDA, device_index},
              /*non_blocking=*/true,
              /*copy=*/false,
              /*memory_format=*/at::MemoryFormat::Preserve);
    }
  }
  return chunks;
}


// ***************** Gather *******************
//
// Gather a list of CUDA tensors on one or more devices to a target tensor or
// device, either CPU or CUDA.

// no checks
static inline
at::Tensor& _gather_out_impl(
    at::TensorList tensors,
    at::Tensor &out_tensor,
    int64_t dim) {
  std::vector<int64_t> chunk_sizes;
  chunk_sizes.reserve(tensors.size());
  for (auto& tensor : tensors) {
    chunk_sizes.push_back(tensor.size(dim));
  }
  auto chunks = out_tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);
  for (size_t i = 0; i < tensors.size(); i++) {
    chunks[i].copy_(
      tensors[i], /*non_blocking=*/out_tensor.is_cuda());
  }
  return out_tensor;
}

at::Tensor& gather_out(
    at::TensorList tensors,
    at::Tensor &out_tensor,
    int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  int64_t total_size = 0;
  auto& first = tensors.front();
  const auto first_size = first.sizes();
  dim = at::maybe_wrap_dim(dim, first);
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  for (size_t i = 0; i < tensors.size(); i++) {
    const auto& tensor = tensors[i];
    TORCH_CHECK(
        tensor.is_cuda(), "Expected all input tensors to be CUDA tensors, but "
        "tensor at index ", i, " has device '", tensor.device(), "'");
    TORCH_CHECK(
        tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
        "Expected all input tensors to have the same number of dimensions, but ",
        "tensor at index ", i, "has ", tensor.ndimension(), " dimensions, (expected ",
        expected_size.size(), ")");
    expected_size[dim] = tensor.size(dim);
    for (size_t dimension = 0; dimension < expected_size.size(); ++dimension) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Input tensor at index ", i, " has invalid shape ", tensor.sizes(),
          ", but expected ", at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim);
  }
  expected_size[dim] = total_size;
  TORCH_CHECK(
    out_tensor.sizes() == expected_size,
    "Expected out tensor to have shape ", at::IntArrayRef(expected_size),
    ", but got ", out_tensor.sizes())

  return _gather_out_impl(tensors, out_tensor, dim);
}


at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    const c10::optional<at::Device>& destination) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  int64_t total_size = 0;
  auto& first = tensors.front();
  const auto first_size = first.sizes();
  dim = at::maybe_wrap_dim(dim, first);
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  auto memory_format = first.suggest_memory_format();
  for (size_t i = 0; i < tensors.size(); i++) {
    const auto& tensor = tensors[i];
    TORCH_CHECK(
        tensor.is_cuda(), "Expected all input tensors to be CUDA tensors, but "
        "tensor at index ", i, " has device ", tensor.device());
    TORCH_CHECK(
        tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
        "Expected all input tensors to have the same number of dimensions, but ",
        "tensor at index ", i, "has ", tensor.ndimension(), " dimensions, (expected ",
        expected_size.size(), ")");
    expected_size[dim] = tensor.size(dim);
    for (size_t dimension = 0; dimension < expected_size.size(); ++dimension) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Input tensor at index ", i, " has invalid shape ", tensor.sizes(),
          ", but expected ", at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim);
    if (memory_format != MemoryFormat::Contiguous && tensor.suggest_memory_format() != memory_format) {
      memory_format = MemoryFormat::Contiguous;
    }
  }
  expected_size[dim] = total_size;

  at::Tensor result = at::empty(
    expected_size,
    first.options().device(destination ? *destination : at::Device(DeviceType::CUDA)).memory_format(memory_format));
  return _gather_out_impl(tensors, result, dim);
}


// ***************** Reduce *******************
//
// Reduce a list of CUDA tensors on one or more devices to a target tensor or
// device, either CPU or CUDA.

namespace {

static inline
at::Tensor _reduce_accumulate(
  const at::Tensor& x, const at::Tensor& y, torch::utils::comm::ReduceOp op) {
    using namespace torch::utils::comm;
    switch (op) {
      case ReduceOp::SUM:
        return x + y;
      case ReduceOp::PRODUCT:
        return x * y;
      case ReduceOp::MIN:
        return x.min(y);
      case ReduceOp::MAX:
        return x.max(y);
      case ReduceOp::BAND:
        return at::bitwise_and(x, y);
      case ReduceOp::BOR:
        return at::bitwise_or(x, y);
      case ReduceOp::BXOR:
        return at::bitwise_xor(x, y);
      default:
        TORCH_CHECK(false, "Unsupported ReduceOp ", op);
    }
}

static inline
at::Tensor& _reduce_accumulate_out(
  at::Tensor& out, const at::Tensor& x, const at::Tensor& y, torch::utils::comm::ReduceOp op) {
    using namespace torch::utils::comm;
    switch (op) {
      case ReduceOp::SUM:
        return at::add_out(out, x, y);
      case ReduceOp::PRODUCT:
        return at::mul_out(out, x, y);
      case ReduceOp::MIN:
        return at::min_out(out, x, y);
      case ReduceOp::MAX:
        return at::max_out(out, x, y);
      case ReduceOp::BAND:
        return at::bitwise_and_out(out, x, y);
      case ReduceOp::BOR:
        return at::bitwise_or_out(out, x, y);
      case ReduceOp::BXOR:
        return at::bitwise_xor_out(out, x, y);
      default:
        TORCH_CHECK(false, "Unsupported ReduceOp ", op);
    }
}

static inline
at::Tensor& _reduce_accumulate_inplace(
  at::Tensor& x, const at::Tensor& y, torch::utils::comm::ReduceOp op) {
    using namespace torch::utils::comm;
    switch (op) {
      case ReduceOp::SUM:
        return x.add_(y);
      case ReduceOp::PRODUCT:
        return x.mul_(y);
      case ReduceOp::MIN:
        return at::min_out(x, x, y);
      case ReduceOp::MAX:
        return at::max_out(x, x, y);
      case ReduceOp::BAND:
        return x.bitwise_and_(y);
      case ReduceOp::BOR:
        return x.bitwise_or_(y);
      case ReduceOp::BXOR:
        return x.bitwise_xor_(y);
      default:
        TORCH_CHECK(false, "Unsupported ReduceOp ", op);
    }
}

}

at::Tensor reduce(
    at::TensorList tensors,
    torch::utils::comm::ReduceOp op,
    const c10::optional<at::Device>& destination) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to reduce from");
  at::Device out_device(DeviceType::CUDA);
  if (destination) {
    out_device = *destination;
  }
  if (out_device.is_cuda() && !out_device.has_index()) {
    out_device.set_index(c10::cuda::current_device());
  }
  auto ref_size = tensors[0].sizes();
  auto result_memory_format = tensors[0].suggest_memory_format();
  ptrdiff_t root_index = -1;
  for (size_t i = 0; i < tensors.size(); i++) {
    if (i > 0) {
      TORCH_CHECK(
        tensors[i].sizes() == ref_size,
        "Expected input tensors to have the same size, but got ",
        "inputs[0] of size ", ref_size, " and inputs[", i, "] of size ",
        tensors[i].sizes());
      if (tensors[i].suggest_memory_format() != result_memory_format) {
        result_memory_format = MemoryFormat::Contiguous;
      }
    }
    TORCH_CHECK(
      tensors[i].is_cuda(), "Expected all inputs to have CUDA type, but "
      "got tensor with device ", tensors[i].device());
    if (tensors[i].device() == out_device) {
      root_index = i;
    }
  }

  if (tensors.size() == 1 && root_index == 0) {
    return tensors[0];
  }

  bool non_blocking = out_device.is_cuda();
  if (tensors.size() == 1) {
     // root_index = -1, i.e., need to move device
    return tensors[0].to(out_device, /*non_blocking=*/non_blocking);
  }

#ifdef USE_NCCL
  if (out_device.is_cuda() && root_index != -1) {
    auto tensors_vec = tensors.vec();
    if (nccl::is_available(tensors_vec) && nccl::is_available(op)) {
      // If NCCL accepts, all inputs are contiguous, so we don't care about the
      // memory_format.
      TORCH_INTERNAL_ASSERT(result_memory_format == MemoryFormat::Contiguous);
      at::Tensor result = at::empty_like(tensors[root_index]);
      nccl::reduce(tensors_vec, result, root_index, op);
      return result;
    }
  }
#endif
  at::Tensor result = _reduce_accumulate(
    tensors[0].to(out_device, /*non_blocking=*/non_blocking),
    tensors[1].to(out_device, /*non_blocking=*/non_blocking),
    op);

  for (size_t i = 2; i < tensors.size(); i++) {
    _reduce_accumulate_inplace(
      result, tensors[i].to(out_device, /*non_blocking=*/non_blocking), op);
  }
  return result;
}


at::Tensor& reduce_out(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    torch::utils::comm::ReduceOp op) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to reduce from");
  auto ref_size = tensors[0].sizes();
  ptrdiff_t root_index = -1;
  for (size_t i = 0; i < tensors.size(); i++) {
    if (i > 0) {
      TORCH_CHECK(
        tensors[i].sizes() == ref_size,
        "Expected input tensors to have the same size, but got ",
        "inputs[0] of size ", ref_size, " and inputs[", i, "] of size ",
        tensors[i].sizes());
    }
    TORCH_CHECK(
      tensors[i].is_cuda(), "Expected all inputs to have CUDA type, but "
      "got tensor with device ", tensors[i].device());
    if (tensors[i].get_device() == out_tensor.get_device()) {
      root_index = i;
    }
  }

  bool non_blocking = out_tensor.is_cuda();

  if (tensors.size() == 1) {
    return out_tensor.copy_(tensors[0], /*non_blocking=*/non_blocking);
  }

#ifdef USE_NCCL
  if (root_index != -1) {
    auto tensors_vec = tensors.vec();
    if (nccl::is_available(tensors_vec) && nccl::is_available({out_tensor}) && nccl::is_available(op)) {
      nccl::reduce(tensors_vec, out_tensor, root_index, op);
      return out_tensor;
    }
  }
#endif
  auto out_device = out_tensor.device();
  if (root_index != -1) {
    // We must reduce `tensors[root_index]` first because it may be the same
    // as (or overlap with) `out_tensor`, and reducing other pairs will
    // overwrite it.
    bool first_accumulate = true;
    for (size_t i = 0; i < tensors.size(); i++) {
      if (i == root_index) {
        continue;
      }
      auto other = tensors[i].to(out_device, /*non_blocking=*/true);
      if (first_accumulate) {
        _reduce_accumulate_out(out_tensor, tensors[root_index], other, op);
        first_accumulate = false;
      } else {
        _reduce_accumulate_inplace(out_tensor, other, op);
      }
    }
  } else {
    _reduce_accumulate_out(
      out_tensor,
      tensors[0].to(out_device, /*non_blocking=*/non_blocking),
      tensors[1].to(out_device, /*non_blocking=*/non_blocking),
      op);

    for (size_t i = 2; i < tensors.size(); i++) {
      _reduce_accumulate_inplace(out_tensor, tensors[i].to(out_device, /*non_blocking=*/non_blocking), op);
    }
  }
  return out_tensor;
}


std::vector<at::Tensor> reduce_coalesced(
    const tensor_list2d& tensor_lists,
    torch::utils::comm::ReduceOp op,
    const c10::optional<at::Device>& destination,
    size_t buffer_size) {
  TORCH_CHECK(!tensor_lists.empty(), "Expected at least one tensor to reduce from");
  size_t num_tensors_per_list = tensor_lists[0].size();
  if (!std::all_of(tensor_lists.begin() + 1, tensor_lists.end(), [&](const std::vector<at::Tensor>& l) { return l.size() == num_tensors_per_list; })) {
    std::vector<int64_t> list_sizes;
    list_sizes.reserve(tensor_lists.size());
    std::transform(tensor_lists.begin(), tensor_lists.end(), std::back_inserter(dst), [](const std::vector<at::Tensor>& l) { return l.size() });
    TORCH_CHECK(false, "Expected all sequences of tensors to have the same number of tensors, but got lengths ", list_sizes);
  }
  at::Device out_device(DeviceType::CUDA);
  if (destination) {
    out_device = *destination;
  }
  if (out_device.is_cuda() && !out_device.has_index()) {
    out_device.set_index(c10::cuda::current_device());
  }
  if (tensor_lists.size() == 1) {
    auto convert = [&](const at::Tensor& t) { return t.to(out_device, /*non_blocking=*/out_device.is_cuda()) };
    return fmap(tensor_lists[0], convert);
  }

  std::vector<tensor_lists::const_iterator> list_iters;
  list_iters.reserve(tensor_lists.size());
  for (const auto& tensors : tensor_lists) {
    list_iters.push_back(tensors.front());
  }

  std::vector<at::Tensor> output;
  output.reserve(num_tensors_per_list);

  tensor_list2d dense_tensor_lists;
  dense_tensor_lists.reserve(tensor_lists.size());
  for (size_t li = 0; li < tensor_lists.size(); li++) {
    dense_tensor_lists.emplace_back();
    dense_tensor_lists.end().reserve(num_tensors_per_list);  // overestimation
  }

  // process sparse ones individually since they may have different sizes on different gpus
  for (size_t ti = 0; ti < num_tensors_per_list; ti++) {
    if (std::all_of(list_iters.begin(), list_iters.end(), [](const tensor_lists::const_iterator& it) { return it->is_sparse(); })) {
      output.push_back(reduce(tensors, op, destination));
    } else {
      for (size_t li = 0; li < tensor_lists.size(); li++) {
        dense_tensor_lists[li].push_back(*(list_iters[li]));
      }
    }
  }
  for (auto& it : list_iters) {
    it++;
  }
}


auto chunks_lists = fmap(dense_tensor_lists, [&](const std::vector<at::Tensor>& tensors) { return utils::take_tensors(tensors, buffer_size); });




// def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
//     """Sums tensors from multiple GPUs.

//     Small tensors are first coalesced into a buffer to reduce the number
//     of synchronizations.

//     Arguments:
//         inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
//             contain tensors from a single device.
//         destination (int, optional): a device on which the output will be
//             placed (default: current device).
//         buffer_size (int): maximum size of the buffer used for coalescing

//     Returns:
//         A tuple of tensors containing an elementwise sum of each group of
//         inputs, placed on the ``destination`` device.
//     """
//     # TODO: When `len(inputs) == 1` and all inputs are on `destination`, just
//     #       return `inputs`.
//     dense_tensors = [[] for _ in inputs]  # shape (num_gpus, num_tensors)
//     output = []
//     ref_order = []
//     # process sparse ones first since they may have different sizes on different gpus
//     for tensor_at_gpus in zip(*inputs):
//         if all(t.is_sparse for t in tensor_at_gpus):
//             result = reduce_add(tensor_at_gpus, destination)  # this will be sparse too
//             output.append(result)
//             ref_order.append(tensor_at_gpus[0])
//         else:
//             for coll, t in zip(dense_tensors, tensor_at_gpus):
//                 coll.append(t.to_dense() if t.is_sparse else t)
//             ref_order.append(dense_tensors[0][-1])
//     itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
//     # now the dense ones, which have consistent sizes
//     for chunks in zip(*itrs):
//         flat_tensors = [_flatten_dense_tensors(chunk) for chunk in chunks]  # (num_gpus,)
//         flat_result = reduce_add(flat_tensors, destination)
//         for t in _unflatten_dense_tensors(flat_result, chunks[0]):
//             # The unflattened tensors do not share storage, and we don't expose
//             # base flat tensor anyways, so give them different version counters.
//             # See NOTE [ Version Counter in comm.*_coalesced ]
//             output.append(t.data)
//     return tuple(_reorder_tensors_as(output, ref_order))

}} // namespace torch::cuda
