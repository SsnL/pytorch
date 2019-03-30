r""""Contains definitions of the methods used by the _DataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
from torch._six import queue, container_abcs, string_classes
from . import MP_STATUS_CHECK_INTERVAL, ExceptionWrapper


def _pin_memory_loop(in_queue, out_queue, device_id, done_event):
    torch.cuda.set_device(device_id)

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while True:
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        except Exception:
            if done_event.is_set():
                # Weird things can happen when shutting down, e.g., fd being
                # closed when tensors are shared via fds.
                break
            raise
        if r is None:
            assert done_event.is_set()
            return
        elif done_event.is_set():
            # Haven't seen the final signal yet. Keep getting until None.
            continue
        elif isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
        else:
            idx, data = r
            try:
                data = pin_memory_data(data)
            except Exception:
                out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                out_queue.put((idx, data))


def pin_memory_data(data):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {k: pin_memory_data(sample) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory_data(sample) for sample in data))
    elif isinstance(data, container_abcs.Sequence):
        return [pin_memory_data(sample) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data
