"""Bootstrap a Windows torchrun worker with the non-libuv TCPStore backend."""

import importlib
import runpy
import sys
from functools import partial


def disable_worker_tcpstore_libuv() -> None:
    """Bind the rendezvous module's TCPStore alias to the backend available in Windows PyTorch wheels."""
    rendezvous = importlib.import_module("torch.distributed.rendezvous")
    rendezvous.TCPStore = partial(rendezvous.TCPStore, use_libuv=False)


def disable_incomplete_triton_detection() -> bool:
    """Prevent PyTorch Dynamo from treating an incomplete Triton namespace package as a usable runtime."""
    try:
        triton = importlib.import_module("triton")
    except ImportError:
        return False

    language = getattr(triton, "language", None)
    if language is not None and hasattr(language, "dtype"):
        return False

    triton_utils = importlib.import_module("torch.utils._triton")
    triton_utils.has_triton_package = lambda: False
    return True


def main() -> None:
    """Patch rendezvous before executing the original training script in this worker process."""
    if len(sys.argv) < 2:
        raise SystemExit("torchrun worker bootstrap requires a training script")

    disable_worker_tcpstore_libuv()
    disable_incomplete_triton_detection()
    script, *script_args = sys.argv[1:]
    sys.argv = [script, *script_args]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
