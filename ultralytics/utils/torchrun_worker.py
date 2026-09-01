"""Bootstrap a Windows torchrun worker with the non-libuv TCPStore backend."""

import importlib
import runpy
import sys
from functools import partial


def disable_worker_tcpstore_libuv() -> None:
    """Bind the rendezvous module's TCPStore alias to the backend available in Windows PyTorch wheels."""
    rendezvous = importlib.import_module("torch.distributed.rendezvous")
    rendezvous.TCPStore = partial(rendezvous.TCPStore, use_libuv=False)


def main() -> None:
    """Patch rendezvous before executing the original training script in this worker process."""
    if len(sys.argv) < 2:
        raise SystemExit("torchrun worker bootstrap requires a training script")

    disable_worker_tcpstore_libuv()
    script, *script_args = sys.argv[1:]
    sys.argv = [script, *script_args]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
