"""Windows-compatible entry point for ``torch.distributed.run``."""

import os
from functools import partial
from pathlib import Path


def disable_static_tcpstore_libuv(rendezvous_module) -> None:
    """Force the legacy TCPStore backend when the Windows torch wheel omits libuv."""
    rendezvous_module.TCPStore = partial(rendezvous_module.TCPStore, use_libuv=False)


def configure_windows_torchrun_environment() -> None:
    """Disable libuv before importing torchrun so its agent and spawned workers inherit the compatible backend."""
    os.environ["USE_LIBUV"] = "0"


def wrap_training_script_for_windows(args) -> None:
    """Run each Windows worker through a bootstrap that disables libuv before process-group initialization."""
    worker = Path(__file__).with_name("torchrun_worker.py")
    args.training_script_args = [args.training_script, *args.training_script_args]
    args.training_script = str(worker)


def main() -> None:
    """Patch the launcher and worker rendezvous backends, then delegate to torchrun."""
    configure_windows_torchrun_environment()

    from torch.distributed.elastic.rendezvous import static_tcp_rendezvous
    from torch.distributed.run import parse_args, run

    disable_static_tcpstore_libuv(static_tcp_rendezvous)
    args = parse_args(None)
    wrap_training_script_for_windows(args)
    run(args)


if __name__ == "__main__":
    main()
