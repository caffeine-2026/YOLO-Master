"""Windows-compatible entry point for ``torch.distributed.run``."""

import os
from functools import partial


def disable_static_tcpstore_libuv(rendezvous_module) -> None:
    """Force the legacy TCPStore backend when the Windows torch wheel omits libuv."""
    rendezvous_module.TCPStore = partial(rendezvous_module.TCPStore, use_libuv=False)


def configure_windows_torchrun_environment() -> None:
    """Disable libuv before importing torchrun so its agent and spawned workers inherit the compatible backend."""
    os.environ["USE_LIBUV"] = "0"


def main() -> None:
    """Patch the upstream static rendezvous backend, then delegate to torchrun."""
    configure_windows_torchrun_environment()

    from torch.distributed.elastic.rendezvous import static_tcp_rendezvous
    from torch.distributed.run import main as torchrun_main

    disable_static_tcpstore_libuv(static_tcp_rendezvous)
    torchrun_main()


if __name__ == "__main__":
    main()
