import os
import runpy
import sys
from types import SimpleNamespace

from ultralytics.utils.torchrun import (
    configure_windows_torchrun_environment,
    disable_static_tcpstore_libuv,
    wrap_training_script_for_windows,
)
from ultralytics.utils.torchrun_worker import disable_worker_tcpstore_libuv, main as worker_main


def test_disable_static_tcpstore_libuv_binds_legacy_backend():
    def tcp_store(*args, **kwargs):
        return args, kwargs

    rendezvous = SimpleNamespace(TCPStore=tcp_store)
    disable_static_tcpstore_libuv(rendezvous)

    _, kwargs = rendezvous.TCPStore("127.0.0.1", 12345)
    assert kwargs["use_libuv"] is False


def test_configure_windows_torchrun_environment_overrides_incompatible_libuv(monkeypatch):
    monkeypatch.setenv("USE_LIBUV", "1")

    configure_windows_torchrun_environment()

    assert os.environ["USE_LIBUV"] == "0"


def test_wrap_training_script_for_windows_preserves_target_and_arguments():
    args = SimpleNamespace(training_script="train.py", training_script_args=["--epochs", "1"])

    wrap_training_script_for_windows(args)

    assert args.training_script.endswith("torchrun_worker.py")
    assert args.training_script_args == ["train.py", "--epochs", "1"]


def test_disable_worker_tcpstore_libuv_binds_legacy_backend(monkeypatch):
    rendezvous = SimpleNamespace(TCPStore=lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr("importlib.import_module", lambda name: rendezvous)

    disable_worker_tcpstore_libuv()

    _, kwargs = rendezvous.TCPStore("127.0.0.1", 12345)
    assert kwargs["use_libuv"] is False


def test_worker_main_patches_before_executing_target(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ultralytics.utils.torchrun_worker.disable_worker_tcpstore_libuv", lambda: calls.append("patch")
    )
    monkeypatch.setattr(runpy, "run_path", lambda path, run_name: calls.append((path, run_name, sys.argv.copy())))
    monkeypatch.setattr(sys, "argv", ["torchrun_worker.py", "train.py", "--epochs", "1"])

    worker_main()

    assert calls == ["patch", ("train.py", "__main__", ["train.py", "--epochs", "1"])]
