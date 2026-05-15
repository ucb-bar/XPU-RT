"""Tests for the ``compgen rt build`` / ``rt list-triples`` verbs.

These verbs wrap CMake to materialise ``libcompgen_rt_static.a`` for an
arbitrary (target triple, toolchain) pair. The tests below cover the
plumbing — argument resolution, error paths, in-tree triple discovery —
without running an actual CMake build (which is covered by an opt-in
integration test).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from compgen.cli import main


def test_rt_list_triples_includes_shipped() -> None:
    """``rt list-triples`` enumerates host + every shipped toolchain."""
    runner = CliRunner()
    result = runner.invoke(main, ["rt", "list-triples"])
    assert result.exit_code == 0, result.output
    assert "host" in result.output
    assert "riscv64-zephyr-elf" in result.output


def test_rt_build_unknown_triple_errors_with_listing(tmp_path: Path) -> None:
    """Unknown triple without --toolchain-file lists shipped triples."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["rt", "build", "--triple", "no_such_triple", "--build-dir", str(tmp_path / "b")],
    )
    assert result.exit_code != 0
    # Error mentions the missing triple AND the available shipped ones.
    assert "no_such_triple" in result.output
    assert "riscv64-zephyr-elf" in result.output


def test_rt_build_accepts_explicit_toolchain_file_for_arbitrary_triple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom triple + --toolchain-file gets past the configure-arg
    stage. We stub cmake so the test doesn't actually build anything;
    we only verify the configure command was assembled correctly."""

    # Write a placeholder toolchain file.
    tc = tmp_path / "fake.cmake"
    tc.write_text("# placeholder toolchain\n")

    # Capture the cmake invocation by stubbing subprocess.call inside the
    # cli module. Returns 0 to simulate success; the post-build artifact
    # check is what fails (no .a was produced) — that's expected.
    captured: list[list[str]] = []

    def fake_call(cmd, *args, **kwargs):  # noqa: ARG001
        captured.append(list(cmd))
        # Materialise an empty .a so the artifact discovery succeeds.
        # The configure call is captured[0]; the build call is captured[1].
        if len(captured) >= 2:
            build_dir = Path(captured[0][captured[0].index("-B") + 1])
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "libcompgen_rt_static.a").write_bytes(b"")
        return 0

    monkeypatch.setattr("compgen.cli.subprocess.call", fake_call)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rt",
            "build",
            "--triple",
            "my_custom_triple",
            "--toolchain-file",
            str(tc),
            "--build-dir",
            str(tmp_path / "build"),
            "--without-cuda",
            "--platform",
            "bare",
            "--cmake",
            "/usr/bin/cmake" if Path("/usr/bin/cmake").is_file() else (shutil.which("cmake") or "/usr/bin/cmake"),
        ],
    )
    assert result.exit_code == 0, result.output

    # The configure command must mention the custom toolchain + flags.
    assert any("my_custom_triple" in arg or str(tmp_path) in arg for arg in captured[0]), captured[0]
    assert f"-DCMAKE_TOOLCHAIN_FILE={tc}" in captured[0]
    assert "-DCG_RT_PLATFORM=bare" in captured[0]
    assert "-DCG_RT_WITH_CUDA=OFF" in captured[0]


def test_rt_build_host_defaults_to_posix_with_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--triple host`` with no overrides uses posix + cuda ON."""
    captured: list[list[str]] = []

    def fake_call(cmd, *args, **kwargs):  # noqa: ARG001
        captured.append(list(cmd))
        if len(captured) >= 2:
            build_dir = Path(captured[0][captured[0].index("-B") + 1])
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "libcompgen_rt_static.a").write_bytes(b"")
        return 0

    monkeypatch.setattr("compgen.cli.subprocess.call", fake_call)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rt",
            "build",
            "--triple",
            "host",
            "--build-dir",
            str(tmp_path / "build"),
            "--cmake",
            "/usr/bin/cmake" if Path("/usr/bin/cmake").is_file() else (shutil.which("cmake") or "/usr/bin/cmake"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "-DCG_RT_PLATFORM=posix" in captured[0]
    assert "-DCG_RT_WITH_CUDA=ON" in captured[0]
    # Host triple = no toolchain file flag.
    assert not any(arg.startswith("-DCMAKE_TOOLCHAIN_FILE=") for arg in captured[0])


def test_rt_build_cross_triple_defaults_to_bare_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shipped cross triple defaults to ``bare`` platform + cuda OFF."""
    captured: list[list[str]] = []

    def fake_call(cmd, *args, **kwargs):  # noqa: ARG001
        captured.append(list(cmd))
        if len(captured) >= 2:
            build_dir = Path(captured[0][captured[0].index("-B") + 1])
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "libcompgen_rt_static.a").write_bytes(b"")
        return 0

    monkeypatch.setattr("compgen.cli.subprocess.call", fake_call)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rt",
            "build",
            "--triple",
            "riscv64-zephyr-elf",
            "--build-dir",
            str(tmp_path / "build"),
            "--cmake",
            "/usr/bin/cmake" if Path("/usr/bin/cmake").is_file() else (shutil.which("cmake") or "/usr/bin/cmake"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "-DCG_RT_PLATFORM=bare" in captured[0]
    assert "-DCG_RT_WITH_CUDA=OFF" in captured[0]
    assert any(
        arg.startswith("-DCMAKE_TOOLCHAIN_FILE=") and arg.endswith("riscv64-zephyr-elf.cmake") for arg in captured[0]
    ), captured[0]


def test_rt_build_clean_wipes_existing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--clean`` removes a stale build directory before configuring."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    sentinel = build_dir / "STALE.txt"
    sentinel.write_text("old")
    assert sentinel.exists()

    def fake_call(cmd, *args, **kwargs):  # noqa: ARG001
        # Re-create the build dir + write the artifact so the verb
        # finishes cleanly; sentinel must NOT come back.
        bd = Path(cmd[cmd.index("-B") + 1] if "-B" in cmd else build_dir)
        bd.mkdir(parents=True, exist_ok=True)
        (bd / "libcompgen_rt_static.a").write_bytes(b"")
        return 0

    monkeypatch.setattr("compgen.cli.subprocess.call", fake_call)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rt",
            "build",
            "--triple",
            "host",
            "--build-dir",
            str(build_dir),
            "--clean",
            "--cmake",
            "/usr/bin/cmake" if Path("/usr/bin/cmake").is_file() else (shutil.which("cmake") or "/usr/bin/cmake"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert not sentinel.exists(), "stale file survived --clean"


def test_rt_build_missing_cmake_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No cmake on PATH and no --cmake → typed error, not crash."""
    monkeypatch.setattr("compgen.cli.shutil.which", lambda _name: None)
    monkeypatch.setattr("compgen.cli.Path.is_file", lambda self: False)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["rt", "build", "--triple", "host", "--build-dir", str(tmp_path / "build")],
    )
    assert result.exit_code != 0
    assert "cmake" in result.output.lower()


@pytest.mark.slow
def test_rt_build_host_actually_builds_and_produces_archive(tmp_path: Path) -> None:
    """End-to-end host build. Opt-in (slow) — real CMake invocation."""
    if not Path("/usr/bin/cmake").is_file():
        pytest.skip("system cmake at /usr/bin/cmake not available")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rt",
            "build",
            "--triple",
            "host",
            "--build-dir",
            str(tmp_path / "build"),
            "--without-cuda",
            "--cmake",
            "/usr/bin/cmake",
        ],
    )
    assert result.exit_code == 0, result.output
    artifact = tmp_path / "build" / "libcompgen_rt_static.a"
    assert artifact.is_file()
    assert artifact.stat().st_size > 1024  # not an empty placeholder
