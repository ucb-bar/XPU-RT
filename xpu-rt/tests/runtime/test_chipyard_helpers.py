"""Tests for the chipyard helpers (REQ-005 + REQ-009).

Covers:
- ``parse_htif_data_stream`` — concatenates LSB=0 payload bytes
  little-endian, stops at first LSB=1 (exit).
- ``htif_data_stream_c`` — emits the C side that paired parser
  consumes (smoke check on the generated source).
- ``shared_dram_section`` — emits a linker fragment placing the
  shared region in system DRAM with the right base.
"""

from __future__ import annotations

from compgen.runtime.baremetal.chipyard import (
    htif_data_stream_c,
    parse_htif_data_stream,
    parse_htif_exit,
    shared_dram_section,
)

# ---------------------------------------------------------------------------
# parse_htif_data_stream — REQ-005
# ---------------------------------------------------------------------------


def test_parse_htif_data_stream_concats_payloads() -> None:
    """Two data words → 8 bytes little-endian; exit terminates."""
    log = (
        "tohost = 0x00000002\n"  # data word 1 (val>>1 = 1)
        "tohost = 0x00000004\n"  # data word 2 (val>>1 = 2)
        "tohost = 0x00000007\n"  # exit code 3 (LSB=1)
    )
    data = parse_htif_data_stream(log)
    # 0x00000001 LE = b'\x01\x00\x00\x00'
    # 0x00000002 LE = b'\x02\x00\x00\x00'
    assert data == bytes.fromhex("0100000002000000")


def test_parse_htif_data_stream_stops_at_exit() -> None:
    """Data words AFTER the first exit are ignored."""
    log = (
        "tohost = 0x00000002\n"  # data
        "tohost = 0x00000003\n"  # exit
        "tohost = 0x00000004\n"  # ignored
    )
    data = parse_htif_data_stream(log)
    assert data == bytes.fromhex("01000000")


def test_parse_htif_data_stream_no_data_no_exit_returns_empty() -> None:
    log = "Cyclotron: created sim object\n[UART] up\n"
    assert parse_htif_data_stream(log) == b""


def test_parse_htif_data_stream_handles_immediate_exit() -> None:
    """Pass/fail-only logs (no data words) return empty bytes."""
    log = "tohost = 0x00000001\n"  # exit code 0
    assert parse_htif_data_stream(log) == b""


def test_parse_htif_data_stream_aligns_with_parse_htif_exit() -> None:
    """The two parsers see the same exit boundary."""
    log = (
        "tohost = 0x00000002\n"
        "tohost = 0x00000004\n"
        "tohost = 0x0000000b\n"  # 0x0b = (5 << 1) | 1 → exit code 5
    )
    data = parse_htif_data_stream(log)
    saw_exit, code = parse_htif_exit(log)
    assert saw_exit is True
    assert code == 5
    # Data parser stopped exactly at the exit; only the two pre-exit
    # data words made it through.
    assert len(data) == 8


def test_parse_htif_data_stream_round_trips_emitted_bytes() -> None:
    """Round-trip: encode floats → simulate HTIF log → decode → equal."""
    import struct

    payload = b"\x01\x02\x03\x04\xde\xad\xbe\xef"
    # Each 4-byte chunk becomes one tohost = (word << 1) line.
    lines = []
    for i in range(0, len(payload), 4):
        word = int.from_bytes(payload[i : i + 4], "little")
        lines.append(f"tohost = 0x{word << 1:08x}\n")
    lines.append("tohost = 0x00000001\n")  # exit code 0
    log = "".join(lines)

    decoded = parse_htif_data_stream(log)
    assert decoded == payload

    # And as an interpretation check: round-trip f32.
    floats_in = struct.pack("<2f", 1.5, -2.25)
    log = "".join(
        [
            f"tohost = 0x{(int.from_bytes(floats_in[i : i + 4], 'little') << 1):08x}\n"
            for i in range(0, len(floats_in), 4)
        ]
        + ["tohost = 0x00000001\n"]
    )
    decoded = parse_htif_data_stream(log)
    assert struct.unpack("<2f", decoded) == (1.5, -2.25)


# ---------------------------------------------------------------------------
# htif_data_stream_c — REQ-009 (guest helpers)
# ---------------------------------------------------------------------------


def test_htif_data_stream_c_exposes_paired_helpers() -> None:
    src = htif_data_stream_c()
    assert "htif_emit_u32" in src
    assert "htif_emit_bytes" in src
    # Documents the LSB-0 contract that parse_htif_data_stream relies on.
    assert "<< 1" in src
    # Reads the standard rocket-chip HTIF symbols.
    assert "__tohost" in src
    assert "extern volatile uint64_t" in src


def test_htif_data_stream_c_compiles_with_host_gcc(tmp_path) -> None:
    """The emitted C must actually compile (smoke check).

    Pairs the emitter with a stub `__tohost` so we don't need the
    rocket-chip symbol to validate syntax.
    """
    import shutil
    import subprocess

    if shutil.which("gcc") is None:
        import pytest

        pytest.skip("gcc not available")

    src = tmp_path / "stream.c"
    src.write_text(
        "#include <stdint.h>\n"
        "volatile uint64_t __tohost;\n"
        "volatile uint64_t __fromhost;\n" + htif_data_stream_c() + "int main(void) {\n"
        "    htif_emit_u32(0x42);\n"
        "    char buf[6] = {1,2,3,4,5,6};\n"
        "    htif_emit_bytes(buf, 6);\n"
        "    return 0;\n"
        "}\n"
    )
    proc = subprocess.run(
        ["gcc", "-c", "-Wall", "-Werror", "-o", str(tmp_path / "stream.o"), str(src)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# shared_dram_section — REQ-009 (linker fragment)
# ---------------------------------------------------------------------------


def test_shared_dram_section_emits_named_region() -> None:
    frag = shared_dram_section(symbol="my_results", size_bytes=0x4000)
    assert "PROVIDE(my_results" in frag
    assert "0x4000" in frag
    # Must be in REGION_DRAM (the system DRAM at 0x80000000+), NOT
    # the upper-DRAM TSI region (0x110000000+).
    assert "REGION_DRAM" in frag


def test_shared_dram_section_default_size_is_64k() -> None:
    frag = shared_dram_section()
    assert "compgen_shared" in frag
    assert "0x10000" in frag  # 64 KiB default
