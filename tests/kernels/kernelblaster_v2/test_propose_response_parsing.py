"""Tests for the Gemini-response parser fallbacks.

The strict-JSON path is the happy path; the fallbacks (markdown
fences + raw C-fence extraction) keep the matrix runs from losing
LLM calls when Gemini emits malformed JSON. Coverage of the strict
path lives in the broader KB-v2 tests; this module specifically
covers the fallback chain.
"""

from __future__ import annotations

import json

import pytest

from xpu_rt.kernels.kernelblaster_v2.generators import (
    ProposeResponse,
    _extract_code_fence,
    _parse_propose_response,
    _strip_markdown_fence,
)


def test_parse_strict_json_happy_path() -> None:
    raw = json.dumps({"kernel_code": "void k(){}", "action": "tile-N=64"})
    r = _parse_propose_response(raw)
    assert r.kernel_code == "void k(){}"
    assert r.action == "tile-N=64"


def test_parse_strips_markdown_json_fence() -> None:
    """Gemini occasionally wraps JSON in a ```json fence."""
    raw = '```json\n{"kernel_code": "void k(){}", "action": "tile"}\n```'
    r = _parse_propose_response(raw)
    assert r.kernel_code == "void k(){}"
    assert r.action == "tile"


def test_parse_recovers_from_unterminated_string() -> None:
    """Reproduces the 0/30 production failure mode: kernel_code's
    value contains un-escaped characters that break strict JSON.
    The fallback must extract a C fence if the LLM included one."""
    raw = (
        '{\n  "kernel_code": "void launch_gpu_implementation('
        '\n    /* unterminated — JSON breaks here */'
        '\n\n```c\n'
        'void launch_gpu_implementation(void *output, void *input_A, void *input_B,\n'
        '                               int64_t M, int64_t K, int64_t N) {\n'
        '    /* actual fenced kernel body */\n'
        '}\n'
        '```'
    )
    r = _parse_propose_response(raw)
    assert r.kernel_code != ""
    assert "launch_gpu_implementation" in r.kernel_code
    assert r.action == "recovered_from_markdown_fence"
    assert r.language == "c"


def test_parse_falls_back_to_empty_when_nothing_usable() -> None:
    """When neither strict JSON nor a code fence is present, return
    an empty response so the agent loop records a clean failure."""
    raw = "this is just prose, no code, no JSON"
    r = _parse_propose_response(raw)
    assert r.kernel_code == ""
    assert r.action == ""


def test_strip_markdown_fence_handles_json_and_c_variants() -> None:
    assert _strip_markdown_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_markdown_fence('```c\nvoid k(){}\n```') == "void k(){}"
    assert _strip_markdown_fence('```python\nprint(1)\n```') == "print(1)"
    # No fence → returned unchanged
    assert _strip_markdown_fence("plain text") == "plain text"


def test_extract_code_fence_prefers_c_over_generic() -> None:
    """When both a c fence and a plain ``` fence are present, the c
    fence wins (it's the more specific match)."""
    raw = (
        "```\nsome generic text\n```\n"
        "```c\nvoid actual_kernel(){}\n```"
    )
    body = _extract_code_fence(raw)
    assert body is not None
    assert "void actual_kernel" in body


def test_extract_code_fence_handles_cpp_and_generic_fallback() -> None:
    raw = "```cpp\n#include <stdio.h>\n```"
    body = _extract_code_fence(raw)
    assert body is not None
    assert "stdio.h" in body

    raw_generic = "```\nplain text\n```"
    body2 = _extract_code_fence(raw_generic)
    assert body2 is not None
    assert body2.strip() == "plain text"


def test_parse_preserves_raw_response_on_strict_path() -> None:
    """Even on the strict-JSON happy path, raw_response is populated
    so the caller can re-inspect the original text."""
    raw = '{"kernel_code": "void k(){}", "action": "x"}'
    r = _parse_propose_response(raw)
    assert r.raw_response != ""


def test_parse_recovers_truncated_json_kernel_code() -> None:
    """Reproduces the Saturn v3 failure mode: Gemini emits valid-
    looking JSON then runs out of tokens mid-string. The 4th
    fallback strategy regex-extracts the kernel_code prefix."""
    # Truncated mid-kernel — no closing quote, no closing brace.
    raw = (
        '{\n  "kernel_code": "#include <riscv_vector.h>\\n'
        '#include <stdint.h>\\n\\n'
        'void launch_gpu_implementation('
        '\\n    void *output, void *input_A, void *input_B,'
        '\\n    int64_t M, int64_t K, int64_t N) {'
        '\\n    int8_t *A = (int8_t *)input_A;'
    )
    r = _parse_propose_response(raw)
    assert r.kernel_code != ""
    # Newlines must be decoded — the recovered text should contain
    # actual `#include` statements with real newlines, not \n
    # literals.
    assert "#include <riscv_vector.h>" in r.kernel_code
    assert "\n" in r.kernel_code
    assert r.action == "recovered_from_truncated_json"


def test_parse_extracts_well_formed_json_string_via_regex_path() -> None:
    """Regex path also handles well-formed JSON strings — useful
    when the JSON is technically valid but `from_dict` raises some
    downstream type error."""
    from xpu_rt.kernels.kernelblaster_v2.generators import (
        _extract_json_kernel_code_prefix,
    )

    raw = '{"kernel_code": "void k(){\\n  // body\\n}", "action": "x"}'
    recovered = _extract_json_kernel_code_prefix(raw)
    assert recovered == "void k(){\n  // body\n}"


def test_parse_handles_unicode_escapes() -> None:
    """JSON \\uXXXX escapes are decoded in the recovery path."""
    from xpu_rt.kernels.kernelblaster_v2.generators import (
        _extract_json_kernel_code_prefix,
    )

    raw = '{"kernel_code": "alpha=\\u03b1", "action": "x"}'
    recovered = _extract_json_kernel_code_prefix(raw)
    assert recovered == "alpha=α"
