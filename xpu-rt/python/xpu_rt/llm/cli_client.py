"""CLI-backed LLM adapters for Claude Code and Codex."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from compgen.llm._env import load_dotenv_map
from compgen.llm._prompt import (
    extract_markdown_artifacts,
    parse_json_payload,
    render_request_prompt,
    stringify_json_payload,
)
from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
)


def _merged_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in load_dotenv_map().items():
        env.setdefault(key, value)
    return env


def _extract_text_payload(stdout: str) -> str:
    stripped = stdout.strip()
    if not stripped:
        return ""
    try:
        payload = parse_json_payload(stripped)
    except json.JSONDecodeError:
        return stripped

    if isinstance(payload, dict):
        for key in ("result", "text", "content", "output", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        if "messages" in payload and isinstance(payload["messages"], list):
            texts = [
                message.get("text", "")
                for message in payload["messages"]
                if isinstance(message, dict)
            ]
            joined = "\n".join(text for text in texts if text)
            if joined:
                return joined
    if isinstance(payload, str):
        return payload
    return stripped


@dataclass
class ClaudeCLIClient:
    """Claude Code CLI adapter implementing ``CompGenLLMProtocol``."""

    model: str = "sonnet"
    command: str = "claude"
    working_dir: Path | None = None
    timeout_s: int = 180
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        prompt = render_request_prompt(request)
        model = request.config.model or self.model
        args = [self.command, "--print", "--no-session-persistence", "--tools", ""]
        if model:
            args.extend(["--model", model])
        args.extend(self.extra_args)
        args.append(prompt)

        t0 = time.perf_counter()
        proc = subprocess.run(
            args,
            cwd=str(self.working_dir) if self.working_dir else None,
            env=_merged_env(),
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI failed ({proc.returncode}): {proc.stderr.strip()}")

        raw_text = _extract_text_payload(proc.stdout)
        return GenerationResponse(
            raw_text=raw_text,
            parsed_artifacts=extract_markdown_artifacts(raw_text),
            model_id=model or "claude",
            latency_ms=latency_ms,
            metadata={"command": self.command, "stderr": proc.stderr.strip()},
        )

    def generate_structured(
        self, request: GenerationRequest, schema: dict[str, Any],
    ) -> GenerationResponse:
        prompt = render_request_prompt(request)
        model = request.config.model or self.model
        args = [
            self.command,
            "--print",
            "--no-session-persistence",
            "--tools",
            "",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, separators=(",", ":")),
        ]
        if model:
            args.extend(["--model", model])
        args.extend(self.extra_args)
        args.append(prompt)

        t0 = time.perf_counter()
        proc = subprocess.run(
            args,
            cwd=str(self.working_dir) if self.working_dir else None,
            env=_merged_env(),
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI failed ({proc.returncode}): {proc.stderr.strip()}")

        payload = parse_json_payload(proc.stdout)
        raw_text = stringify_json_payload(payload)
        return GenerationResponse(
            raw_text=raw_text,
            parsed_artifacts=[raw_text],
            model_id=model or "claude",
            latency_ms=latency_ms,
            metadata={"command": self.command, "stderr": proc.stderr.strip(), "format": "json"},
        )


@dataclass
class CodexCLIClient:
    """Codex CLI adapter implementing ``CompGenLLMProtocol``."""

    model: str = "gpt-5.4-mini"
    command: str = "codex"
    working_dir: Path | None = None
    timeout_s: int = 300
    sandbox_mode: str = "read-only"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return self._run(request, schema=None)

    def generate_structured(
        self, request: GenerationRequest, schema: dict[str, Any],
    ) -> GenerationResponse:
        return self._run(request, schema=schema)

    def _run(
        self,
        request: GenerationRequest,
        schema: dict[str, Any] | None,
    ) -> GenerationResponse:
        prompt = render_request_prompt(request)
        model = request.config.model or self.model

        with TemporaryDirectory(prefix="compgen_codex_") as tmpdir:
            out_path = Path(tmpdir) / "last_message.txt"
            args = [
                self.command,
                "exec",
                "--ephemeral",
                "--color",
                "never",
                "-s",
                self.sandbox_mode,
                "-o",
                str(out_path),
            ]
            if self.working_dir:
                args.extend(["-C", str(self.working_dir)])
            if model:
                args.extend(["-m", model])
            if schema is not None:
                schema_path = Path(tmpdir) / "schema.json"
                schema_path.write_text(json.dumps(schema))
                args.extend(["--output-schema", str(schema_path)])
            args.extend(self.extra_args)
            args.append("-")

            t0 = time.perf_counter()
            proc = subprocess.run(
                args,
                cwd=str(self.working_dir) if self.working_dir else None,
                env=_merged_env(),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            if proc.returncode != 0:
                raise RuntimeError(f"Codex CLI failed ({proc.returncode}): {proc.stderr.strip()}")

            raw_text = out_path.read_text().strip() if out_path.exists() else proc.stdout.strip()
            artifacts = extract_markdown_artifacts(raw_text)
            metadata = {"command": self.command, "stderr": proc.stderr.strip()}
            if schema is not None:
                payload = parse_json_payload(raw_text)
                raw_text = stringify_json_payload(payload)
                artifacts = [raw_text]
                metadata["format"] = "json"

            return GenerationResponse(
                raw_text=raw_text,
                parsed_artifacts=artifacts,
                model_id=model or "codex",
                latency_ms=latency_ms,
                metadata=metadata,
            )


__all__ = ["ClaudeCLIClient", "CodexCLIClient"]
