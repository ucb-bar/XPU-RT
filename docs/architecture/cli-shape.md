# CompGen CLI Shape — Design Note

> Status: design contract for the MCP→CLI migration. Lock the shape here before
> code lands so every migrated subcommand looks the same.
>
> Audience: contributors migrating tools out of `python/compgen/mcp/tools/`
> into `python/compgen/cli.py`. Read alongside
> [`feedback_agentic_cli_first`](../../python/compgen/) (memory) and the
> migration audit in the project tracker.

## 1. Why this exists

CompGen's interface stack has three layers — CLI, MCP, skills — and each layer
must be **agentic-first** (agent-drivable as a primary user) AND
**fast-iterating** (`pip install -e .` is the only step between an edit and the
next agent invocation). MCP is the slowest layer to iterate because every
schema change forces a client reconnect. The CLI is the fastest. Therefore:

- **CLI is the default** for new functionality.
- **MCP is the graduation tier** for genuinely stateful session work — recipe
  iteration, the request/register/lookup/list quadruplets over `McpSession`.
- **Skills** wrap multi-step workflows that compose CLI (and MCP) calls.

This note specifies how a CLI subcommand must be shaped so that an agent can
drive it as competently as a typed MCP tool.

## 2. The contract every subcommand follows

Six rules. No exceptions without an explicit waiver in the command's
docstring.

### R1 — Logic lives in `compgen.api`, not in the click handler

Every subcommand is a thin wrapper around a function in `compgen.api` (or a
sibling `compgen.api.*` submodule). The same function is called by:

- The CLI handler (this note's subject).
- The MCP tool wrapper, if one exists.
- Any in-process Python user.

The click handler does only: argument parsing, `compgen.api` invocation, output
formatting, exit-code mapping. **No business logic in `cli.py`.** This rule is
what makes the CLI/MCP distinction a transport choice rather than a code
duplication tax.

### R2 — Every command has both human and machine output

Every subcommand accepts `--json`. Without it, output is human-readable text
via `click.echo` (the existing convention). With it, output is exactly **one
JSON object** on stdout, terminated by a newline, with the envelope from §3.

Logs (`structlog`) **always go to stderr**. Stdout is reserved for the result.
This is what makes the CLI safely pipeable and parseable by an agent.

### R3 — Errors are typed, structured, and stable

CLI handlers catch `compgen.runtime.errors.CompGenError` (and its subclasses)
and emit a structured error envelope (§3.2). Unexpected exceptions are caught
at the top level, logged with full traceback to stderr, and reported as
`{"type": "InternalError", ...}`. **Never** `print(traceback)` to stdout.

Exit codes (§4) are part of the contract — agents and CI rely on them.

### R4 — Subcommand naming mirrors MCP tool names

Predictable mapping reduces cognitive load for agents that have seen one
surface and are encountering the other:

| MCP tool name                       | CLI invocation                          |
|-------------------------------------|-----------------------------------------|
| `compgen_list_targets`              | `compgen targets list`                  |
| `compgen_describe_target`           | `compgen targets describe <name>`       |
| `compgen_register_target`           | `compgen targets register <profile>`    |
| `scan_vendor_repo`                  | `compgen vendor scan <repo>`            |
| `scaffold_vendor_package`           | `compgen vendor scaffold <descriptor>`  |
| `verify_vendor_package`             | `compgen vendor verify <package>`       |
| `etc_conformance_run`               | `compgen conformance run`               |
| `etc_conformance_summarize`         | `compgen conformance summarize <dir>`   |
| `etc_megakernel_inspect`            | `compgen conformance inspect <bundle>`  |
| `compgen_compile_torch_model`       | `compgen compile-torch <model>`         |
| `compgen_run_compiled_bundle`       | `compgen run <bundle>` (already exists) |
| `compgen_cublasdx_header_smoke`     | `compgen smoke cublasdx`                |
| `compgen_run_cuda_source`           | `compgen smoke cuda <source>`           |

The `compgen_` prefix is dropped (redundant under the `compgen` binary) and
verb-from-noun is split into `<noun-group> <verb>` for nounish commands.
**Existing pipeline stages stay flat** (`analyze`, `verify`, `run`, `promote`):
they are stage-aligned, not noun-scoped, and the convention is already
established.

### R5 — Shared option groups, declared once

The following options are common enough to share via click decorators in
`compgen.cli._options`:

- LLM selection — already at the root group; do not duplicate per-command.
- `--target / --target-profile <path>` — for any command that consumes a
  target profile YAML.
- `--output-dir <path>` — for any command that emits artifacts.
- `--json` — global, applied at the root group, read via context.
- `--quiet / -q` — suppress structlog output below WARNING (default INFO).
- `--strict / --no-strict` — for commands that take a strictness mode (e.g.,
  `compile-torch` with respect to bundle artifact slots).

Per-command options must be specific to that command's intent; reach for a
shared option before inventing a new one with similar semantics.

### R6 — `--help` is the agent's discovery contract

The agent has no MCP-style schema introspection here, so `--help` must be
rich enough to substitute. Every command's docstring **must** describe:

1. What the command does in one sentence.
2. The shape of its `--json` result on success (one or two lines).
3. The error types it can emit.
4. A worked example invocation.

If you wouldn't be comfortable handing an agent only `compgen <cmd> --help`
and expecting it to drive the command correctly, the docstring is not done.

## 3. JSON envelope

### 3.1 Success

```json
{
  "ok": true,
  "data": { ... command-specific payload ... },
  "schema": "compgen.targets.list/v1"
}
```

- `ok`: always `true` on success.
- `data`: command-specific, documented in the command's docstring.
- `schema`: a stable identifier of the form `<area>.<verb>/v<n>`. Bump the
  version when `data`'s shape changes incompatibly. Agents that pin a schema
  version get a clean break instead of silent drift.

### 3.2 Error

```json
{
  "ok": false,
  "error": {
    "type": "TargetNotFound",
    "message": "No target named 'saturn-opu' is registered.",
    "details": { "target": "saturn-opu" }
  },
  "schema": "compgen.error/v1"
}
```

- `type`: the typed error class name (e.g., `BundleEmissionError`,
  `VerificationFailed`, `TargetNotFound`). Stable; treat as part of the
  contract.
- `message`: human-readable single line.
- `details`: command-specific structured payload — the same data that the
  exception carries, serialized.

The envelope is identical in shape regardless of `--json`; without `--json`,
the human formatter chooses how to render it.

## 4. Exit codes

| Code | Meaning                                                          |
|------|------------------------------------------------------------------|
| 0    | Success.                                                         |
| 1    | User error — invalid args, missing file, malformed YAML, etc.    |
| 2    | Internal error — unexpected exception, programming bug.          |
| 3    | Verification failed — structural/functional/formal gate rejected.|
| 4    | Resource unavailable — GPU absent, vendor SDK missing, etc.      |

Agents read these codes to branch retry/repair logic. Do not invent new codes
without updating this table.

## 5. Output formatting

### 5.1 Human mode

- Use `click.echo` (existing convention). Never `print`.
- Lead with a single status line: `[area] verb result`. Example:
  `[targets] listed 4 registered profiles`.
- Tables via `click.echo` + simple alignment. Avoid third-party table
  libraries.
- On error, write the human-formatted error envelope to **stderr** and exit
  with the appropriate code.

### 5.2 JSON mode

- One object on stdout, newline-terminated, no other output on stdout.
- All `structlog` output goes to stderr regardless.
- No `tqdm`/spinners in JSON mode — they break parsers.
- The envelope is emitted exactly once, even on error.

## 6. State and side effects

### 6.1 No implicit state

A CLI command must not depend on hidden environment state beyond:

- `COMPGEN_LLM_*` env vars (resolved by the root group).
- `COMPGEN_HOME` if defined (for shared registries / knowledge stores).

Anything else — target profile, output directory, model path — is an explicit
argument or option. Two invocations with the same arguments in a clean
environment must produce the same result modulo timestamps.

### 6.2 Side effects are opt-in via `--output-dir`

Commands that write artifacts require an explicit `--output-dir`. They never
default to `/tmp` or scratch directories (this rule mirrors the artifact
contract in `CLAUDE.md` and matches `BundleStage`'s rejection of
`output_dir=None`).

### 6.3 Cross-session knowledge writes

The three `knowledge` MCP tools (`record_lesson`, `query_knowledge`,
`get_context_brief`) write to a process-wide `KnowledgeStore`. They get **CLI
aliases** (`compgen knowledge record|query|brief`) since they are not
session-bound. The MCP tools remain so they can be driven from within a
compile session.

## 7. The migration playbook

For each tool migrated from MCP to CLI:

1. **Extract the core function.** If logic currently lives inside the MCP tool
   handler, lift it to `compgen.api.<area>.<verb>` (or an existing module if
   it already has a natural home). The function takes typed arguments and
   returns a typed result; it raises typed errors. **It must not import
   `click` and must not touch stdout/stderr directly.**
2. **Wire the MCP tool to call it.** The MCP wrapper becomes a thin shim:
   parse, call, serialize. If the tool was stateless to begin with, this step
   is trivial.
3. **Add the CLI subcommand.** Use the option groups from R5; emit the
   envelope per §3; map errors to exit codes per §4.
4. **Write the docstring per R6.** Include the success-payload shape, the
   error types, and an example invocation.
5. **Add tests.** A CLI test mirrors the source tree
   (`tests/cli/test_<area>.py`). Cover: human-mode happy path, `--json` happy
   path, expected error → expected exit code + envelope. Use `CliRunner`.
6. **Decommission or keep.** Stateless tools whose only consumer was the
   agent can be removed from MCP entirely. Stateless tools that compose with
   in-session work (e.g., `verify_vendor_package` invoked mid-recipe) keep
   the MCP wrapper as a thin call to the same `compgen.api` function.
7. **Update docs.** Remove the tool from any MCP-tool documentation; add it
   to a CLI command index in `docs/cli/` (created on first use).

A migration PR should include all seven steps for every tool it touches.
Half-migrations leak duplicate logic.

## 8. What this does NOT cover

- **Long-running commands and progress.** Defer to a dedicated note when the
  first migrated command needs streaming output. Until then, all commands are
  expected to be short-running enough that a single envelope is appropriate.
- **Interactive subcommands.** None planned. The CLI is non-interactive by
  design — agentic-first means scriptable.
- **Authentication and credentials.** Inherited from the LLM root group;
  vendor SDK credentials are vendor-specific and documented per command.
- **Subcommand deprecation.** When a command's schema bumps incompatibly,
  ship the new version under a new schema string and keep the old one with a
  warning for one minor release.

## 9. Open questions

- Whether to add a `compgen list-commands --json` introspection command that
  emits a manifest of every subcommand with its docstring, options, and
  schema. Useful for skills that want to validate command availability before
  invoking. Decide once we have ≥10 migrated commands.
- Whether `--json` should default to true when stdout is not a TTY. Most
  modern tools (`gh`, `kubectl`) do not auto-switch; explicit is safer. Lean
  toward keeping `--json` opt-in.
- Whether to expose the JSON schemas as JSON Schema documents (for agent
  validation libraries). Defer; revisit when an agent needs it.
