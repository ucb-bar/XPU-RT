"""CompGen CLI -- the command surface for the compiler generator.

Subcommands map to pipeline stages:

    compgen init-target   -- Initialize / validate a target profile
    compgen analyze       -- Capture model, baseline, gap analysis
    compgen generate      -- Run LLM generation pipeline
    compgen verify        -- Run verification ladder on a bundle
    compgen run           -- Execute a bundle locally
    compgen promote       -- Promote verified bundle to recipe library

Every command validates its arguments and prints what it would do.
Implementation is deferred -- commands raise NotImplementedError after
printing their contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click

from compgen import __version__
from compgen.llm import (
    LLMSelection,
    PromptContext,
    SUPPORTED_PROVIDERS,
    apply_selection_to_env,
    build_llm_runtime,
    resolve_llm_selection,
    selection_status,
)
from compgen.llm.base import GenerationRequest, LLMConfig, Objective


@dataclass(frozen=True)
class CLIContext:
    """Shared CLI context, including project-level LLM selection."""

    llm: LLMSelection


def _emit_llm_selection(selection: LLMSelection) -> None:
    click.echo(f"  LLM backend:   {selection.provider}")
    click.echo(f"  LLM model:     {selection.model}")
    click.echo(f"  LLM record:    {'enabled' if selection.record else 'disabled'}")
    if selection.record:
        click.echo(f"  LLM logs:      {selection.record_dir}")


@click.group()
@click.option(
    "--llm-backend",
    type=click.Choice(SUPPORTED_PROVIDERS),
    default=None,
    help="Select the project-level LLM backend.",
)
@click.option(
    "--llm-model",
    type=str,
    default=None,
    help="Override the default model/alias for the selected backend.",
)
@click.option(
    "--llm-record-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory where LLM request/response logs are written.",
)
@click.option(
    "--llm-no-record",
    is_flag=True,
    help="Disable LLM request/response recording.",
)
@click.pass_context
@click.version_option(version=__version__)
def main(
    ctx: click.Context,
    llm_backend: str | None,
    llm_model: str | None,
    llm_record_dir: Path | None,
    llm_no_record: bool,
) -> None:
    """CompGen -- an LLM-driven compiler generator for heterogeneous hardware targets."""
    selection = resolve_llm_selection(
        llm_backend,
        model=llm_model,
        record=False if llm_no_record else None,
        record_dir=llm_record_dir,
    )
    apply_selection_to_env(selection)
    ctx.obj = CLIContext(llm=selection)


@main.group()
def llm() -> None:
    """Inspect or directly exercise the configured LLM backend."""


@llm.command("show")
@click.pass_obj
def llm_show(cli: CLIContext) -> None:
    """Show the resolved project-level LLM backend selection."""
    status = selection_status(cli.llm)
    click.echo("[llm] Resolved backend")
    click.echo(f"  Provider:      {status['provider']}")
    click.echo(f"  Model:         {status['model']}")
    click.echo(f"  Transport:     {status['transport']}")
    click.echo(f"  Source:        {status['source']}")
    click.echo(f"  Available:     {status['available']}")
    click.echo(f"  Detail:        {status['detail']}")
    click.echo(f"  Recording:     {status['recording']}")
    click.echo(f"  Record dir:    {status['record_dir']}")


@llm.command("smoke")
@click.option("--prompt", type=str, default="Say ready in one word.", show_default=True)
@click.option("--structured", is_flag=True, help="Request structured JSON output instead of plain text.")
@click.option(
    "--working-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Working directory used for CLI-backed providers and logs.",
)
@click.pass_obj
def llm_smoke(cli: CLIContext, prompt: str, structured: bool, working_dir: Path) -> None:
    """Run a small smoke test against the selected backend."""
    try:
        runtime = build_llm_runtime(cli.llm, working_dir=working_dir)
        request = GenerationRequest(
            prompt_template=prompt,
            context=PromptContext(
                model_ir_summary="smoke-test",
                target_profile_summary="cli-smoke",
                available_transforms=["tile", "eqsat"],
                kernel_contracts=[],
                objective=Objective.LATENCY,
                frontend_diagnostics_summary="graph_breaks=0",
                analysis_dossier_summary="regions=1",
            ),
            config=LLMConfig(model=cli.llm.model, temperature=0.0, max_tokens=128),
        )
        if structured:
            response = runtime.generate_structured(
                request,
                {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
            )
        else:
            response = runtime.generate(request)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("[llm] Smoke test response")
    click.echo(f"  Provider:      {cli.llm.provider}")
    click.echo(f"  Model:         {response.model_id}")
    click.echo(f"  Latency ms:    {response.latency_ms:.1f}")
    click.echo("  Output:")
    click.echo(response.raw_text.strip() or "<empty>")


@main.command()
@click.argument("profile", type=click.Path(exists=True))
@click.pass_obj
def init_target(cli: CLIContext, profile: str) -> None:
    """Initialize and validate a target profile.

    PROFILE: Path to a target_profile.yaml file.

    Outputs:
        - Validated profile summary to stdout
        - calibration_data/ (if --calibrate is passed, future)
    """
    click.echo(f"[init-target] Would validate target profile: {profile}")
    _emit_llm_selection(cli.llm)
    click.echo("  Expected output: validated profile summary, schema check results")
    click.echo("  Artifact path:   <profile>.validated.yaml")
    raise NotImplementedError("init-target is not yet implemented")


@main.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--inputs", type=click.Path(exists=True), required=True, help="Path to input spec YAML")
@click.option("--target", type=click.Path(exists=True), required=True, help="Path to target profile YAML")
@click.option("--output-dir", type=click.Path(), default="compgen_output", help="Output directory")
@click.pass_obj
def analyze(cli: CLIContext, model: str, inputs: str, target: str, output_dir: str) -> None:
    """Capture model, run baseline, and perform gap analysis.

    MODEL: Path to a Python model file (nn.Module).

    Pipeline stages executed:
        Stage 0 -- Capture & baseline (torch.export, torch.compile diagnostics)
        Stage 1 -- Build canonical payload IR (FX -> xDSL)
        Stage 2 -- Kernel gap analysis

    Outputs:
        - golden_inputs.pt, golden_outputs.pt
        - compile_baseline.json, graph_breaks.json
        - exported_program.pt2
        - payload.mlir
        - kernel_contracts/*.yaml
        - gap_analysis.json
    """
    click.echo(f"[analyze] Would analyze model: {model}")
    click.echo(f"  Input spec:    {inputs}")
    click.echo(f"  Target:        {target}")
    click.echo(f"  Output dir:    {output_dir}")
    _emit_llm_selection(cli.llm)
    click.echo("  Stages:        0 (capture) -> 1 (IR) -> 2 (gap analysis)")
    click.echo("  Artifacts:     golden_*.pt, payload.mlir, kernel_contracts/, gap_analysis.json")
    raise NotImplementedError("analyze is not yet implemented")


@main.command()
@click.option("--target", type=click.Path(exists=True), required=True, help="Path to target profile YAML")
@click.option("--objective", type=click.Choice(["latency", "throughput", "memory", "energy"]), default="latency")
@click.option("--analysis-dir", type=click.Path(exists=True), required=True, help="Output from 'analyze' command")
@click.option("--output-dir", type=click.Path(), default="compgen_output", help="Output directory")
@click.option("--budget", type=int, default=50, help="Max LLM generation iterations")
@click.pass_obj
def generate(
    cli: CLIContext,
    target: str,
    objective: str,
    analysis_dir: str,
    output_dir: str,
    budget: int,
) -> None:
    """Run the LLM generation pipeline.

    Pipeline stages executed:
        Stage 3 -- LLM transform generation (transform scripts, lowering params)
        Stage 4 -- Autocomp kernel search loop
        Stage 5 -- Execution plan generation

    Outputs:
        - transforms/*.mlir
        - generated_kernels/
        - execution_plan.yaml
        - memory_plan.yaml
        - bundle/manifest.json
    """
    click.echo("[generate] Would run generation pipeline")
    click.echo(f"  Target:        {target}")
    click.echo(f"  Objective:     {objective}")
    click.echo(f"  Analysis dir:  {analysis_dir}")
    click.echo(f"  Output dir:    {output_dir}")
    click.echo(f"  Budget:        {budget} iterations")
    _emit_llm_selection(cli.llm)
    click.echo("  Stages:        3 (transforms) -> 4 (kernels) -> 5 (execution plan)")
    click.echo("  Artifacts:     transforms/*.mlir, generated_kernels/, execution_plan.yaml, bundle/")
    raise NotImplementedError("generate is not yet implemented")


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--level", type=click.Choice(["structural", "functional", "performance", "formal", "all"]), default="all")
@click.option("--report", type=click.Path(), default=None, help="Output path for verification report JSON")
@click.pass_obj
def verify(cli: CLIContext, bundle_path: str, level: str, report: str | None) -> None:
    """Run the verification ladder on a generated bundle.

    BUNDLE_PATH: Path to a bundle directory (must contain manifest.json).

    Verification levels:
        structural   -- Schema validation, IR verifier, parser round-trip, CHECK assertions
        functional   -- Eager vs compiled outputs, randomized tensor tests, dynamic shapes
        performance  -- Compile time, warm run time, graph coverage, bytes moved
        formal       -- Translation validation, rewrite verification (optional, solver-backed)
        all          -- Run all levels in order

    Outputs:
        - verification_report.json
    """
    click.echo(f"[verify] Would verify bundle: {bundle_path}")
    _emit_llm_selection(cli.llm)
    click.echo(f"  Level:         {level}")
    click.echo(f"  Report path:   {report or '<bundle_path>/verification_report.json'}")
    click.echo("  Ladder:        structural -> functional -> performance -> formal")
    raise NotImplementedError("verify is not yet implemented")


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--inputs", type=click.Path(exists=True), help="Path to input tensors")
@click.option("--device", type=str, default="cpu", help="Execution device")
@click.pass_obj
def run(cli: CLIContext, bundle_path: str, inputs: str | None, device: str) -> None:
    """Execute a generated bundle locally.

    BUNDLE_PATH: Path to a bundle directory (must contain manifest.json).

    Outputs:
        - Execution results to stdout
        - Profiling data (if --profile is passed, future)
    """
    click.echo(f"[run] Would execute bundle: {bundle_path}")
    _emit_llm_selection(cli.llm)
    click.echo(f"  Inputs:        {inputs or '<from bundle golden inputs>'}")
    click.echo(f"  Device:        {device}")
    click.echo("  Executor:      LocalExecutor")
    raise NotImplementedError("run is not yet implemented")


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--library", type=click.Path(), default="recipe_library", help="Recipe library path")
@click.option("--force", is_flag=True, help="Promote even if verification report has warnings")
@click.pass_obj
def promote(cli: CLIContext, bundle_path: str, library: str, force: bool) -> None:
    """Promote a verified bundle to the recipe library.

    BUNDLE_PATH: Path to a bundle directory (must contain verification_report.json).

    Promotion key: hash(target_profile) + hash(model_ir) + hash(objective)

    Requirements:
        - verification_report.json must exist and show all-pass
        - Bundle must contain all required artifacts

    Outputs:
        - Recipe added to recipe_library/ keyed by promotion key
        - Audit log entry
    """
    click.echo(f"[promote] Would promote bundle: {bundle_path}")
    _emit_llm_selection(cli.llm)
    click.echo(f"  Library:       {library}")
    click.echo(f"  Force:         {force}")
    click.echo("  Key:           hash(target) + hash(model_ir) + hash(objective)")
    click.echo("  Requires:      verification_report.json with all-pass status")
    raise NotImplementedError("promote is not yet implemented")


@main.command()
@click.argument("hardware_spec", type=click.Path(exists=True))
@click.option("--docs-dir", type=click.Path(exists=True), default=None, help="Hardware documentation directory")
@click.option("--output-dir", type=click.Path(), default="target_packages", help="Output directory")
@click.option(
    "--pack",
    "packs",
    multiple=True,
    help="Extension pack path or builtin pack name to compose into the target package.",
)
@click.option(
    "--existing-backend",
    type=click.Choice(["merlin", "iree", "xla"]),
    default=None,
    help="Plug into existing backend instead of generating full backend",
)
@click.pass_obj
def scaffold_target(
    cli: CLIContext,
    hardware_spec: str,
    docs_dir: str | None,
    output_dir: str,
    packs: tuple[str, ...],
    existing_backend: str | None,
) -> None:
    """Generate a target enablement package from a hardware specification.

    HARDWARE_SPEC: Path to a hardware spec file (YAML) or target profile.

    This generates a target package -- NOT a full compiler. The package contains:

      1. target_profile.yaml     -- validated hardware description
      2. capabilities.yaml       -- op-to-backend-lane mapping
      3. constraints.yaml        -- system constraints
      4. recipes/                -- transform templates for this target class
      5. kernels/                -- kernel search configs per backend lane
      6. ir/                     -- accel dialect skeleton (if needed)
      7. verification/           -- test corpus, CHECK files, golden harness
      8. runtime/                -- driver config, planner constraints

    Target classification:
      - TRITON_FRIENDLY:  GPU-like, Triton covers most ops
      - ACCEL_NATIVE:     Custom accelerator, needs accel dialect
      - UKERNEL_RUNTIME:  Firmware-driven NPU, runtime API calls
      - HYBRID:           Mixed system, multiple lanes

    If --existing-backend is set (merlin/iree/xla), generates an integration
    layer that plugs into the existing backend rather than a full backend.

    Maturity levels:
      L0: Recognized  (profile valid, capabilities inferred)
      L1: Correctness (fallback path works)
      L2: Optimized   (recipes beat fallback)
      L3: Promoted    (verified, stable, reusable)
    """
    from compgen.targets.package import generate_target_package
    from compgen.targets.schema import load_profile
    import yaml

    spec_path = Path(hardware_spec)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    generated_output = output_root / f"{spec_path.stem}_generated"

    raw_spec = yaml.safe_load(spec_path.read_text()) or {}
    if isinstance(raw_spec, dict) and "devices" in raw_spec:
        profile = load_profile(spec_path)
        click.echo(f"[scaffold-target] Loaded target profile directly: {hardware_spec}")
    else:
        from compgen.targetgen.generate import generate_target

        generated = generate_target(spec_path, generated_output)
        profile = generated.profile
        click.echo(f"[scaffold-target] Generated target from hardware spec: {hardware_spec}")

    package_dir = output_root / f"{profile.name}_package"
    package = generate_target_package(
        profile,
        package_dir,
        docs_dir=docs_dir,
        existing_backend=existing_backend,
        extension_packs=packs or None,
    )

    click.echo(f"  Docs dir:          {docs_dir or '<none>'}")
    click.echo(f"  Output dir:        {output_dir}")
    click.echo(f"  Existing backend:  {existing_backend or '<standalone>'}")
    click.echo(f"  Package dir:       {package.root}")
    click.echo(f"  Target class:      {package.manifest.target_class.value}")
    click.echo(f"  Maturity:          {package.maturity.name}")
    click.echo(f"  Extension packs:   {', '.join(package.manifest.composed_from_packs) or '<none>'}")
    _emit_llm_selection(cli.llm)
    if package.manifest.sealed_surfaces:
        click.echo(f"  Sealed surfaces:   {', '.join(package.manifest.sealed_surfaces)}")
    if package.manifest.generation_apertures:
        click.echo(f"  Apertures:         {', '.join(package.manifest.generation_apertures)}")


if __name__ == "__main__":
    main()
