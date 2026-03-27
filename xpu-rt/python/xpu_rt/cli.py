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

import click


@click.group()
@click.version_option(package_name="compgen")
def main() -> None:
    """CompGen -- an LLM-driven compiler generator for heterogeneous hardware targets."""


@main.command()
@click.argument("profile", type=click.Path(exists=True))
def init_target(profile: str) -> None:
    """Initialize and validate a target profile.

    PROFILE: Path to a target_profile.yaml file.

    Outputs:
        - Validated profile summary to stdout
        - calibration_data/ (if --calibrate is passed, future)
    """
    click.echo(f"[init-target] Would validate target profile: {profile}")
    click.echo("  Expected output: validated profile summary, schema check results")
    click.echo("  Artifact path:   <profile>.validated.yaml")
    raise NotImplementedError("init-target is not yet implemented")


@main.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--inputs", type=click.Path(exists=True), required=True, help="Path to input spec YAML")
@click.option("--target", type=click.Path(exists=True), required=True, help="Path to target profile YAML")
@click.option("--output-dir", type=click.Path(), default="compgen_output", help="Output directory")
def analyze(model: str, inputs: str, target: str, output_dir: str) -> None:
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
    click.echo("  Stages:        0 (capture) -> 1 (IR) -> 2 (gap analysis)")
    click.echo("  Artifacts:     golden_*.pt, payload.mlir, kernel_contracts/, gap_analysis.json")
    raise NotImplementedError("analyze is not yet implemented")


@main.command()
@click.option("--target", type=click.Path(exists=True), required=True, help="Path to target profile YAML")
@click.option("--objective", type=click.Choice(["latency", "throughput", "memory", "energy"]), default="latency")
@click.option("--analysis-dir", type=click.Path(exists=True), required=True, help="Output from 'analyze' command")
@click.option("--output-dir", type=click.Path(), default="compgen_output", help="Output directory")
@click.option("--budget", type=int, default=50, help="Max LLM generation iterations")
def generate(target: str, objective: str, analysis_dir: str, output_dir: str, budget: int) -> None:
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
    click.echo("  Stages:        3 (transforms) -> 4 (kernels) -> 5 (execution plan)")
    click.echo("  Artifacts:     transforms/*.mlir, generated_kernels/, execution_plan.yaml, bundle/")
    raise NotImplementedError("generate is not yet implemented")


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--level", type=click.Choice(["structural", "functional", "performance", "formal", "all"]), default="all")
@click.option("--report", type=click.Path(), default=None, help="Output path for verification report JSON")
def verify(bundle_path: str, level: str, report: str | None) -> None:
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
    click.echo(f"  Level:         {level}")
    click.echo(f"  Report path:   {report or '<bundle_path>/verification_report.json'}")
    click.echo("  Ladder:        structural -> functional -> performance -> formal")
    raise NotImplementedError("verify is not yet implemented")


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--inputs", type=click.Path(exists=True), help="Path to input tensors")
@click.option("--device", type=str, default="cpu", help="Execution device")
def run(bundle_path: str, inputs: str | None, device: str) -> None:
    """Execute a generated bundle locally.

    BUNDLE_PATH: Path to a bundle directory (must contain manifest.json).

    Outputs:
        - Execution results to stdout
        - Profiling data (if --profile is passed, future)
    """
    click.echo(f"[run] Would execute bundle: {bundle_path}")
    click.echo(f"  Inputs:        {inputs or '<from bundle golden inputs>'}")
    click.echo(f"  Device:        {device}")
    click.echo("  Executor:      LocalExecutor")
    raise NotImplementedError("run is not yet implemented")


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--library", type=click.Path(), default="recipe_library", help="Recipe library path")
@click.option("--force", is_flag=True, help="Promote even if verification report has warnings")
def promote(bundle_path: str, library: str, force: bool) -> None:
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
    "--existing-backend",
    type=click.Choice(["merlin", "iree", "xla"]),
    default=None,
    help="Plug into existing backend instead of generating full backend",
)
def scaffold_target(hardware_spec: str, docs_dir: str | None, output_dir: str, existing_backend: str | None) -> None:
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
    click.echo(f"[scaffold-target] Would generate target package from: {hardware_spec}")
    click.echo(f"  Docs dir:          {docs_dir or '<none>'}")
    click.echo(f"  Output dir:        {output_dir}")
    click.echo(f"  Existing backend:  {existing_backend or '<standalone>'}")
    click.echo("  Steps:")
    click.echo("    1. Parse hardware spec -> TargetProfile")
    click.echo("    2. Classify target (Triton-friendly / accel / ukernel / hybrid)")
    click.echo("    3. Infer capability spec")
    click.echo("    4. Generate target package directory")
    click.echo("    5. Assess maturity (L0)")
    if existing_backend:
        click.echo(f"  Mode: integration layer for {existing_backend} (not full backend)")
    raise NotImplementedError("scaffold-target is not yet implemented")
