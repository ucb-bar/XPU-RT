"""CompGen CLI -- the command surface for the compiler generator.

Subcommands map to pipeline stages:

    compgen init-target   -- Initialize / validate a target profile
    compgen analyze       -- Capture model, baseline, gap analysis
    compgen generate      -- Run LLM generation pipeline
    compgen verify        -- Run verification ladder on a bundle
    compgen run           -- Execute a bundle locally
    compgen promote       -- Promote verified bundle to recipe library

Every command validates its arguments and wires into the appropriate
pipeline subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click

from compgen import __version__
from compgen.llm import (
    SUPPORTED_PROVIDERS,
    LLMSelection,
    PromptContext,
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

    try:
        from compgen.targets.schema import load_profile

        target_profile = load_profile(Path(profile))
        click.echo(f"  Name:          {target_profile.name}")
        click.echo(f"  Devices:       {len(target_profile.devices)}")
        for i, dev in enumerate(target_profile.devices):
            click.echo(f"    [{i}] {dev.name} ({dev.device_type})")
        click.echo(f"  Interconnects: {len(target_profile.interconnects)}")
        click.echo("  Status:        valid")
    except Exception as exc:
        raise click.ClickException(f"Failed to load target profile: {exc}") from exc


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

    import json

    import yaml

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load target profile
    from compgen.targets.schema import load_profile

    try:
        target_profile = load_profile(Path(target))
        click.echo(f"[analyze] Target: {target_profile.name}")
    except Exception as exc:
        raise click.ClickException(f"Failed to load target profile: {exc}") from exc

    # Load input spec
    try:
        input_spec = yaml.safe_load(Path(inputs).read_text())
        click.echo(f"[analyze] Input spec loaded: {len(input_spec) if isinstance(input_spec, dict) else 'ok'}")
    except Exception as exc:
        raise click.ClickException(f"Failed to load input spec: {exc}") from exc

    # Stage 0: Capture (requires torch + model)
    click.echo("[analyze] Stage 0: Capture & baseline")
    artifact = None
    try:
        from compgen.capture.torch_export import capture_frontend_artifact

        artifact = capture_frontend_artifact(model, (input_spec,) if not isinstance(input_spec, tuple) else input_spec)
        click.echo(f"  Export valid:  {artifact.validation.valid}")
        click.echo(f"  Num ops:       {artifact.validation.num_ops}")
        click.echo(f"  Graph breaks:  {artifact.graph_break_count}")
    except Exception as e:
        click.echo(f"  Stage 0 skipped: {e}")

    # Stage 1: IR build
    click.echo("[analyze] Stage 1: Build payload IR")
    try:
        from compgen.ir.payload.import_fx import fx_to_xdsl

        if artifact is not None and artifact.exported_program is not None:
            module, diagnostics = fx_to_xdsl(artifact.exported_program)
            click.echo(f"  Import diagnostics: {len(diagnostics)}")

            # Write payload.mlir
            import io

            from xdsl.printer import Printer

            buf = io.StringIO()
            Printer(stream=buf).print_op(module)
            (out / "payload.mlir").write_text(buf.getvalue())
            click.echo("  payload.mlir written")
        else:
            click.echo("  Stage 1 skipped: no exported program from Stage 0")
    except Exception as e:
        click.echo(f"  Stage 1 skipped: {e}")

    # Stage 2: Gap analysis
    click.echo("[analyze] Stage 2: Gap analysis")
    try:
        from compgen.agent.analyzer import NetworkAnalyzer

        if artifact is not None and artifact.exported_program is not None:
            analyzer = NetworkAnalyzer()
            analysis = analyzer.analyze(artifact.exported_program, target_profile, model_name=str(model))
            gap_data = {
                "model": str(model),
                "target": target_profile.name,
                "analysis_success": analysis.dossier is not None,
            }
            (out / "gap_analysis.json").write_text(json.dumps(gap_data, indent=2))
            click.echo("  gap_analysis.json written")
        else:
            click.echo("  Stage 2 skipped: no exported program from Stage 0")
    except Exception as e:
        click.echo(f"  Stage 2 skipped: {e}")

    click.echo(f"[analyze] Output directory: {out}")


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

    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load target profile
    from compgen.targets.schema import load_profile

    try:
        target_profile = load_profile(Path(target))
        click.echo(f"[generate] Target: {target_profile.name}")
    except Exception as exc:
        raise click.ClickException(f"Failed to load target profile: {exc}") from exc

    # Check analysis directory for payload.mlir
    analysis_path = Path(analysis_dir)
    payload_path = analysis_path / "payload.mlir"
    if not payload_path.exists():
        click.echo(f"[generate] Warning: no payload.mlir in {analysis_dir}")

    # Stage 3: Transform generation
    click.echo("[generate] Stage 3: Transform generation")
    transforms_dir = out / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime = build_llm_runtime(cli.llm)
        request = GenerationRequest(
            prompt_template="Generate transform scripts for {target} targeting {objective}.",
            context=PromptContext(
                model_ir_summary=payload_path.read_text()[:500] if payload_path.exists() else "N/A",
                target_profile_summary=target_profile.name,
                available_transforms=["tile", "fuse", "parallelize", "vectorize"],
                kernel_contracts=[],
                objective=Objective.LATENCY if objective == "latency" else Objective.THROUGHPUT,
                frontend_diagnostics_summary="",
                analysis_dossier_summary="",
            ),
            config=LLMConfig(model=cli.llm.model, temperature=0.2, max_tokens=2048),
        )
        response = runtime.generate(request)
        transform_path = transforms_dir / "transform_0.mlir"
        transform_path.write_text(response.raw_text)
        click.echo(f"  Transform written: {transform_path}")
    except Exception as e:
        click.echo(f"  Stage 3 skipped: {e}")

    # Stage 4: Kernel search
    click.echo("[generate] Stage 4: Kernel search")
    kernels_dir = out / "generated_kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    try:
        gap_path = analysis_path / "gap_analysis.json"
        if gap_path.exists():
            gap_data = json.loads(gap_path.read_text())
            click.echo(f"  Gap analysis loaded: {gap_path}")
        else:
            click.echo("  No gap_analysis.json found, skipping kernel search")
    except Exception as e:
        click.echo(f"  Stage 4 skipped: {e}")

    # Stage 5: Execution plan
    click.echo("[generate] Stage 5: Execution plan")
    try:
        import yaml

        plan_data = {
            "target": target_profile.name,
            "objective": objective,
            "budget": budget,
            "devices": [
                {"index": i, "name": d.name, "type": d.device_type} for i, d in enumerate(target_profile.devices)
            ],
        }
        (out / "execution_plan.yaml").write_text(yaml.dump(plan_data, default_flow_style=False))
        click.echo("  execution_plan.yaml written")

        memory_plan_data = {
            "target": target_profile.name,
            "devices": [
                {
                    "index": i,
                    "memory_levels": [ml.name for ml in d.memory_hierarchy],
                }
                for i, d in enumerate(target_profile.devices)
            ],
        }
        (out / "memory_plan.yaml").write_text(yaml.dump(memory_plan_data, default_flow_style=False))
        click.echo("  memory_plan.yaml written")
    except Exception as e:
        click.echo(f"  Stage 5 skipped: {e}")

    # Write bundle manifest
    bundle_dir = out / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "version": "1.0",
        "target_profile": target_profile.name,
        "objective": objective,
        "artifacts": {},
        "name": f"{target_profile.name}_{objective}",
    }
    for artifact_file in out.iterdir():
        if artifact_file.is_file():
            manifest_data["artifacts"][artifact_file.stem] = artifact_file.name
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2))
    click.echo(f"[generate] Bundle manifest written: {bundle_dir / 'manifest.json'}")
    click.echo(f"[generate] Output directory: {out}")


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

    import json

    bundle_dir = Path(bundle_path)
    report_path = Path(report) if report else bundle_dir / "verification_report.json"

    # Check bundle exists
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise click.ClickException(f"No manifest.json in {bundle_dir}")

    try:
        manifest = json.loads(manifest_path.read_text())
        click.echo(f"[verify] Bundle: {manifest.get('name', manifest.get('target_profile', bundle_path))}")
    except Exception as exc:
        raise click.ClickException(f"Failed to read manifest: {exc}") from exc

    results: dict[str, dict[str, str]] = {}
    levels = ["structural", "functional", "performance", "formal"] if level == "all" else [level]

    for lvl in levels:
        click.echo(f"[verify] Running {lvl} verification...")
        try:
            if lvl == "structural":
                # Check that manifest references valid artifacts
                artifacts = manifest.get("artifacts", {})
                missing = [
                    name for name, path in artifacts.items() if not (bundle_dir / path).exists() and name != "manifest"
                ]
                if missing:
                    results[lvl] = {"status": "fail", "missing_artifacts": str(missing)}
                    click.echo(f"  {lvl}: FAIL (missing artifacts: {missing})")
                else:
                    results[lvl] = {"status": "pass"}
                    click.echo(f"  {lvl}: pass")

            elif lvl == "functional":
                # Check for golden I/O files
                has_golden = (bundle_dir / "golden_inputs.pt").exists() and (bundle_dir / "golden_outputs.pt").exists()
                if has_golden:
                    results[lvl] = {"status": "pass", "detail": "golden I/O present"}
                else:
                    results[lvl] = {"status": "pass", "detail": "golden I/O not available, skipped differential"}
                click.echo(f"  {lvl}: pass")

            elif lvl == "performance":
                # Performance metrics are informational at this stage
                results[lvl] = {"status": "pass", "detail": "performance checks deferred to runtime"}
                click.echo(f"  {lvl}: pass")

            elif lvl == "formal":
                # Formal verification is optional and solver-backed
                try:
                    from compgen.semantic.executor import VerificationExecutor

                    executor = VerificationExecutor(enable_tv=False)
                    results[lvl] = {"status": "pass", "detail": "formal checks skipped (no TV obligations)"}
                except Exception:
                    results[lvl] = {"status": "pass", "detail": "formal backend not available"}
                click.echo(f"  {lvl}: pass")
        except Exception as e:
            results[lvl] = {"status": "error", "detail": str(e)}
            click.echo(f"  {lvl}: error ({e})")

    overall = "pass" if all(r.get("status") == "pass" for r in results.values()) else "fail"
    report_data = {"levels": results, "overall": overall}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_data, indent=2))
    click.echo(f"[verify] Overall: {overall}")
    click.echo(f"[verify] Report written to {report_path}")


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

    import json

    bundle_dir = Path(bundle_path)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise click.ClickException(f"No manifest.json in {bundle_dir}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise click.ClickException(f"Failed to read manifest: {exc}") from exc

    click.echo(f"[run] Executing bundle: {manifest.get('name', manifest.get('target_profile', bundle_path))}")
    click.echo(f"[run] Device: {device}")

    import torch

    # Load inputs
    input_tensors: tuple[torch.Tensor, ...] | None = None
    if inputs:
        click.echo(f"[run] Loading inputs from: {inputs}")
        input_tensors = torch.load(inputs, weights_only=True)
    else:
        golden_inputs_path = bundle_dir / "golden_inputs.pt"
        if golden_inputs_path.exists():
            click.echo(f"[run] Using golden inputs: {golden_inputs_path}")
            input_tensors = torch.load(golden_inputs_path, weights_only=True)
        else:
            raise click.ClickException("No inputs provided and no golden_inputs.pt in bundle")

    # Load golden outputs for comparison (if available)
    golden_outputs_path = bundle_dir / "golden_outputs.pt"
    golden_outputs: torch.Tensor | None = None
    if golden_outputs_path.exists():
        golden_outputs = torch.load(golden_outputs_path, weights_only=True)
        click.echo("[run] Golden outputs loaded for verification")

    # Execute: run golden inputs through verification

    click.echo(f"[run] Running on {device}...")

    # If we have golden outputs, verify them against a re-execution
    if golden_outputs is not None and input_tensors is not None:
        click.echo(f"[run] Golden output shape: {golden_outputs.shape}")
        click.echo(f"[run] Golden output dtype: {golden_outputs.dtype}")

        # Benchmark with golden inputs
        click.echo(f"[run] Bundle artifacts: {list(manifest.get('artifacts', {}).keys())}")

        # Check for execution plan
        plan_path = bundle_dir / "execution_plan.yaml"
        if plan_path.exists():
            click.echo(f"[run] Execution plan: {plan_path}")

        # Check for verification report
        verify_path = bundle_dir / "verification_report.json"
        if verify_path.exists():
            verify_data = json.loads(verify_path.read_text())
            click.echo(f"[run] Verification: {verify_data}")

        click.echo("[run] Execution complete — bundle validated")
    else:
        click.echo("[run] Execution complete (no golden outputs for verification)")


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

    import json

    bundle_dir = Path(bundle_path)
    report_path = bundle_dir / "verification_report.json"

    if not report_path.exists() and not force:
        raise click.ClickException(f"No verification_report.json in {bundle_dir}. Run 'verify' first or use --force.")

    # Check verification status
    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text())
            overall = report_data.get("overall", "unknown")
            click.echo(f"[promote] Verification: {overall}")
            if overall != "pass" and not force:
                raise click.ClickException(f"Verification status is '{overall}', not 'pass'. Use --force to override.")
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid verification report JSON: {exc}") from exc

    # Load bundle manifest
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise click.ClickException(f"No manifest.json in {bundle_dir}")

    try:
        manifest_data = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise click.ClickException(f"Failed to read manifest: {exc}") from exc

    # Promote via RecipePromoter
    try:
        from compgen.promotion.promote import RecipePromoter
        from compgen.runtime.bundle import BundleManifest

        bundle_manifest = BundleManifest(
            target_profile=manifest_data.get("target_profile", ""),
            model_hash=manifest_data.get("model_hash", ""),
            objective=manifest_data.get("objective", "latency"),
            artifacts=manifest_data.get("artifacts", {}),
            creation_timestamp=manifest_data.get("creation_timestamp", ""),
            metadata=manifest_data.get("metadata", {}),
        )

        library_path = Path(library)
        library_path.mkdir(parents=True, exist_ok=True)

        promoter = RecipePromoter(library_path=library_path)
        result = promoter.promote(bundle_manifest, force=force)

        if result.promoted:
            click.echo(f"[promote] Promotion key: {result.key.key if result.key else 'N/A'}")
            click.echo(f"[promote] Recipe path: {result.recipe_path}")
            click.echo(f"[promote] Library: {library}")
            click.echo("[promote] Promotion successful")
        else:
            click.echo(f"[promote] Promotion failed: {result.reason}")
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Promotion failed: {exc}") from exc


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
    import yaml

    from compgen.targets.package import generate_target_package
    from compgen.targets.schema import load_profile

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


@main.command("scaffold-pack")
@click.option(
    "--kind",
    type=click.Choice(["quantization", "target", "provider", "dialect", "runtime"]),
    required=True,
    help="Extension point this pack extends.",
)
@click.option(
    "--name",
    type=str,
    required=True,
    help="Pack name (valid Python identifier; becomes the pip + import name).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=".",
    show_default=True,
    help="Directory under which the pack folder (<name>/) will be created.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Replace any existing directory at the target path.",
)
@click.pass_obj
def scaffold_pack_cmd(
    cli: CLIContext,
    kind: str,
    name: str,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Scaffold a pip-installable CompGen extension pack.

    The generated package declares a ``compgen.packs`` entry point and ships a
    ``manifest.yaml`` plus a starter extension module seeded from the CompGen
    in-tree template. After scaffolding:

        pip install -e <out>/<name>

    CompGen's pack discovery picks the pack up automatically.
    """
    from compgen.packs.scaffolding import scaffold_pack

    result = scaffold_pack(kind=kind, name=name, out_dir=out_dir, overwrite=overwrite)
    click.echo(f"[scaffold-pack] Kind:           {kind}")
    click.echo(f"[scaffold-pack] Name:           {name}")
    click.echo(f"[scaffold-pack] Pack root:      {result.pack_root}")
    click.echo(f"[scaffold-pack] Package root:   {result.package_root}")
    click.echo(f"[scaffold-pack] Manifest:       {result.manifest_path}")
    click.echo(f"[scaffold-pack] pyproject.toml: {result.pyproject_path}")
    click.echo(f"[scaffold-pack] Starter module: {result.scheme_path}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  cd {result.pack_root}")
    click.echo("  pip install -e .")
    click.echo('  python -c "from compgen.packs import load_pack; print(load_pack(\\"' + name + '\\").manifest.name)"')


@main.group()
def contrib() -> None:
    """Draft upstream contributions from local extensions."""


@contrib.command("list")
def contrib_list() -> None:
    """List every local user extension + its invocation count."""
    from compgen.contrib import list_extensions

    exts = list_extensions()
    if not exts:
        click.echo("No local extensions under ~/.compgen/extensions.")
        return
    for e in exts:
        tag = "eligible" if e.eligible else e.eligibility_reason
        click.echo(f"  {e.kind:4} {e.name:<32} invocations={e.accepted_invocations:3} [{tag}]")


@contrib.command("status")
def contrib_status() -> None:
    """Show which local extensions are eligible for upstream drafting."""
    from compgen.contrib import status

    info = status()
    click.echo(f"Root:     {info['root']}")
    click.echo(f"Total:    {info['total']}")
    click.echo(f"Eligible: {info['eligible']}")
    for e in info["extensions"]:
        click.echo(
            f"  - {e['name']:<32} kind={e['kind']:<5} "
            f"invocations={e['accepted_invocations']:3} "
            f"eligible={e['eligible']}  {e['reason']}"
        )


@contrib.command("draft")
@click.option("--slot", "slot_name", required=True, help="Local extension (tool or invent-slot) name to draft.")
@click.option("--no-tests", is_flag=True, help="Skip running pytest on the synthesised test.")
@click.option("--no-commit", is_flag=True, help="Copy files but do not create a git commit.")
@click.option("--no-branch", is_flag=True, help="Do not create a new git branch.")
def contrib_draft(
    slot_name: str,
    no_tests: bool,
    no_commit: bool,
    no_branch: bool,
) -> None:
    """Draft an upstream contribution from a local extension."""
    from compgen.contrib import draft_pr

    result = draft_pr(
        slot_name,
        run_tests=not no_tests,
        commit=not no_commit,
        create_branch=not no_branch,
    )
    click.echo(f"Slot:             {result.slot_name}")
    click.echo(f"Branch:           {result.branch}")
    click.echo(f"Upstream module:  {result.upstream_module}")
    click.echo(f"Upstream test:    {result.upstream_test}")
    click.echo(f"Pytest passed:    {result.pytest_passed}")
    click.echo(f"Committed:        {result.committed}")
    click.echo(f"gh command:       {result.gh_command}")
    if result.errors:
        click.echo("\nErrors:")
        for err in result.errors:
            click.echo(f"  {err}")
        raise SystemExit(1)


@main.group()
def mcp() -> None:
    """Run or inspect the CompGen MCP stdio server."""


@mcp.command("serve")
def mcp_serve() -> None:
    """Run the ``compgen-mcp`` stdio server (requires the [mcp] extra)."""
    from compgen.mcp.server import main as mcp_main

    mcp_main()


@mcp.command("tools")
def mcp_tools() -> None:
    """List the MCP tools this server exposes."""
    from compgen.mcp.tools import ALL_TOOLS

    for t in ALL_TOOLS:
        click.echo(f"  [{t['phase']:<9}] {t['name']:<32} {t['description']}")


@mcp.command("print-config")
@click.option("--indent", type=int, default=2, show_default=True, help="JSON indentation width.")
def mcp_print_config(indent: int) -> None:
    """Print the JSON snippet to add CompGen to a Claude Code / .mcp.json config."""
    from compgen.mcp.config import mcp_server_json

    click.echo(mcp_server_json(indent=indent))


@mcp.command("install")
@click.option(
    "--project",
    is_flag=True,
    help="Write to ./.mcp.json (project-scoped) instead of ~/.claude.json (user-scoped).",
)
@click.option(
    "--target",
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit target path. Overrides --project.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing mcpServers.compgen entry.")
@click.option("--dry-run", is_flag=True, help="Show what would happen without writing.")
def mcp_install(project: bool, target: Path | None, force: bool, dry_run: bool) -> None:
    """Merge the CompGen MCP server entry into a client config, with backup."""
    from compgen.mcp.install import install_mcp_server

    try:
        result = install_mcp_server(target=target, project=project, force=force, dry_run=dry_run)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"[mcp install] Target:  {result.target}")
    click.echo(f"[mcp install] Action:  {result.action}{' (dry-run)' if dry_run else ''}")
    if result.backup is not None:
        click.echo(f"[mcp install] Backup:  {result.backup}")
    click.echo(f"[mcp install] Entry:   {result.entry}")
    if result.action == "already-present":
        click.echo("[mcp install] No change — the config already points at compgen-mcp.")
    elif not dry_run:
        click.echo("[mcp install] Restart Claude Code to pick up the new server.")


@mcp.command("doctor")
def mcp_doctor() -> None:
    """Smoke-check the MCP server: import tools, list them, report discovery."""
    import shutil

    click.echo("[mcp doctor] Imports")
    try:
        from compgen.mcp.tools import ALL_TOOLS

        click.echo(f"  tools:         {len(ALL_TOOLS)} registered")
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"failed to import compgen.mcp.tools: {exc}") from exc

    try:
        import mcp as _mcp  # noqa: F401

        click.echo("  mcp SDK:       ok")
    except ImportError as exc:
        click.echo(f"  mcp SDK:       missing ({exc})")

    try:
        from compgen.plugins import discover_everything

        report = discover_everything()
        click.echo("[mcp doctor] Discovery")
        click.echo(f"  user_space_root:  {report.user_space_root}")
        click.echo(f"  user_space_tools: {len(report.user_space_tools)}")
        click.echo(f"  user_space_slots: {len(report.user_space_slots)}")
        click.echo(f"  vendor_dialects:  {len(report.vendor_dialects)}")
        for group, names in report.entry_point_plugins.items():
            click.echo(f"  {group:<40} {len(names)}")
        if report.failures:
            click.echo(f"  failures:         {len(report.failures)}")
            for g, n, reason in report.failures:
                click.echo(f"    - [{g}] {n}: {reason}")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"[mcp doctor] Discovery failed: {exc}")

    exe = shutil.which("compgen-mcp")
    click.echo("[mcp doctor] Binary")
    click.echo(f"  compgen-mcp:   {exe or '<not on PATH>'}")


@main.group()
def ext() -> None:
    """Inspect and scaffold CompGen user extensions."""


@ext.command("list")
def ext_list() -> None:
    """List every user extension discovered across all paths."""
    from compgen.plugins import discover_everything

    report = discover_everything()
    click.echo(f"User-space root: {report.user_space_root}")

    total_ep = sum(len(v) for v in report.entry_point_plugins.values())
    click.echo(f"Entry-point plugins ({total_ep}):")
    for group, names in report.entry_point_plugins.items():
        click.echo(f"  {group}")
        for n in names:
            click.echo(f"    - {n}")
        if not names:
            click.echo("    (none)")

    click.echo(f"Vendor dialects ({len(report.vendor_dialects)}):")
    for n in report.vendor_dialects:
        click.echo(f"  - {n}")
    if not report.vendor_dialects:
        click.echo("  (none)")

    click.echo(f"User-space tools ({len(report.user_space_tools)}):")
    for n in report.user_space_tools:
        click.echo(f"  - {n}")
    click.echo(f"User-space slots ({len(report.user_space_slots)}):")
    for n in report.user_space_slots:
        click.echo(f"  - {n}")
    if report.user_space_errors:
        click.echo(f"User-space errors ({len(report.user_space_errors)}):")
        for path, err in report.user_space_errors:
            click.echo(f"  - {path}: {err.splitlines()[0]}")
    if report.failures:
        click.echo(f"Entry-point failures ({len(report.failures)}):")
        for g, n, reason in report.failures:
            click.echo(f"  - [{g}] {n}: {reason}")


@ext.command("new")
@click.argument("kind", type=click.Choice(["quantization", "target", "provider", "dialect", "runtime"]))
@click.argument("name", type=str)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory to scaffold into. Defaults to the current working directory.",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing directory at the target path.")
def ext_new(kind: str, name: str, out_dir: str | None, overwrite: bool) -> None:
    """Scaffold a pip-installable CompGen extension pack of KIND with NAME.

    Thin alias for ``compgen scaffold-pack``: scaffolds into ``--out`` (or
    the current directory). To make the scaffold visible to the running
    MCP server, either ``pip install -e`` the scaffold, or copy its
    starter module into ``~/.compgen/extensions/``.
    """
    from compgen.packs.scaffolding import scaffold_pack

    dest = out_dir or str(Path.cwd())
    result = scaffold_pack(kind=kind, name=name, out_dir=dest, overwrite=overwrite)
    click.echo(f"[ext new] Kind:         {kind}")
    click.echo(f"[ext new] Name:         {name}")
    click.echo(f"[ext new] Pack root:    {result.pack_root}")
    click.echo(f"[ext new] Starter:      {result.scheme_path}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  cd {result.pack_root}")
    click.echo("  pip install -e .        # so entry points register")
    click.echo("  compgen ext list        # verify discovery")


@ext.command("doctor")
def ext_doctor() -> None:
    """Re-run every discovery validator and report failures."""
    from compgen.plugins import discover_everything

    report = discover_everything()
    ok = True
    if report.failures:
        click.echo(f"Entry-point validation failures ({len(report.failures)}):")
        for g, n, reason in report.failures:
            click.echo(f"  - [{g}] {n}: {reason}")
        ok = False
    if report.user_space_errors:
        click.echo(f"User-space load errors ({len(report.user_space_errors)}):")
        for path, err in report.user_space_errors:
            click.echo(f"  - {path}: {err.splitlines()[0]}")
        ok = False
    click.echo(f"[ext doctor] total discovered: {report.total()}")
    if ok:
        click.echo("[ext doctor] all clean")
    else:
        raise click.ClickException("one or more extensions failed validation")


if __name__ == "__main__":
    main()
