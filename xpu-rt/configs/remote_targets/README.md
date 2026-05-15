# Remote-target descriptors

YAML files in this directory describe remote hardware the user has
access to. Each descriptor is loaded by
:func:`compgen.runtime.remote_target.load_remote_target_config` and
fed to :func:`compgen.runtime.remote_target.build_runner` to pick a
transport (SSH today; Modal / k8s in the future).

Schema:

```yaml
target_id: tpu_v5e_pod_1            # required — must match a TargetCard target_id
transport: ssh                       # "ssh" | "modal" | "k8s"
host: tpu-host.example.com           # required for ssh
user: compgen                        # optional
workdir: /tmp/compgen_remote         # optional, default /tmp/compgen_remote
toolchain_probe_cmd: "python -c 'import jax; print(jax.__version__)'"
build_cmd_template: "python {source} --build"
run_cmd_template: "python {source} --run"
timeout_s: 1800
extras: {}
```

When the descriptor file is missing OR `host` is empty, the runner's
``probe()`` returns ``status=blocked`` with the typed reason, and the
provider records a typed ``blocked_proof.json`` instead of a real
``remote_receipt.json``.

The five canonical HW-gated providers each have a stub descriptor
here that you fill in once you have the real hostname. Until then,
each stub probes ``unreachable`` and the audit records the gap.
