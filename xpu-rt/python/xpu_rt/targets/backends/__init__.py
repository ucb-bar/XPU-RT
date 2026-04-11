"""Hardware target backends — extension point.

Add custom target backends by creating a subdirectory here implementing
the ``TargetBackendProtocol`` from ``compgen.targets.backend``.

See ``_template.py`` for the starting point.
See ``docs/architecture/target-backend-model.md`` for the full architecture.
"""

__extension_point__ = True
__extension_type__ = "target_backend"
__extension_protocol__ = "compgen.targets.backend.TargetBackendProtocol"
__extension_template__ = "compgen.targets.backends._template"
