"""Template — copy this into ``targets/{class}/{vendor}/{arch}/``
to start a new in-tree target package (Wave 1.17 cookbook).

The full annotated walkthrough lives at
``docs/architecture/add-a-new-target.md``. This package is the
50-LoC starting point: rename the four files, fill in the
specifics, register, ship.

Validation that you've filled it in correctly is automatic —
``compgen.targets.registry`` calls into your adapters when a user
chooses your target_id, and the universal compile path rejects
typed errors if any Protocol method is missing or returns the
wrong shape.

This file is a no-op when imported into the registry; it's a
template, not a live target. Copy + rename to use.
"""

from __future__ import annotations

# When you copy this template, replace the imports with your own
# adapters — your package's ``probe.py``, ``body_emitter.py``,
# ``runtime.py``, ``cost.py``.
from compgen.targets._template.body_emitter import TemplateBodyEmitter
from compgen.targets._template.cost import TemplateCostModel
from compgen.targets._template.probe import TemplateProbe
from compgen.targets._template.runtime import TemplateRuntime


def _register_template() -> None:
    """When this template is copied into a real target package,
    rename to ``_register_<your_arch>`` and call
    :func:`compgen.targets.registry.register_target` with concrete
    metadata."""
    # Intentionally not registered — this is a template, not a live
    # target. Real packages call ``register_target`` here.
    _ = TemplateProbe, TemplateBodyEmitter, TemplateRuntime, TemplateCostModel


# Don't actually register the template; it's not a live target.
# The placeholder _register_template() exists as a docs anchor so
# new-target authors know what shape to write.
