"""Runtime execution adapters — extension point.

Add custom execution backends by placing a Python file in this directory
implementing the runtime adapter interface.

See ``_template.py`` for the starting point.
"""

__extension_point__ = True
__extension_type__ = "runtime_adapter"
__extension_protocol__ = "compgen.runtime.adapters._template.TemplateRuntimeAdapter"
__extension_template__ = "compgen.runtime.adapters._template"
