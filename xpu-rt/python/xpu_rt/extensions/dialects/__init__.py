"""Custom MLIR dialects — extension point.

Add custom hardware-specific dialects by creating a subdirectory here
with a ``DialectSpec`` definition.

See ``_template.py`` for the starting point.
See ``xpu_rt.extensions.xdsl_generate`` for the generation framework.
"""

__extension_point__ = True
__extension_type__ = "mlir_dialect"
__extension_protocol__ = "xpu_rt.extensions.xdsl_generate.DialectSpec"
__extension_template__ = "xpu_rt.extensions.dialects._template"
