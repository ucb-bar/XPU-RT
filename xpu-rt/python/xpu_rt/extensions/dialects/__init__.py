"""Custom MLIR dialects — extension point.

Add custom hardware-specific dialects by creating a subdirectory here
with a ``DialectSpec`` definition.

See ``_template.py`` for the starting point.
See ``compgen.extensions.xdsl_generate`` for the generation framework.
"""

__extension_point__ = True
__extension_type__ = "mlir_dialect"
__extension_protocol__ = "compgen.extensions.xdsl_generate.DialectSpec"
__extension_template__ = "compgen.extensions.dialects._template"
