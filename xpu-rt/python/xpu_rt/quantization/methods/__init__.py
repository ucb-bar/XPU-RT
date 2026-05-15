"""Quantization methods — extension point.

Add custom quantization schemes by placing a Python file in this directory
implementing a torchAO ``AOBaseConfig`` subclass with a registered handler.

See ``_template.py`` for the starting point.
"""

__extension_point__ = True
__extension_type__ = "quantization_method"
__extension_protocol__ = "torchao.core.config.AOBaseConfig"
__extension_template__ = "compgen.quantization.methods._template"
