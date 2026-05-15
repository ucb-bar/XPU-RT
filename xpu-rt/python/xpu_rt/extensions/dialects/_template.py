"""Template: Custom MLIR Dialect

Define hardware-specific operations as an xDSL dialect. CompGen's
``xdsl_generate.py`` can auto-generate the Python dialect code from
a ``DialectSpec``.

See ``compgen.extensions.xdsl_generate`` for the generation framework.
See Hexagon's HexKL dialect as a reference for hardware-specific microops.
See ``docs/architecture/target-backend-model.md`` Section 4.

Steps:
    1. Create a directory: ``dialects/my_dialect/``
    2. Copy this file: ``cp _template.py dialects/my_dialect/__init__.py``
    3. Define your ``DialectSpec`` with operations and attributes
    4. Generate the xDSL code: ``generate_xdsl_dialect(my_spec, output_dir)``
    5. Register with CompGen's IR infrastructure

Example: Hexagon's HexKL defines:
    - hexkl.matmul (tiled matrix multiply)
    - hexkl.micro_hmx_setup_acc_read (setup accumulator)
    - hexkl.micro_hmx_mm (hardware matrix multiply)
    - hexkl.micro_hmx_copy_submatrix (DMA tile copy)
"""

from __future__ import annotations

from compgen.extensions.xdsl_generate import DialectOpSpec, DialectSpec

# Define your dialect specification
template_dialect = DialectSpec(
    name="my_accel",
    ops=[
        DialectOpSpec(
            name="tiled_matmul",
            description="Tiled matrix multiply on the hardware matrix unit",
            operands=["lhs", "rhs"],
            results=["result"],
            attrs={"tile_m": "int", "tile_n": "int", "tile_k": "int"},
        ),
        DialectOpSpec(
            name="dma_load",
            description="Async DMA transfer from main memory to scratchpad",
            operands=["src"],
            results=["dst"],
            attrs={"channel": "int", "stride": "int"},
        ),
        DialectOpSpec(
            name="dma_store",
            description="Async DMA transfer from scratchpad to main memory",
            operands=["src"],
            results=["dst"],
            attrs={"channel": "int"},
        ),
        DialectOpSpec(
            name="vector_add",
            description="BF16 vector addition on the vector processing unit",
            operands=["lhs", "rhs"],
            results=["result"],
            attrs={},
        ),
    ],
)

# To generate the xDSL Python dialect:
#   from compgen.extensions.xdsl_generate import generate_xdsl_dialect
#   generate_xdsl_dialect(template_dialect, output_dir="python/compgen/ir/my_accel/")
