"""SMT context management and dialect registration.

Creates and configures ``MLContext`` instances with the SMT dialect
from upstream xDSL, plus any CompGen-specific semantic extensions.
"""

from __future__ import annotations

from xdsl.context import Context as MLContext
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.memref import MemRef
from xdsl.dialects.smt import SMT


class SMTContextFactory:
    """Create and configure MLContext for SMT-based verification."""

    @staticmethod
    def create() -> MLContext:
        """Create a context with all dialects needed for verification.

        Registers: Builtin, Func, Arith, MemRef, Linalg, SMT.
        """
        ctx = MLContext()
        ctx.allow_unregistered = True
        ctx.load_dialect(Builtin)
        ctx.load_dialect(Func)
        ctx.load_dialect(Arith)
        ctx.load_dialect(MemRef)
        ctx.load_dialect(Linalg)
        ctx.load_dialect(SMT)
        return ctx

    @staticmethod
    def create_with_compgen_dialects() -> MLContext:
        """Create a context that also includes CompGen's custom dialects.

        Adds Tile and Accel dialect registrations on top of the base context.
        """
        ctx = SMTContextFactory.create()
        try:
            from compgen.ir.tile.dialect import TileDialect
            ctx.load_dialect(TileDialect)
        except (ImportError, NotImplementedError):
            pass
        return ctx


__all__ = ["SMTContextFactory"]
