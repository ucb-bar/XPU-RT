"""Template Runtime — fill in for your target's JIT + dispatch."""

from __future__ import annotations

from typing import Any


class TemplateRuntime:
    """Replace with ``YourArchRuntime``."""

    def compile_source(
        self,
        *,
        source: str,
        kernel_name: str = "",
        symbol_name: str = "",
        arch: str = "",
        extra_options: tuple[str, ...] = (),
        extra_include_paths: tuple[str, ...] = (),
        compile_flags: tuple[str, ...] = (),
    ) -> Any:
        """JIT compile ``source`` and return a vendor-defined module
        handle. The signature accepts both GPU-shaped (kernel_name +
        arch + extra_include_paths) and CPU-shaped (symbol_name +
        compile_flags) parameters — fill in what your target
        actually consumes."""
        raise NotImplementedError("Template — fill this in for your arch")

    def launch(
        self,
        *,
        module_handle: Any,
        grid_dim: tuple[int, int, int] = (1, 1, 1),
        block_dim: tuple[int, int, int] = (1, 1, 1),
        cluster_dim: tuple[int, int, int] | None = None,
        shared_mem_bytes: int = 0,
        kernel_params: Any = None,
        cooperative: bool = False,
    ) -> None:
        """GPU-side launch — kept here for Protocol parity. CPU
        targets implement ``dispatch`` instead and leave this
        method as ``raise NotImplementedError``."""
        raise NotImplementedError("Template — fill this in for your arch")

    def dispatch(
        self,
        *,
        library_handle: Any,
        kernel_params: Any,
    ) -> None:
        """CPU-side dispatch — synchronous call into the loaded
        symbol. GPU targets leave this raising and use ``launch``
        instead."""
        raise NotImplementedError("Template — fill this in for your arch")

    def synchronize(self) -> None:
        """Block until queued work completes. CPU is naturally
        synchronous; GPU calls into the vendor's stream sync."""
        return None
