""" — real published models wired end-to-end through ``compile_with_llm``.

Each module in this package loads a real HuggingFace checkpoint (no
miniatures, no toy graphs) and drives the full agentic compile loop,
proving that what we ship in tests is what we'd ship to a user.
"""
