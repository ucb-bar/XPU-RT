"""Claude-Code-style interactive question widget.

A small prompt_toolkit ``Application`` that renders a numbered list of
options with ``[✔] / [ ]`` markers, "Type something" and "Skip
interview" trailing rows, and the usual ``↑ / ↓ / space / enter`` key
bindings. Designed to feel like the ``AskUserQuestion`` widget the
Claude Code CLI surfaces, so the XPU-RT CLI's startup interview reads
naturally to users who already know that shape.

Modes:

* ``"radio"`` — pick one option (Enter submits).
* ``"checkbox"`` — toggle multiple options (Space toggles; Enter submits).
* ``"freeform"`` — skip the list and just ask for a string.

Falls back to a non-interactive prompt-style flow when stdin is not a
TTY (CI, piped invocations); never raises just because the terminal
can't host the rich widget.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Sequence
from typing import Literal

QuestionKind = Literal["radio", "checkbox", "freeform"]


@dataclasses.dataclass(frozen=True)
class QuestionOption:
    """One option shown in the widget."""

    label: str
    value: str
    hint: str = ""


@dataclasses.dataclass(frozen=True)
class QuestionAnswer:
    """The user's answer.

    Exactly one of ``selected_values`` (radio/checkbox) or
    ``freeform_text`` (the user picked "Type something") will be
    populated. ``skipped`` is True when the user picked "Skip" or
    pressed Esc.
    """

    selected_values: tuple[str, ...] = ()
    freeform_text: str | None = None
    skipped: bool = False

    @property
    def first(self) -> str | None:
        if self.selected_values:
            return self.selected_values[0]
        return None


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and not os.environ.get(
        "XPURT_NONINTERACTIVE"
    )


def _fallback(
    title: str,
    options: Sequence[QuestionOption],
    *,
    kind: QuestionKind,
    default_value: str | None,
) -> QuestionAnswer:
    """Non-TTY path: print the question, accept stdin lines."""
    print(f"\n{title}", file=sys.stderr)
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt.value == default_value else ""
        print(f"  {i}. {opt.label}{marker}", file=sys.stderr)
        if opt.hint:
            print(f"     {opt.hint}", file=sys.stderr)
    if kind == "freeform":
        try:
            line = input("> ").strip()
        except EOFError:
            return QuestionAnswer(skipped=True)
        return QuestionAnswer(freeform_text=line or None)
    if not options:
        return QuestionAnswer(skipped=True)
    if default_value is not None:
        return QuestionAnswer(selected_values=(default_value,))
    return QuestionAnswer(selected_values=(options[0].value,))


def ask_user_question(
    title: str,
    options: Sequence[QuestionOption] = (),
    *,
    kind: QuestionKind = "radio",
    allow_freeform: bool = True,
    allow_skip: bool = True,
    default_value: str | None = None,
    freeform_placeholder: str = "Type something",
    skip_label: str = "Skip interview",
) -> QuestionAnswer:
    """Ask the user a question; return their answer.

    Mirrors Claude Code's AskUserQuestion in look and key bindings.
    Falls back to a plain ``print + input`` flow on non-TTY stdin so
    the CLI never deadlocks under CI.
    """
    if not _is_interactive():
        return _fallback(
            title, options, kind=kind, default_value=default_value,
        )
    try:
        return _interactive(
            title,
            list(options),
            kind=kind,
            allow_freeform=allow_freeform,
            allow_skip=allow_skip,
            default_value=default_value,
            freeform_placeholder=freeform_placeholder,
            skip_label=skip_label,
        )
    except Exception:  # noqa: BLE001 - never let UI bring down the CLI
        return _fallback(
            title, options, kind=kind, default_value=default_value,
        )


def _interactive(
    title: str,
    options: list[QuestionOption],
    *,
    kind: QuestionKind,
    allow_freeform: bool,
    allow_skip: bool,
    default_value: str | None,
    freeform_placeholder: str,
    skip_label: str,
) -> QuestionAnswer:
    # Lazy import — only pay this when actually rendering.
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.formatted_text import to_formatted_text
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.styles import Style

    n_real = len(options)
    n_extra = (1 if allow_freeform else 0) + (1 if allow_skip else 0)
    n_rows = n_real + n_extra
    if n_rows == 0:
        return QuestionAnswer(skipped=True)

    # State
    state: dict = {
        "cursor": 0,
        "checked": [False] * n_real,
        "submitting": False,
        "freeform": False,
    }
    if default_value is not None:
        for i, opt in enumerate(options):
            if opt.value == default_value:
                state["cursor"] = i
                if kind == "checkbox":
                    state["checked"][i] = True
                break
    elif kind == "radio" and n_real:
        # Pre-select the first real row by default.
        pass

    freeform_buffer = Buffer(multiline=False)

    def _row_for_cursor(c: int) -> tuple[str, object]:
        # Returns ("real" | "freeform" | "skip", payload)
        if c < n_real:
            return "real", options[c]
        idx = c - n_real
        if allow_freeform and idx == 0:
            return "freeform", None
        return "skip", None

    def render():
        out: list[tuple[str, str]] = []
        out.append(("class:title", f"{title}\n"))
        out.append(("class:dim", "  ↑/↓ move · space toggle · enter submit · t type · s skip · esc cancel\n\n"))
        for i, opt in enumerate(options):
            cursor = "❯ " if state["cursor"] == i and not state["freeform"] else "  "
            if kind == "checkbox":
                marker = "[✔] " if state["checked"][i] else "[ ] "
            else:
                marker = "● " if state["cursor"] == i and not state["freeform"] else "○ "
            cls = "class:active" if state["cursor"] == i and not state["freeform"] else "class:row"
            out.append((cls, f"{cursor}{i + 1}. {marker}{opt.label}\n"))
            if opt.hint:
                out.append(("class:hint", f"     {opt.hint}\n"))
        extra_index = n_real
        if allow_freeform:
            cursor = "❯ " if state["cursor"] == extra_index else "  "
            cls = "class:active" if state["cursor"] == extra_index else "class:row"
            out.append((cls, f"{cursor}{extra_index + 1}. {freeform_placeholder}\n"))
            extra_index += 1
        if allow_skip:
            cursor = "❯ " if state["cursor"] == extra_index else "  "
            cls = "class:active" if state["cursor"] == extra_index else "class:row"
            out.append((cls, f"{cursor}{extra_index + 1}. {skip_label}\n"))
        return to_formatted_text(out)

    def freeform_visible() -> bool:
        return state["freeform"]

    text_control = FormattedTextControl(render, focusable=False)
    list_window = Window(content=text_control, always_hide_cursor=True)
    input_window = Window(
        content=BufferControl(buffer=freeform_buffer),
        height=1,
    )

    body = HSplit([
        list_window,
        Window(
            content=FormattedTextControl(
                lambda: to_formatted_text([("class:prompt", "› ")]) if freeform_visible() else "",
            ),
            height=lambda: 1 if freeform_visible() else 0,
        ),
        input_window,
    ])

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        if state["freeform"]:
            return
        state["cursor"] = (state["cursor"] - 1) % n_rows

    @kb.add("down")
    def _down(event):
        if state["freeform"]:
            return
        state["cursor"] = (state["cursor"] + 1) % n_rows

    @kb.add("space")
    def _space(event):
        if state["freeform"]:
            freeform_buffer.insert_text(" ")
            return
        kind_, payload = _row_for_cursor(state["cursor"])
        if kind_ == "real" and kind == "checkbox":
            state["checked"][state["cursor"]] = not state["checked"][state["cursor"]]

    @kb.add("t")
    def _t(event):
        if state["freeform"]:
            freeform_buffer.insert_text("t")
            return
        state["freeform"] = True
        # Move the cursor onto the freeform row.
        if allow_freeform:
            state["cursor"] = n_real

    @kb.add("s")
    def _s(event):
        if state["freeform"]:
            freeform_buffer.insert_text("s")
            return
        if allow_skip:
            state["cursor"] = n_real + (1 if allow_freeform else 0)
            event.app.exit(result="skip")

    @kb.add("escape")
    def _esc(event):
        event.app.exit(result="skip")

    @kb.add("c-c")
    def _ctrl_c(event):
        event.app.exit(result="skip")

    @kb.add("enter")
    def _enter(event):
        kind_, payload = _row_for_cursor(state["cursor"])
        if state["freeform"] or kind_ == "freeform":
            text = freeform_buffer.text.strip()
            event.app.exit(result=("freeform", text))
            return
        if kind_ == "skip":
            event.app.exit(result="skip")
            return
        # real row
        if kind == "checkbox":
            event.app.exit(result="checkbox")
        else:
            event.app.exit(result=("radio", payload))

    style = Style.from_dict({
        "title": "bold",
        "active": "fg:ansicyan bold",
        "row": "",
        "hint": "fg:ansibrightblack",
        "dim": "fg:ansibrightblack",
        "prompt": "fg:ansicyan bold",
    })

    app: Application = Application(
        layout=Layout(body),
        key_bindings=kb,
        full_screen=False,
        style=style,
        mouse_support=False,
    )
    result = app.run()

    if result == "skip":
        return QuestionAnswer(skipped=True)
    if isinstance(result, tuple) and result[0] == "freeform":
        text = result[1]
        return QuestionAnswer(freeform_text=text or None)
    if isinstance(result, tuple) and result[0] == "radio":
        opt: QuestionOption = result[1]
        return QuestionAnswer(selected_values=(opt.value,))
    if result == "checkbox":
        values = tuple(
            opt.value for opt, on in zip(options, state["checked"]) if on
        )
        return QuestionAnswer(selected_values=values)
    return QuestionAnswer(skipped=True)
