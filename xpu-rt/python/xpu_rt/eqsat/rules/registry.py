"""Rule registry for equality saturation.

Discovers, loads, and filters rewrite rules by category and target.
Supports both built-in rules and dynamically added LLM-generated rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from compgen.eqsat.rules.python_rules import EqSatRewriteRule

log = structlog.get_logger()


@dataclass
class RuleRegistry:
    """Registry of available eqsat rewrite rules.

    Organizes rules by category (algebraic, layout, fusion, target-specific,
    llm-generated) and provides filtering.
    """

    _rules: dict[str, list[EqSatRewriteRule]] = field(default_factory=dict)

    def register(self, category: str, rule: EqSatRewriteRule) -> None:
        """Register a rule under a category."""
        if category not in self._rules:
            self._rules[category] = []
        # Avoid duplicates by name
        existing_names = {r.name for r in self._rules[category]}
        if rule.name not in existing_names:
            self._rules[category].append(rule)
            log.debug("eqsat.rule_registered", category=category, rule=rule.name)

    def get_rules(self, categories: tuple[str, ...] | None = None) -> list[EqSatRewriteRule]:
        """Get rules filtered by categories.

        Args:
            categories: Tuple of category names. If None, returns all rules.

        Returns:
            List of matching rules.
        """
        if categories is None:
            return [rule for rules in self._rules.values() for rule in rules]

        result: list[EqSatRewriteRule] = []
        for cat in categories:
            result.extend(self._rules.get(cat, []))
        return result

    def categories(self) -> list[str]:
        """List all registered categories."""
        return list(self._rules.keys())

    def count(self, category: str | None = None) -> int:
        """Count rules, optionally filtered by category."""
        if category is None:
            return sum(len(rules) for rules in self._rules.values())
        return len(self._rules.get(category, []))

    def remove(self, rule_name: str) -> bool:
        """Remove a rule by name from all categories. Returns True if found."""
        found = False
        for cat in self._rules:
            before = len(self._rules[cat])
            self._rules[cat] = [r for r in self._rules[cat] if r.name != rule_name]
            if len(self._rules[cat]) < before:
                found = True
        return found


def create_default_registry() -> RuleRegistry:
    """Create a registry with all built-in rules."""
    from compgen.eqsat.rules.algebraic import get_default_algebraic_rules
    from compgen.eqsat.rules.fusion import get_default_fusion_rules
    from compgen.eqsat.rules.layout import get_default_layout_rules

    registry = RuleRegistry()

    for rule in get_default_algebraic_rules():
        registry.register("algebraic", rule)

    for rule in get_default_layout_rules():
        registry.register("layout", rule)

    for rule in get_default_fusion_rules():
        registry.register("fusion", rule)

    return registry
