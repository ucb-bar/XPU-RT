# Target Packages

CompGen's target story is about generating a target enablement package, not just compiling one model once.

## What a Target Package Represents

A target package is the collection of information and artifacts CompGen needs in order to support a hardware target:

- hardware description
- capabilities and constraints
- lowering and kernel strategy hooks
- runtime integration details
- verification surface

## Current Status

The concept is important and the codebase already has target-generation machinery, but the public `compgen scaffold-target` CLI command is still a stub. Treat target packages as the direction of the system and use the Python API or targetgen modules for current experimentation.

## Related Pages

- [Bring Up a Target](../guides/bring-up-a-target.md)
- [Python API](python-api.md)
- [Target Profile Schema](target-profile-schema.md)
