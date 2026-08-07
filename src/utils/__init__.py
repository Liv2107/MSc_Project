"""Small cross-cutting utilities for configuration, seeds, and run logging.

Utilities should remain boring and explicit. Dataset or experiment policy hidden in a
generic helper becomes difficult to discover six months later.

IMPLEMENTATION CHECKLIST
------------------------
[ ] Validate and persist resolved configuration.
[ ] Seed every used library and DataLoader worker.
[ ] Record environment and structured run logs.
[ ] Keep research-domain decisions in their owning modules.
"""
