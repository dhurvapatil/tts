# Issues (PRD-002: Hardening, Shape-Range Warmup, and Pre-Generate Buffer)

Vertical-slice issues for [PRD-002](../PRD-002-warmup-and-buffering.md).
All carry the `ready-for-agent` triage label. They are listed in
dependency order (blockers first); all five are independent at the
code level and can be picked up in any order.

| ID | Title | Blocked by | Status |
|---|---|---|---|
| [002-001](./002-001-nvidia-ml-py-migration.md) | Migrate pynvml → nvidia-ml-py | none | shipped |
| [002-002](./002-002-summary-line-generation-rtf.md) | Fix WAV summary line to use generation-RTF | none | shipped |
| [002-003](./002-003-shape-range-warmup.md) | Shape-range warmup to eliminate first-chunk cold-compile spike | none | shipped |
| [002-004](./002-004-vram-plateau-regression.md) | VRAM plateau regression in bench_ttfw | none | shipped |
| [002-005](./002-005-look-ahead-buffer.md) | Look-ahead buffer: pre-generate + buffer health metrics | none | shipped |
