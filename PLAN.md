# pylibhmm — Plan / Status

## Decided [DERIVED]
- Bindings via nanobind + scikit-build-core.
- Prefers a local `../libhmm` checkout when present; otherwise
  `FetchContent`-fetches the pinned release tag.
- NumPy ⇔ libhmm conversion is copy-based by default (owned libhmm value
  types on the way in, capsule-owned heap buffers on the way out); the one
  documented exception is `obs_matrix_views()`'s transient zero-copy spans
  into the input array, scoped to a single bound call — see AGENTS.md
  Architecture.
- `__init__.pyi` / `_core.pyi` are hand-written, not tool-generated.
- Python tooling: ruff adopted (`E`/`F`/`I`/`UP`), config in
  `pyproject.toml`. `B` (bugbear) deliberately deferred — see Known Gaps.
  mypy not adopted — see Known Gaps.
- C++ binding tooling: `scripts/lint-cpp.sh` — cppcheck with its own
  invocation (not a copy of libhmm's), requiring `--language=c++` for
  `_common.h` and a path-based suppression for libhmm's own headers.
  Verified clean as of 2026-07-14.

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-07-14.
- GitHub is the collaborator-facing source for issues and milestones; this
  PLAN.md is the agent-facing durable project state. Keep both in sync.
- When creating, closing, reopening, retitling, or moving a GitHub issue or
  milestone, update this section in the same change set or note why it could
  not be updated.
- Reconcile this section against live GitHub state when either is true:
  (a) the task at hand involves reading the backlog to decide what to work
  on next, or creating/closing/retitling/moving an issue or milestone, or
  (b) more than 7 days have passed since the "Last reconciled" date above.
  Skip the check for tasks that don't touch the backlog or this file at
  all — a per-session or per-task refresh regardless of relevance is
  wasted effort in one direction and a rubber stamp in the other. Update
  the "Last reconciled" date whenever this section is actually re-checked,
  whether or not anything had drifted.
- Convention: open (actionable) milestones/issues are fully itemized here;
  closed/historical ones are summarized as counts only.

## GitHub Milestones [DERIVED]
- None currently exist in this repository (checked 2026-07-14).

## GitHub Issues Without Milestone [DERIVED]
- Open issues: 5 as of 2026-07-14 (all opened this session from the
  tooling-setup pass's Known Gaps, none assigned a milestone yet):
  - #12 Wire ruff check and lint-cpp.sh into CI
  - #13 Run deferred ruff format pass across Python surface
  - #14 Triage B017 blind-exception test assertions, then enable ruff B rule
  - #15 Periodic check for stale libhmm FetchContent pin
  - #16 Adopt mypy: annotate __init__.py wrapper surface
- Closed issues: 4 as of 2026-07-14 (fetch via
  `gh issue list --state closed --json number,title,milestone -q
  '.[] | select(.milestone == null)'` if ever needed).

## In Progress [OPEN]
- (none currently tracked — populate as work starts)

## Known Gaps [OPEN]
- `ruff format` would reformat 13 files (nearly the whole Python surface,
  including `__init__.py`) under the new config — not applied in this pass
  since it's a large, purely cosmetic diff that deserves its own visible
  change rather than being bundled silently into a tooling-setup pass. Run
  `ruff format src/pylibhmm tests examples` as a deliberate follow-up.
  Tracked as GitHub issue #13.
- `B` (flake8-bugbear) ruff rules are deferred: 5 `pytest.raises(Exception)`
  findings in `tests/` need a real decision on the exact exception type
  each binding raises (nanobind `type_error`? `ValueError`? `RuntimeError`?)
  before they can be tightened — not a mechanical fix. Tracked as GitHub
  issue #14.
- mypy is not adopted: `__init__.py`'s wrapper methods (`set_pi`,
  `set_trans`, calculator/trainer `__init__`s, etc.) are only partially
  annotated (many params like `pi`, `trans`, `sequences`, `observations`
  have no type hints). Adopting mypy needs an annotation pass across
  `__init__.py` first, not just an empty config. Tracked as GitHub issue
  #16.
- Neither `ruff check` nor `scripts/lint-cpp.sh` are wired into CI yet
  (`.github/workflows/ci.yml` only runs pytest + ASan) — no decision
  recorded on when to add them. Tracked as GitHub issue #12.
- Stale FetchContent pin risk: this repo prefers a local `../libhmm`
  checkout when present, meaning the FetchContent pin can silently drift
  out of date on any machine that always has a fresh local libhmm
  alongside it — the fetched path never gets exercised there to catch it.
  No periodic check currently exists for this. Tracked as GitHub issue #15.

## Cross-Repo Dependencies [OPEN]
Pins libhmm via FetchContent at `v4.2.4` (`CMakeLists.txt`). Verified
2026-07-14: this is libhmm's actual current release tag (libhmm `main` is
1 docs-only commit ahead of `v4.2.4`, not a new release) — the pin is
current, not stale. When libhmm cuts a new release, this pin must be
bumped deliberately — check libhmm's PLAN.md/AGENTS.md for its current
version before assuming this one is still current. Verify sync explicitly
on any machine that builds from the FetchContent path alone (no local
`../libhmm` checkout), rather than relying on the local-preference path,
which won't surface a stale pin. See GitHub issue #15 for the periodic
check to catch drift.

## Next Steps
- #12 Decide when to wire `ruff check` and `scripts/lint-cpp.sh` into CI.
- #13 Run the deferred `ruff format` pass as its own reviewable change.
- #14 Triage the 5 `B017` blind-exception test assertions, then enable `B`.
- #15 Implement a periodic check for the libhmm FetchContent pin.
- #16 Annotate `__init__.py` and adopt mypy.
