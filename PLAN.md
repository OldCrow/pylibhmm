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
Last reconciled against live GitHub state: 2026-08-23.
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
- Open issues: 4 as of 2026-07-24 (opened in the tooling-setup pass
  from its Known Gaps, none assigned a milestone yet):
  - #12 Wire ruff check and lint-cpp.sh into CI
  - #13 Run deferred ruff format pass across Python surface
  - #14 Triage B017 blind-exception test assertions, then enable ruff B rule
  - #16 Adopt mypy: annotate __init__.py wrapper surface
- Closed issues: 5 as of 2026-07-24 (#15 closed — pin-currency CI job
  implemented, see Cross-Repo Dependencies; fetch full list via
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
- Stale FetchContent pin risk: **closed 2026-07-24** (issue #15). This
  repo prefers a local `../libhmm` checkout when present, so the pin could
  silently drift on any machine that always has a fresh local libhmm
  alongside it. Now caught mechanically by the `pin-currency` canary job —
  see Cross-Repo Dependencies. Retained here as a record of why the check
  exists, not as an open gap.

## Cross-Repo Dependencies [OPEN]
Pins libhmm via `FetchContent` at the `GIT_TAG` in `CMakeLists.txt` —
**that line is the single source of truth and the version is deliberately
not restated in prose here or anywhere else.** Restated copies drifted
before (libhmm's own PLAN.md carried a wrong value and a fabricated
"lag"); the rule now is to read the tag from `CMakeLists.txt`, never from
documentation.

Currency is enforced mechanically, not by prose: the `pin-currency` job in
the monthly CI canary compares `GIT_TAG` against libhmm's newest release
tag and fails on mismatch (GitHub issue #15).

**Done 2026-08-22**: pin bumped to libhmm v4.4.0 (released 2026-08-19),
pylibhmm to 0.11.0 in the same change — **released 2026-08-22** (tag
v0.11.0, wheels + CI green, PyPI 0.11.0, GitHub release) — minor on the 0.10.0 precedent
(parent minor → binding minor; users observe the fixes). libhmm's v4.4.0
API additions (topology constraints, `fit_best_of_n()`, `sample()`,
`clone()`) are not yet bound — file as feature issues. libhmm's v4.4.1
correctness patch (#86/#87 memory safety first) is imminent; re-bump then.

**Done 2026-08-16 at the libhmm v4.3.0 bump**: the seven forced
`set(... FORCE)` option lines are deleted from the `FetchContent` branch, so
both dependency paths now converge on libhmm's `PROJECT_IS_TOP_LEVEL`
defaults. v4.3.0 retired the unprefixed spellings outright, so forcing them
had become a no-op as well as unnecessary.

Buildability of the pinned path is separately and continuously covered:
CI has no `../libhmm` sibling, so every CI run already exercises
`FetchContent`. Verified manually on macOS/AppleClang 2026-07-24 —
configure + build + 55-symbol import against the pin, clean. Note that
buildability and currency are different questions; CI covers the first on
every run, the second only monthly.

## Build-Stack Standardization (2026-07-23) [DERIVED]
Cross-repo effort tracked in the fleet standards repo
([record](https://github.com/OldCrow/standards/blob/main/records/BUILD-STANDARDIZATION-PLAN.md)).
Commits: `7e6db1d` (minimal CMakePresets.json, CMake minimum bumped to
3.25), `7a06b42` (local-source path adopts libhmm's `LIBHMM_*` option
names — coordinated with libhmm `8b0b6f7`; the local path forces
nothing, relying on libhmm's `PROJECT_IS_TOP_LEVEL` defaults). The
`FetchContent` path kept forcing the old unprefixed names off only until
the pin moved past libhmm's rename; that ended at the v4.3.0 bump
(`c48008c`, 2026-08-16 — see Cross-Repo Dependencies), so both paths now
converge on the same defaults. AGENTS.md's "Dependency resolution order"
section matches.

## Wheel Build Contract (2026-08-16) [DERIVED]
Settled at v0.10.0, after the shipped 0.9.2/0.9.3 declared
`requires-python >= 3.11` while building no cp311 wheel — metadata admitted
users the wheel set did not serve, so pip handed them the sdist and a local
libhmm compile. Three rules, all of which failed silently before:

- **The interpreter set is an ALLOWLIST** (`CIBW_BUILD`), never a denylist. A
  denylist cannot express "nothing ships untested": it fails open for every
  interpreter upstream adds, and for prefixes it does not literally match
  (`cp314-*` misses free-threaded `cp314t-`). `musllinux` stays under
  `CIBW_SKIP` — an ABI axis orthogonal to the interpreter set, applied after
  `CIBW_BUILD`.
- **The allowlist is DEFINED as "what ci.yml's matrix covers."** Adding an
  interpreter to one without the other makes the rule decoration. This is why
  ci.yml gained free-threaded 3.14t; pylibstats still ships `cp314t-*` with no
  3.14t row and has the gap this repo closed.
- **`requires-python` must match the built set.** Narrowing the wheels without
  narrowing the metadata is the 0.9.2 defect, and it is invisible in CI.

`wheel.py-api = "cp312"` and `${SKBUILD_SABI_COMPONENT}` on
`find_package(Python)` are a PAIR — the first only TAGS a wheel abi3, the
second is what actually builds one. Setting the first alone yields an
abi3-tagged, version-locked binary that installs on 3.13+ and then fails to
import, and **CI cannot detect it**, because cibuildwheel tests each wheel on
the interpreter that built it — the one version where the broken form works.
Check it directly instead: the module must be `_core.pyd`/`_core.so` importing
`python3.dll`/`libpython3.so`, not a version-stamped name against
`python312.*`. Keeping cp313/cp314 in the allowlist (they emit no extra wheel)
is what makes cibuildwheel install the abi3 wheel on interpreters it was not
built with, which is the only check that closes the class.

cibuildwheel is pinned, not floating: unpinned, the interpreter set it knows
about grows on upstream's schedule rather than ours.

Published as a fleet standard 2026-08-16:
[CI House Style §9](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md#9-wheel-builds-pylibhmm-pylibstats),
merged with pylibstats' twin incidents. AGENTS.md's CI/Validation section
carries the day-to-day summary; this section stays as the incident record
behind the rules.

## Next Steps
- #12 Decide when to wire `ruff check` and `scripts/lint-cpp.sh` into CI.
- #13 Run the deferred `ruff format` pass as its own reviewable change.
- #14 Triage the 5 `B017` blind-exception test assertions, then enable `B`.
- #16 Annotate `__init__.py` and adopt mypy.
