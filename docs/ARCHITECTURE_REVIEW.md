# Architecture Review: Juicer

Date: 2026-08-24
Scope: whole-repo senior-engineer pass, focused on duplication, dead code, and
operational risk in the Spark/scikit-learn execution paths. Keras and COMPSs
are being removed on a separate branch and are intentionally out of scope
here — they are not described below as active backends.

This document has three parts: an architecture summary, a ranked list of
problem areas, and refactoring strategies for the areas *not* addressed in
this PR. The last section of the PR itself (not this file) covers the 4 demo
refactors that *were* implemented, with pointers to their diffs.

## Architecture summary

### Two entry points, one shared transpile core

- **CLI path** (`juicer/app.py`): fetches workflow JSON from Tahiti, builds a
  `networkx` task/flow graph via `Workflow` (`juicer/workflow/workflow.py`),
  picks a `Transpiler` subclass by platform slug, and calls
  `Transpiler.generate_code()` (`juicer/transpiler.py:296`), which
  instantiates one `Operation` subclass per task and renders Jinja2
  templates into a `.py` source file. This path only *writes* code; it does
  not execute it.
- **Server/minion path** (production, driven by the Stand UI):
  `juicer/runner/server.py` (`JuicerServer`) blocks on a Redis queue
  (`BLPOP` on `queue_start`) and spawns or forwards to a **minion** — one OS
  subprocess (or k8s Job) per `app_id`. `juicer/runner/minion.py` selects a
  platform-specific `Minion` (`SparkMinion` in `juicer/spark/spark_minion.py`
  is the only one still active post-Keras/COMPSs-removal). A minion runs two
  daemon loops: `execute()` (single-worker `ThreadPoolExecutor` — jobs for
  one app run strictly serially) and `ping()` (Redis heartbeat, refreshed
  every 5s with a 10s TTL — see `juicer/runner/minion_base.py:104-123`).
  `perform_execute()` re-transpiles the workflow, `importlib`-reloads the
  generated module, and calls its `main()`; results accumulate in
  `self._state` until `terminate()` clears them.

### Backend layout

Every backend is `transpiler.py` + `*_operation.py` (one `Operation`
subclass per task type) + `*_minion.py`, rooted in
`juicer/operation.py::Operation` (parameter bookkeeping, `must_be_executed()`,
`to_deploy_format()`, a `render_template()` Jinja wrapper — no shared
param-parsing helper, so `__init__`s duplicate parsing logic across classes
and across backends).

- **`juicer/spark/`** — the primary/production backend (PySpark).
- **`juicer/scikit_learn/`** — pandas-based, with two alternate
  dataframe-engine variants, `juicer/scikit_learn/polars/` and
  `juicer/scikit_learn/duckdb/`, that subclass the base scikit-learn
  operation classes and are meant to only override `generate_code()`
  (Demo 2 fixes one place where a duckdb class didn't actually override
  anything).
- **`juicer/meta/`** — not a peer execution backend; a no-code/template layer
  (`MetaTranspiler.SUPPORTED_TARGET_PLATFORMS = {'spark': 1, 'scikit-learn': 4}`)
  generating Spark or scikit-learn code for a template-driven UI, importing
  `juicer.spark.*` operation classes directly.

### Hidden global-config-singleton / import-time coupling

- `juicer/runner/configuration.py` holds a module-level `__CONFIG__`, set
  independently from three places (`app.py`, `server.py`,
  `spark_minion.py`) and implicitly read by every `Operation.__init__`. An
  `Operation` instantiated before `set_config()` runs gets `None`.
- Several entry-point modules call `logging.config.fileConfig()`,
  `matplotlib.use('Agg', force=True)`, or mutate `sys.path` at module scope
  — importing them (not just running them) reconfigures process-global
  state, which makes these modules unsafe to import incidentally (e.g. from
  a test file that just wants one class).
- i18n resolves through `gettext`'s builtin `_()`, installed via
  `translation(...).install()` in each minion entry point separately
  (`juicer/app.py:110`, `juicer/runner/server.py:544`, and per-backend
  minions). Code that calls bare `_(...)` only works once some entry point
  has already called `.install()`.

## Problem areas, ranked

| # | Area | Severity | Where |
|---|------|----------|-------|
| 1 | Serial job execution (single-worker `ThreadPoolExecutor`) + no HTTP timeouts anywhere in the service layer | **Critical** | `juicer/spark/spark_minion.py` executor setup; `juicer/service/{tahiti,limonero,stand,caipirinha}_service.py` (`requests.get/post` calls, none pass `timeout=`) |
| 2 | Dead retry path in `server.py` — a minion that dies mid-flight silently drops its pending queue instead of respawning | **Critical** | `juicer/runner/server.py:444` (`if pendings and False:`) |
| 3 | `ping()` has no exception handling around the Redis heartbeat call — a transient Redis blip kills the heartbeat loop forever while the minion process (and its resources) stays alive, producing a "ghost minion" the server no longer tracks | **High** | `juicer/runner/minion_base.py:118-123` |
| 4 | Copy-paste boilerplate across `ml_operation.py` / `etl_operation.py` — the `ctor_params` loop (Demo 4) is one instance of a wider pattern; most `Operation.__init__`s duplicate parameter-extraction/casting logic per backend with no shared helper | **High** | `juicer/spark/ml_operation.py` (many classifier/regressor classes beyond the 3 touched here); `juicer/spark/etl_operation.py`, `juicer/scikit_learn/etl_operation.py` |
| 5 | `dataframe_util.py` is a grab-bag module (sampling, CSV/JSON conversion, 5 JSONEncoder variants, attribute analysis, etc. all in one file) — the dead duplicate `emit_sample_sklearn` (Demo 1) was a symptom of this file being too large to eyeball for duplication | **Medium** | `juicer/util/dataframe_util.py` |
| 6 | Jinja `Environment` + gettext `Translations` rebuilt from disk on every `generate_code()` call; no Redis connection pooling (each caller opens its own `StrictRedis`); `cancel_job()` busy-polls in a tight loop; N+1-style sequential HTTP calls to Tahiti/Limonero per workflow | **Medium** | `juicer/transpiler.py:373-390`; `juicer/runner/server.py:99,420`; `juicer/spark/spark_minion.py:922-931` (`cancel_job`); `juicer/service/*.py` call sites |
| 7 | Dead `sc` reference in `spark_minion.py::terminate()` (Demo 3) | **Low** (was inert — always caught by a bare `except`) | `juicer/spark/spark_minion.py` (now fixed) |
| 8 | Fragile 3-mechanism i18n (gettext builtin install, Jinja's `jinja2.ext.i18n`, per-minion `translation(...).install()` calls) with no single source of truth for "is `_()` installed yet"; an unused custom exception hierarchy | **Low** | i18n: as described above; exceptions: `juicer/exceptions.py` (`JuicerException`, `InvalidGeneratedCode` — grep shows they're barely raised/caught outside their own definitions) |
| 9 | Zero CI test execution (pytest is commented out of `.travis.yml`) despite a real test suite existing; coverage is uneven (heavy on scikit-learn ETL, thin on Spark/minion/server code, several Spark minion tests are `@pytest.mark.skip(reason="Not working")`) | **Structural multiplier** | `.travis.yml`; `tests/` (e.g. `tests/spark/test_spark_minion.py` has 4 skipped tests) |
| 10 | `requirements.txt` / `requirements-3.7.txt` version-support duplication (two manually-kept-in-sync dependency lists) | **Low** | `requirements.txt`, `requirements-3.7.txt` |

Why #9 is called a "multiplier" rather than its own severity tier: it's the
reason several of the above (especially #1, #2, #3, #6) can persist
undetected — there's no automated signal that would catch a regression in
any of them.

## Refactoring strategies (not implemented in this PR)

For each area above that isn't one of the 4 demos, what should change, why,
and why it's deliberately left as a report item rather than a diff here.

### 1. Serial execution + no HTTP timeouts (critical)
**What:** Either give each minion more than one worker (requires auditing
whether `self._state` and Spark session usage are actually thread-safe for
concurrent jobs — they may not be, since `_state` is a plain dict mutated
without locking), or at minimum add `timeout=` to every `requests.get/post`
call in `juicer/service/*.py` so a single hung metadata call can't stall the
one worker indefinitely.
**Why not here:** the concurrency half needs a careful audit of shared
mutable state (`self._state`, cached Spark session, `job_future`) under
concurrent access — get that wrong and you get silent data corruption, not
just a hang. The HTTP-timeout half is safer and smaller, but "what timeout
value" and "what should happen to a job whose metadata call now fails
that previously would have hung" are policy decisions that need a live
Tahiti/Limonero/Stand environment to validate against, which isn't
available in this environment.

### 2. Dead retry path in server.py (critical)
**What:** Either implement the retry (respawn a minion for pending queue
items when its process disappears) or remove the dead branch and pending
messages explicitly, with a decision on what "lost job" behavior should be
(retry vs. surface an error to the UI).
**Why not here:** this is a behavior *decision*, not a mechanical
delete — turning `if pendings and False:` into `if pendings:` changes
runtime behavior in a way that needs testing against a real Redis +
minion pair to be sure it doesn't double-start minions or race the
`active_minions` hash update just above it.

### 3. `ping()` has no exception handling (high)
**What:** Wrap `self._perform_ping()` in `juicer/runner/minion_base.py:122`
in a try/except that logs and continues (or backs off and retries) instead
of letting an unhandled exception silently end the `while q.empty():` loop.
**Why not here:** cheap fix in isolation, but validating that it actually
prevents "ghost minions" needs a way to inject a transient Redis failure
and confirm the server's view of `active_minions` stays consistent
afterward — that's an integration-level test this environment can't run
(no live Redis).

### 4. Copy-paste boilerplate beyond the 3 classes touched (high)
**What:** Apply `build_ctor_params` (added in Demo 4) to the remaining
classifier/regressor/clusterer classes in `ml_operation.py` that share the
same `params_name` table + loop shape, and look for the analogous pattern
in `etl_operation.py` (both Spark and scikit-learn variants).
**Why not here:** the task explicitly scoped Demo 4 to 3 classes "leave the
others for a later pass" — each remaining class needs its own read to
confirm the loop body really is the unmodified boilerplate (some classes in
this file have extra logic mixed into the loop, e.g. validation or
special-casing a parameter) rather than assuming uniformity across ~15+
classes.

### 5. `dataframe_util.py` grab-bag (medium)
**What:** Split by concern — sampling/emit functions, CSV/JSON conversion
helpers, the JSONEncoder family, attribute analysis — into separate modules
under `juicer/util/`.
**Why not here:** explicitly out of scope per task instructions; also a
wide-diff change (every importer of `juicer.util.dataframe_util` would need
updating) that's easy to get wrong without a full call-site inventory.

### 6. Jinja/gettext rebuilt per call, no Redis pooling, `cancel_job` busy-poll, N+1 HTTP (medium)
**What:**
- Cache the Jinja `Environment` + loaded `Translations` on the
  `Transpiler` instance (or a module-level cache keyed by template dir),
  instead of rebuilding from disk in every `generate_code()` call
  (`juicer/transpiler.py:373-390`).
- Use a shared `redis.ConnectionPool` instead of each `StrictRedis(...)`
  call opening its own connection (`juicer/runner/server.py:99,420`, and
  wherever `StateControlRedis` is instantiated per-call).
- Replace `cancel_job`'s `while True: cancelAllJobs(); result(timeout=1)`
  busy-loop (`juicer/spark/spark_minion.py:922-931`) with a bounded
  retry/backoff or a Spark job-listener callback instead of polling every
  second indefinitely.
- Batch or cache the sequential Tahiti/Limonero HTTP calls made per
  workflow transpile instead of issuing them one at a time per referenced
  dataset/task.
**Why not here:** each of these needs either a live Redis/Spark/Tahiti
environment to verify the fix doesn't regress (pooling and caching bugs are
notoriously hard to catch by static inspection — e.g. a stale cached
`Translations` object after a locale change, or a connection pool
exhausting under the same concurrency this review flags as risky in #1),
or, for the Jinja cache, a check that no template render depends on
mutating environment/global state between calls (the `AutoPep8Extension`
and `HandleExceptionExtension` custom extensions weren't audited for
statefulness here).

### 8. i18n + unused exception hierarchy (low)
**What:** Pick one i18n installation point (e.g. always install at process
start in a shared bootstrap function) instead of three independent
`translation(...).install()' call sites; either use
`JuicerException`/`InvalidGeneratedCode` consistently or remove them.
**Why not here:** low severity, and consolidating i18n installation touches
every entry point (`app.py`, `server.py`, each `*_minion.py`) — a change
that's simple to describe but has a wide blast radius for a low-value fix,
better done as its own reviewed change.

### 9. Zero CI / uneven coverage (structural multiplier)
**What:** Re-enable `pytest tests/` in `.travis.yml` (or migrate to GitHub
Actions), and un-skip or delete the 4 `@pytest.mark.skip(reason="Not
working")` tests in `tests/spark/test_spark_minion.py` after fixing or
removing whatever made them fail.
**Why not here:** this environment has no `pytest`, `pandas`, `pyspark`, or
`duckdb` installed (see verification notes in the PR), so re-enabling CI
and confirming the suite actually passes can't be validated from here —
doing it blind risks turning on CI against a suite that immediately goes
red.

### 10. `requirements.txt` / `requirements-3.7.txt` duplication (low)
**What:** Decide whether the 3.7-pinned variant is still needed (project
now runs on 3.10 per this worktree's Python), and either drop it or
generate it from the same source (e.g. a constraints file per Python
version) instead of hand-maintaining two full lists.
**Why not here:** explicitly out of scope per task instructions; also needs
confirmation nothing in the deployment pipeline still targets Python 3.7
before deleting it.

## The 4 demos actually implemented

See the PR description for a summary; each demo's full detail (what was
found, why it's safe, what was verified) is in its own commit message.
Diffs:

1. **Dead duplicate `emit_sample_sklearn`** —
   `juicer/util/dataframe_util.py`
2. **DuckDB `JoinOperation` delegation** —
   `juicer/scikit_learn/duckdb/etl_operation.py`,
   `tests/scikit_learn/etl/test_duckdb_join_delegates_to_sklearn.py`
3. **Dead `sc` reference removed from `terminate()`** —
   `juicer/spark/spark_minion.py`, `tests/spark/test_spark_minion.py`
4. **Shared `build_ctor_params` helper** —
   `juicer/operation.py`, `juicer/spark/ml_operation.py`
