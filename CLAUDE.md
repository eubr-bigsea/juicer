# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Juicer is the workflow processor for Lemonade (a visual data-science platform). It receives a workflow
specification in JSON from Citron/Tahiti, **transpiles** it into executable Python code targeting a
backend (Spark or scikit-learn), executes that code under its control, and reports status back to the
UI. Generated code creates intermediate datasets via the Limonero API.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt          # current
pip install -r requirements-3.7.txt      # Python 3.7-pinned variant

# Run the full test suite (CI does NOT currently run this — pytest is commented
# out in .travis.yml — but the suite exists and conftest.py wires up i18n +
# service mocks for it)
pytest tests/

# Run a single test
pytest tests/spark/test_ml_operations.py::test_svm_classifier_operation_success
pytest tests/scikit_learn/etl/test_join.py -k join_success

# Lint (pylint, config at .pylintrc)
pylint juicer

# Start/stop/status the server daemon (spawns juicer/runner/server.py)
./sbin/juicer-daemon.sh start
./sbin/juicer-daemon.sh status
./sbin/juicer-daemon.sh stop

# One-shot CLI transpile (no execution, no Redis/minion) — useful for testing
# code generation in isolation
python juicer/app.py -c conf/juicer-config.yaml -w <workflow_id>

# i18n (GNU gettext / pybabel) — regenerate translation catalogs after adding
# or changing translatable strings
pybabel extract -F babel.cfg -o juicer/i18n/juicer.pot .
pybabel compile -d juicer/i18n/locales
```

Configuration lives in a YAML file, default `conf/juicer-config.yaml` (see
`conf/juicer-config.yaml.template` for the expected shape: `juicer.servers.database_url`,
`juicer.servers.redis_url`, `juicer.services.tahiti.{url,auth_token}`, `juicer.config.tmp_dir`).

## Architecture

### Two entry points, one shared transpile core

- **CLI path** (`juicer/app.py`): fetch workflow JSON from Tahiti → `Workflow`
  (`juicer/workflow/workflow.py`) builds a `networkx` graph of tasks/flows → a `Transpiler` subclass is
  selected by the workflow's platform slug → `Transpiler.generate_code()` instantiates one `Operation`
  subclass per task and renders Jinja2 templates → a `.py` source file, optionally POSTed back to Stand.
  This path only *writes* code; it does not execute it.
- **Server path** (production, driven by the Stand UI): `juicer/runner/server.py` (`JuicerServer`)
  blocks on a Redis queue (`BLPOP` on `queue_start`) and, per incoming job, spawns or forwards to a
  **minion** — one OS subprocess (or Kubernetes Job) dedicated to a single `app_id`
  (`juicer/runner/minion.py` picks a platform-specific `Minion` subclass, e.g. `SparkMinion` in
  `juicer/spark/spark_minion.py`). A minion runs two daemon loops: `execute()` (pops jobs off its own
  Redis queue and runs them through a **single-worker** `ThreadPoolExecutor` — jobs for one app never
  run concurrently) and `ping()` (refreshes a Redis heartbeat key every 5s, 30s TTL). `perform_execute()`
  re-transpiles the workflow, `importlib`-imports/reloads the generated module, and calls its `main()`;
  results accumulate in `self._state` across jobs in the same minion (cleared only on `terminate()`).
  Minions self-terminate on idle timeout, explicit `TERMINATE`, batch-job completion, or an unhandled
  exception in `execute()`.

### Backends

Every backend follows the same `transpiler.py` + `*_operation.py` (one `Operation` subclass per task
type) + `*_minion.py` shape, all rooted in `juicer/operation.py::Operation` (shared base: parameter
bookkeeping, `must_be_executed()`, `to_deploy_format()`, a `render_template()` Jinja wrapper — it does
not provide shared param-parsing helpers, so operation `__init__`s tend to duplicate parsing logic
across backends).

- **`juicer/spark/`** — the primary/production backend (PySpark).
- **`juicer/scikit_learn/`** — pandas-based, with two alternate dataframe-engine variants that subclass
  the base scikit-learn operation classes and only override `generate_code()`:
  `juicer/scikit_learn/polars/` and `juicer/scikit_learn/duckdb/`.
- **`juicer/meta/`** — not a peer execution backend. It's a no-code/template layer
  (`MetaTranspiler.SUPPORTED_TARGET_PLATFORMS = {'spark': 1, 'scikit-learn': 4}`) that generates Spark or
  scikit-learn code for a template-driven UI (model-builder / data-explorer); `juicer/meta/operations.py`
  directly imports and wraps `juicer.spark.*` operation classes.

### Cross-cutting things worth knowing before touching code

- **Global config singleton**: `juicer/runner/configuration.py` holds a module-level `__CONFIG__`, set
  independently from three places (`app.py`, `server.py`, `spark_minion.py`) and implicitly read by
  every `Operation.__init__`. An `Operation` instantiated before `set_config()` runs gets `None`.
- **Import-time side effects**: several entry-point modules call `logging.config.fileConfig()`,
  `matplotlib.use('Agg', force=True)`, or mutate `sys.path` at module scope (not inside a function) —
  merely importing them reconfigures process-global state.
- **Minion job execution is serial per app** and **no outbound HTTP call has a timeout**
  (`juicer/service/*.py`'s `requests.get/post` to Tahiti/Limonero/Stand/Caipirinha) — a single hung
  metadata call can stall an entire minion, since the executor has only one worker.
  `juicer/runner/server.py` also has a dead retry path (`if pendings and False:`) — a message to a
  minion that dies mid-flight is dropped rather than redispatched.
- **Redis is used three separate ways**: `StateControlRedis` (`juicer/runner/control.py`) wraps
  queues/heartbeat keys/workflow-status hashes; `python-socketio`'s `RedisManager` rides Redis pub/sub
  to push real-time job/task events to the Stand UI; and `rq` (Redis Queue) is used independently in
  `juicer/transpiler.py` to enqueue auditing jobs. No connection pooling — each of these opens its own
  client.
- **i18n** uses gettext (`_()`/`gettext()`), installed as a Python builtin via `translation(...).install()`
  in most minion entry points — code that calls bare `_(...)` only resolves correctly once some minion
  has already called `.install()`; it will `NameError` if imported/run standalone.
