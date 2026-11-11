# TODO — Markov_Blanket_Detection (DMBD)

Backlog for agent-ergonomics work. Source: cold-start audit 2026-08-31
(REVIEW_LOG_2026-08-31.md). All Minor/Medium items fixed in that pass unless
marked open.

## Minor
- [x] Root `README.md` "Running Tests" section gives a broken command
      (`python -m unittest discover tests` -> ImportError; tests live in
      `DMBD/tests/`). Fixed to point at `DMBD/run_tests.py` +
      `bash DMBD/run_dmbd_tests.sh`. — README.md
- [x] 42 auto-generated `SKELETON` stub files (AGENTS.md/README.md pairs)
      dated 2026-08-30 presented themselves as docs; deleted across
      `DMBD/**` and `docs/manuscript/` (see REVIEW_LOG for list).
- [x] Root `AGENTS.md` claims the repo is "never committed" as a
      local-only `projects/ongoing/` path — false: this is a git repo with
      remote `github.com/docxology/Markov_Blanket_Detection` on branch
      `main`. Corrected. — AGENTS.md

## Medium
- [x] No orientation ladder in the entry doc: status, next actions, and
      verification command were absent or scattered. Added "Status" and
      "For agents" sections to README.md linking canonical homes.
- [x] No backlog file existed. This TODO.md is now the single canonical
      backlog; `AGENTS.md` links to it.
- [x] Test runner requires a project venv (`DMBD/.venv`) or system deps
      (torch, pandas, ...); no doc stated this, and a cold agent hits
      `ModuleNotFoundError`. Documented the verified command in README.md
      and AGENTS.md.

## Major
- [ ] **Verify `pip install dmbd` claim.** Root README says `pip install
      dmbd` works, but no PyPI publication is verifiable from inside the
      repo and package author metadata (`OpenManus Team`,
      `openmanus.org`) does not match this repo's origin. Needs owner
      decision: publish to PyPI or rewrite README install section as
      development-install only. Deferred — out of doc-pass scope.
- [ ] **Fix 4 failing tests.** Measured 2026-08-31 with
      `cd DMBD && .venv/bin/python run_tests.py` (venv deps verified):
      41 tests, 4 failures — test_detect_blanket,
      test_detect_dynamic_blanket (test_markov_blanket.py);
      test_compute_information_flow, test_identify_cognitive_structures
      (test_cognitive_identification.py). All are expected-set/threshold
      mismatches in detection output, i.e. source/test logic, out of scope
      for a doc pass. Full log: `DMBD/output/test_results.txt`.

## Maintenance pass 2026-10-25

- [x] Documented install path was broken: `pip install -e .` from `DMBD/`
      died in `setup.py` with `FileNotFoundError: README.md`, and
      `pyproject.toml` pointed `readme` at the same missing file. `setup.py`
      is now a shim over PEP 621 metadata in `pyproject.toml`; editable
      install re-verified.
- [x] Removed tracked build artifacts (`DMBD/**/__pycache__/*.pyc`,
      `DMBD/dmbd.egg-info/`); added `*.egg-info/` to `DMBD/.gitignore`.
- [x] Corrected package metadata: URLs pointed at `github.com/openmanus/dmbd`,
      license claimed MIT (repo is CC BY-NC-SA 4.0), author was an unrelated
      project's team, and the direct `scipy` import was undeclared.
      Dependency floors raised to current supported releases.
- [ ] Open: gridworld animation tests error when `ffmpeg` is absent from
      `PATH` — matplotlib falls back to `PillowWriter` and cannot write
      `.mp4`. Re-run with ffmpeg installed to confirm; the 4 logic failures
      above are unchanged (41 tests, 4 failures, 2 ffmpeg errors on
      2026-10-25).
- [ ] Open: `DMBD/src/analyze_results.py` is referenced nowhere (no imports,
      no docs, no shell wrapper) and imports undeclared `cv2` + `PIL`; its
      purpose overlaps the documented `DMBD/analyze_partition.py`. Decide:
      declare the extra deps, or drop the script as superseded.

## Workflowz review — 2026-11-10

- [ ] **WFZ-01 (P1): Make the documented test command work from repo root.** `AGENTS.md:38` and `README.md:11` tell agents to run `python DMBD/run_tests.py` from repo root, but `DMBD/run_tests.py:49` creates `Path('output')` relative to the current working directory and `DMBD/run_tests.py:28-31` imports test modules that the packet does not show; meanwhile `DMBD/run_dmbd_tests.sh:22` creates `DMBD/output` and `DMBD/run_dmbd_tests.sh:43` does `cd DMBD` before invoking the runner, so the two documented entry points disagree about where outputs land. Observable outcome: running the documented root command exits 0 and writes `test_results.txt`/`test_log.txt` under `DMBD/output/` (or the docs are corrected to the `cd DMBD` form and the `--output-dir` flag at `DMBD/run_tests.py:120` actually controls the directory). Acceptance probe: from repo root run `python DMBD/run_tests.py` (with `DMBD/.venv` deps per `AGENTS.md:38-39`) and confirm exit code plus the location of `test_results.txt`; then run `python DMBD/run_tests.py --output-dir /tmp/dmbd-out` and confirm results appear there, not in `./output`. Dependencies/constraints: requires the venv with torch/pandas; the 4 known logic failures in `TODO.md:38-45` mean exit code may be 1 — the probe's contract is output location and the `--output-dir` flag, not green tests.
- [ ] **WFZ-02 (P2): Honor `--skip-animation` in the gridworld analysis pipeline.** `DMBD/run_gridworld_analysis.py:82-86` defines `--skip-animation` and `DMBD/run_gridworld_analysis.py:125` prints it, but the call at `DMBD/run_gridworld_analysis.py:141-149` never forwards it to `run_gridworld_dmbd_analysis`; the shell wrapper `DMBD/run_gridworld_dmbd.sh:137-139` builds `ANIMATION_ARG` and passes it at line 158, so the flag is accepted and silently dropped. Observable outcome: with `--skip-animation`, no `.mp4` is produced and the run succeeds without ffmpeg, directly addressing the open ffmpeg item in `TODO.md:60-64`. Acceptance probe: from repo root run `cd DMBD && python run_gridworld_analysis.py --quick-mode --skip-animation --output-dir /tmp/dmbd-gw` and assert no `*.mp4` exists under `/tmp/dmbd-gw` while the summary/PNG outputs are still generated. Dependencies/constraints: depends on the signature of `run_gridworld_dmbd_analysis` in `DMBD/src/gridworld_dmbd.py` (not in packet — confirm it accepts a skip/animate parameter before wiring; if it lacks one, that is the fix site).
- [ ] **WFZ-03 (P2): Fix `--use-torch` default/flag mismatch in gridworld scripts.** `DMBD/run_gridworld_analysis.py:76-80` uses `action='store_true'` with help text "default: True", so the actual default is `False` and `DMBD/run_gridworld_analysis.py:124` prints `Using PyTorch: False` unless the flag is passed; the wrapper `DMBD/run_gridworld_dmbd.sh:27` sets `USE_TORCH=true` by default and only passes the flag when true (lines 133-135), masking the mismatch for shell users but not for direct Python users. Observable outcome: default behavior matches the documented default (or docs/help are corrected), and `--use-torch`/`--no-use-torch` behave symmetrically. Acceptance probe: from repo root run `cd DMBD && python run_gridworld_analysis.py --help` and then a `--quick-mode` run without the flag, checking the printed `Using PyTorch:` line matches the documented default. Dependencies/constraints: none beyond the venv; do not change numerical defaults silently — align help text and actual default.
- [ ] **WFZ-04 (P2): Resolve the orphaned `analyze_results.py` script.** `TODO.md:65-68` already flags `DMBD/src/analyze_results.py` as unreferenced with undeclared `cv2`/`PIL` imports (`DMBD/src/analyze_results.py:11-12`); the visible source confirms a real defect independent of the decl/drop decision: `DMBD/src/analyze_results.py:70` computes `relative_frame = int((frame_idx / max(frames)) * ...)` which raises `ZeroDivisionError` when `frames=[0]` (single-frame comparison) and mis-maps frames whenever indices don't start at 0. Observable outcome: either the script is removed as superseded by `DMBD/analyze_partition.py`, or `opencv-python`/`Pillow` are declared in `DMBD/pyproject.toml:19-29` and the frame-mapping math handles a single frame at index 0. Acceptance probe: if kept, from repo root run `cd DMBD && python -c "from src.analyze_results import compare_simulation_and_partition"` against a one-frame fixture directory (frame index 0) and confirm a comparison PNG is produced instead of `ZeroDivisionError`; if dropped, confirm no docs/scripts reference it. Dependencies/constraints: owner decision on keep-vs-drop per `TODO.md:65-68`; the division bug only needs fixing on the keep path.
