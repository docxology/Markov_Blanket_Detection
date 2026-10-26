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
