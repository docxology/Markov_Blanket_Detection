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
- [ ] **Test suite run-to-completion on this machine.** System pythons
      lack pandas/torch; the project venv exists at `DMBD/.venv` but
      import of torch/pandas there was too slow to confirm within the
      audit window (external-drive I/O). Verify with:
      `cd DMBD && .venv/bin/python run_tests.py`. Deferred — environment
      issue, not a docs issue; documented as unverified.
