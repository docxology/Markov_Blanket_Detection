# Reproducibility

The build uses the fixture setup methods in
`DMBD/tests/test_markov_blanket.py` and the corresponding
`classify_nodes` and `detect_dynamic_blanket` calls. It compares the
observations with the fixture's own expected labels, writes
`fixture_receipt.json`, and draws a bitmap from the observed and expected
membership sets.

Run the native discovery command in the setup guide to exercise the
separate gridworld, partition, visualization, and detector tests. The
manuscript uses Template's independent PDF and HTML checks. Neither
build updates the retained data and media in `DMBD/output/`.
