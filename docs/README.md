# DMBD documentation

- [Setup and input data](USAGE.md) covers installation, core inputs,
  detector options, and native verification.
- [Gridworld guide](GRIDWORLD.md) describes the moving-Gaussian demonstration,
  command-line flags, and output layout.
- [Technical manuscript](manuscript/README.md) contains the canonical source
  and the reproducible synthetic fixture study.
- [Earlier manuscript scope audit](archive/manuscript_scope_audit.md)
  records the prior documentation state.

The implementation lives in [DMBD](../DMBD/). The
[framework modules](../DMBD/framework/) contain the detector, partitioner,
and visualization utilities. The [analysis code](../DMBD/src/) provides
higher-level entry points. [Historical outputs](../DMBD/output/) contain
earlier CSV, image, video, test, and report snapshots.
