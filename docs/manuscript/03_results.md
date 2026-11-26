# Results

The detector matched all five expected blanket member sets. It matched the
parent, child, and spouse sets together for two targets. For X3, the detected
blanket contained the expected X1 and X2. Both were reported as children,
while the generator's graph identifies them as parents. The independent
variable X4 had an empty blanket.

| Comparison | Result |
| --- | ---: |
| Exact membership matches across targets | 5 |
| Exact parent, child, and spouse matches across targets | 2 |
| Dynamic lag matrix rows | 80 |
| Dynamic lag matrix columns | 10 |
| Targets with a reported lag member | 0 |

The current-index role assertion in the dynamic tests also fails. The
matrix below shows membership agreement for the static fixture.

![Expected and estimated blanket membership for every static fixture target. Both panels show the same member marks. A blank row for X4 indicates an empty blanket.](../../output/figures/fixture_membership.png)
