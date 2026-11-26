# Methods

The static fixture generates 100 observations across five variables with
NumPy seed 42. In the known graph, X0 supplies X1 and X2, and X1 and X2
supply X3. X4 is generated independently. The detector receives
the resulting observations. The separately declared test expectations
provide the comparison labels.

The detector screens marginal correlations, then checks whether dependence
disappears when conditioning on a single variable or the full remaining
set. It assigns role labels to surviving neighbors and probes candidates
that may share a child with the target. The comparison evaluates the
union of role sets and exact agreement of the ordered parent, child, and
spouse categories for every target.

The time-indexed fixture forms equal-sized observation groups. We inspect
the shape of the constructor's lag matrix and record reported lag roles
for each target. We retain the native results as JSON and draw the
membership matrix directly from those results.
