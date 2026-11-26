# Discussion

The fixed fixture gives evidence for an undirected neighborhood estimate
under its sampling assumptions. A direction claim requires more evidence.
The orientation branch retests single-variable conditional relationships
after the skeleton filter has already screened those relationships. The
observed role mismatch also affects checks that expect one of the
incoming neighbors to be labeled sensory.

The dynamic class builds a valid-shaped matrix for this fixture, but its
lagged tests call the inherited conditional routine with indices offset
by the original variable count. That routine treats indices beyond the
original width as independent before evaluating any correlations. An
empty lag result under this implementation therefore gives no evidence
about whether a real series has lagged effects.

The test uses a seeded numerical graph and grouped samples. It supplies
no benchmark for causal identification in real datasets, missing data,
or changing graphs. The historical gridworld runs are separate
synthetic examples of the visualization pipeline.
