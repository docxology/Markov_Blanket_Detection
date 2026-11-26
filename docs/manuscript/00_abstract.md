# Abstract

DMBD uses statistical tests to estimate Markov blanket memberships from
numeric observations. We compare its reported memberships and role
labels with the known graph used by the repository's seeded tests.
Membership agrees for each target in this fixture, while several role
labels disagree. The dynamic constructor assembles lagged columns, but
the conditional-independence test treats their indices as out of range.
These checks distinguish the behavior of local membership estimates
from claims about edge orientation and temporal dependencies.
