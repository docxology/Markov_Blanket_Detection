# Introduction

In a directed graphical model, a target's parents, children, and the
other parents of its children form a Markov blanket [@pearl1988].
A blanket membership estimate and recovery of directed graph roles
are different verification targets. Membership can agree even if a
neighbor receives a different directional label.

The local DMBD repository implements a correlation-based approach
and explores partitions in a synthetic gridworld [@dmbdSoftware].
Its stated inspirations include the dynamic inference work of Beck
and Ramstead [@beck2025]. The present study evaluates the repository's
own procedures on its numerical test fixtures.
