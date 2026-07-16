Frozen copy of the package at the end of stage A (all neighbour queries on the
CellList, trajectories still bit-identical to the pre-refactor engine).

It exists for one reason: stage B changes the ANSWERS -- the exchange path stops
carrying a compact `pos` array, membership is decided from cell gathers, and the
RNG stream necessarily shifts. Bit-identity is gone, so equivalence has to be
argued statistically, and that argument needs an engine to argue against. This
is that engine. Import it as `gcmcA` (the only edit is the package name; every
intra-package import is relative, so nothing else changed).

Do not fix bugs here. If stage B reveals a bug that predates it, fix it in
`gcmc/` and record in the ledger that this reference carries the old behaviour.
