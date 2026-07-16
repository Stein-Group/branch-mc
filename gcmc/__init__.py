"""branch_mc: grand-canonical Monte Carlo for hard-sphere and Lennard-Jones
fluids, built around multi-trial local exchange moves and the transfer-matrix
channel sweep.

Layout, by responsibility:

  sim_setup   config schema -> frozen Config (all derived constants live here)
  runner      run/resume lifecycle: checkpoints, locks, logging, mu control
  MC_engine   the moves and their acceptance algebra; `advance` is one window
  neighbors   the persistent periodic cell list every hot-path query uses
  geometry    admissibility tests and cavity membership (exact oracles + the
              cell-list shortlist variants)
  trials      the TrialSet currency between geometry and the samplers
  samplers/   selection machinery: rosenbluth (sequential), branch (clique
              enumeration), channel (transfer-matrix DP sweep), weighers
  potentials/ hard sphere, Lennard-Jones, analytic/tabulated auxiliaries
  io          checkpoint/trajectory/log formats; hash_fingerprint names runs
  parallel /  legacy domain decomposition (2D checkerboard), kept working
  decomposition
  topology    species/molecule templates (groundwork for the rigid-molecule
              extension; the current engine is single-site)
  widom, rdf  analysis helpers
  migrate     legacy trajectory -> resumable run directory
"""
