"""branch_mc: grand-canonical Monte Carlo for hard-sphere and Lennard-Jones
fluids, built around multi-trial local exchange moves and the transfer-matrix
channel sweep.

THE DIRECTORY IS THE ARCHITECTURE. Every module sits in the layer it belongs to,
and imports may only go DOWN this list -- asserted, not hoped for, by
`tests/unit/test_layer_ladder.py`, which holds the same table as data and fails
if a new module is added without someone deciding where it goes.

  _types, config_fields, config_specs   L0  the vocabulary every layer speaks:
                                            type aliases and the six sub-configs
                                            a layer uses to say what it needs
  forcefield/  L1  what interacts and how strongly, and it knows no geometry.
                   The potential registry, the per-pair tables, the overlap
                   convention (the floor), species and their force-field files
  space/       L2  where things are, and it knows no physics. Distances and
                   periodic images, the persistent cell list, the oriented
                   bodies a move is proposed inside, the domain decomposition
  system/      L3  the configuration being sampled: the chain state, where L1
                   and L2 meet
  proposals/   L4  what move to try: the move alphabet, the TrialSet currency,
                   the channel exchange geometry both move loops share
  sampling/    L5  how to weight a trial cloud: rosenbluth (sequential), branch
                   (clique enumeration), channel (transfer-matrix DP sweep),
                   reweighting, and the weight policies
  engine/      L6  the move loops. `MC_engine.advance` is one window; `parallel`
                   is the same window per cell; `framework` is the
                   framework loop
  run/         L7  making a params.json runnable: validation into a frozen
                   Config, the run/resume lifecycle, the on-disk formats, the
                   run id, the migrator and the sweep launcher

  analysis/    OFF THE LADDER, deliberately, at layer 3. g(r), Widom insertion
               and the demixing order parameter read a FINISHED configuration.
               Giving them a high layer would let the engine import a plotting
               helper, which is how analysis code ends up in a hot loop.

Two commands:

    python -m gcmc run <params.json> --root <dir>
    python -m gcmc.run.launcher <params.json> --cores N --root <dir>
"""
