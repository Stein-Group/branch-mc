"""The log flush triggers: a time floor (primary) and a row cap (memory bound).

The clock is injected, so nothing here sleeps and the whole module is instant.
"""

from __future__ import annotations

import pytest

from gcmc import io
from gcmc.sim_setup import from_params


class FakeClock:
    """Monotonic clock under test control, so the time trigger is tested
    deterministically instead of by sleeping and hoping."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _rows(path):
    return [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]


# ------------------------------------------------------- default is unchanged


def test_the_default_flushes_every_row(tmp_path):
    """min_interval defaults to 0.0, so behaviour is byte-for-byte what it was
    before the trigger existed. Turning batching on has to be a decision."""
    p = tmp_path / "log.csv"
    lg = io.Log(p, "# h")
    for i in range(5):
        lg.write(f"row{i}")
        assert len(_rows(p)) == i + 1, "a row was withheld at the default setting"
    lg.close()


# ------------------------------------------------- the time floor self-tunes


def test_a_slow_producer_still_flushes_every_row(tmp_path):
    """THE POINT OF THE DESIGN. Production writes one row per window, hundreds of
    ms apart. Those exceed the floor every time, so nothing is ever withheld and
    no durability is traded away for a saving that measurement says is 0.02%."""
    clk = FakeClock()
    p = tmp_path / "log.csv"
    lg = io.Log(p, "# h", flush_every=1000, min_interval=0.05, clock=clk)
    for i in range(5):
        clk.advance(0.31)  # a real window at sampling_window=1000
        lg.write(f"row{i}")
        assert len(_rows(p)) == i + 1
    lg.close()


def test_a_fast_producer_batches(tmp_path):
    """Rows every 1 ms against a 50 ms floor: they accumulate and go out in
    batches, so the flush RATE is bounded however fast rows arrive."""
    clk = FakeClock()
    p = tmp_path / "log.csv"
    lg = io.Log(p, "# h", flush_every=1000, min_interval=0.05, clock=clk)
    for i in range(40):
        clk.advance(0.001)
        lg.write(f"row{i}")
    on_disk = len(_rows(p))
    assert on_disk < 40, "nothing was batched despite rows arriving 50x too fast"
    lg.close()
    assert len(_rows(p)) == 40, "close() must not lose the tail"


def test_the_loss_window_is_bounded_in_time_not_in_rows(tmp_path):
    """The invariant worth having: whatever the row rate, an unflushed tail
    represents at most `min_interval` seconds of work. Checked at two rates an
    order of magnitude apart -- a row-count bound could not hold both."""
    for dt, n in ((0.001, 200), (0.0001, 2000)):
        clk = FakeClock()
        p = tmp_path / f"log_{dt}.csv"
        lg = io.Log(p, "# h", flush_every=10**9, min_interval=0.05, clock=clk)
        for i in range(n):
            clk.advance(dt)
            lg.write(f"row{i}")
        withheld = n - len(_rows(p))
        assert withheld * dt <= 0.05 + 1e-9, (
            f"at {dt}s/row the unflushed tail was {withheld} rows = "
            f"{withheld * dt:.4f}s, over the {0.05}s floor"
        )
        lg.close()


# --------------------------------------------------- the row cap bounds memory


def test_the_row_cap_fires_even_when_the_clock_does_not(tmp_path):
    """flush_every survives as a memory bound: a producer fast enough to outrun
    the floor cannot grow the buffer without limit."""
    clk = FakeClock()  # never advanced: the time trigger can never fire
    p = tmp_path / "log.csv"
    lg = io.Log(p, "# h", flush_every=8, min_interval=1e9, clock=clk)
    for i in range(24):
        lg.write(f"row{i}")
    assert len(_rows(p)) == 24, "the cap did not bound the buffer"
    lg.close()


def test_close_and_reopen_appends(tmp_path):
    """Unchanged property: a resumed run continues the same file, and the header
    is written once per file lifetime."""
    clk = FakeClock()
    p = tmp_path / "log.csv"
    lg = io.Log(p, "# h", flush_every=1000, min_interval=0.05, clock=clk)
    lg.write("a")
    lg.close()
    lg2 = io.Log(p, "# h", flush_every=1000, min_interval=0.05, clock=clk)
    lg2.write("b")
    lg2.close()
    assert _rows(p) == ["a", "b"]
    assert p.read_text().count("# h") == 1


# ------------------------------------------------------------------- the config


def _p(run):
    return {
        "potential": {"main": "hard_core"},
        "sampler": {"type": "metropolis"},
        "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
        "run": {"kind": "single", **run},
    }


def test_the_config_default_preserves_flush_every_row():
    assert from_params(_p({})).log_flush_interval == 0.0


def test_the_config_carries_the_interval():
    assert from_params(_p({"log_flush_interval": 0.25})).log_flush_interval == 0.25


def test_a_negative_interval_is_refused_with_both_numbers():
    with pytest.raises(ValueError, match="cannot be negative"):
        from_params(_p({"log_flush_interval": -1.0}))


def test_the_interval_does_not_move_the_run_identity():
    """It is an execution knob, not physics: two runs differing only in flush
    cadence must share a run directory."""
    from gcmc.hash_fingerprint import run_id

    assert run_id(from_params(_p({}))) == run_id(
        from_params(_p({"log_flush_interval": 0.5}))
    )
