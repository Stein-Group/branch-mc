"""The schema is described in three places; they must not drift apart.

`from_params` is the authority: it validates each params block against a literal
allowlist and rejects anything else, so a key it does not know is a hard error
and a key it knows but nobody documents is invisible. The other two copies are
examples/schema.json (machine-readable, what config validation uses) and the
Schema block in gcmc/sim_setup.py's docstring (what a person reads).

Copies rot. A detached annotated copy of this schema had drifted far enough to
document `use_depth`, a key the code now REJECTS (it became `full_repacking`),
and to miss `tail`, the target_* keys and the controller knobs entirely. Anyone
following it wrote configs the code refused. These tests exist so the copies
cannot drift again silently: the allowlists are read straight out of the source
with the AST, so the test cannot be fooled by a comment.
"""

import ast
import json
from pathlib import Path

import pytest

import gcmc.sim_setup as S

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_JSON = ROOT / "examples" / "schema.json"

# blocks whose keys _check() enforces, in the order from_params validates them
_BLOCKS = ("potential", "sampler", "ensemble", "run", "auxiliary")


def _allowlists_from_source():
    """Pull the literal key sets out of the `_check(section, {...}, {...})` calls
    inside from_params. Read from the AST, not from a docstring or a comment, so
    this measures the code that actually runs."""
    src = (ROOT / "gcmc" / "sim_setup.py").read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "from_params"
    )
    found = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "_check" or len(node.args) < 2:
            continue
        allowed = node.args[1]
        if not isinstance(allowed, ast.Set):
            continue
        keys = {e.value for e in allowed.elts if isinstance(e, ast.Constant)}
        found.append(keys)
    assert found, "no _check(...) allowlists found; did from_params change shape?"
    return found


def _block_allowlists():
    """Map each block name to its allowlist by identifying the set that contains
    the block's own required key(s). The top-level call (the one listing the
    block names themselves) is dropped."""
    lists = _allowlists_from_source()
    top = {"potential", "auxiliary", "sampler", "ensemble", "run"}
    out = {}
    marker = {
        "potential": "main",
        "sampler": "trial_box_side",
        "ensemble": "packing_fraction",
        "run": "cell_length",
        "auxiliary": "g3_csv",
    }
    for keys in lists:
        if keys == top:
            continue  # the top-level block list, not a block's keys
        for blk, m in marker.items():
            if m in keys:
                out[blk] = keys
    missing = set(_BLOCKS) - set(out)
    assert not missing, f"could not identify allowlists for {sorted(missing)}"
    return out


@pytest.mark.parametrize("block", _BLOCKS)
def test_schema_json_matches_code_allowlist(block):
    """Every key from_params accepts is declared, and nothing else is. The
    'nothing else' half matters: each block sets additionalProperties=false, so a
    key declared here but unknown to the code would validate and then raise, and
    a key the code writes but the schema omits makes valid configs fail. That is
    exactly how run.sweep_label broke every launcher-expanded config."""
    schema = json.loads(SCHEMA_JSON.read_text())
    declared = set(schema["properties"][block]["properties"])
    accepted = _block_allowlists()[block]
    assert declared == accepted, (
        f"examples/schema.json {block}: "
        f"missing {sorted(accepted - declared)}, extra {sorted(declared - accepted)}"
    )


def test_schema_json_forbids_unknown_keys():
    """The schema must be as strict as the code, or it is not a check."""
    schema = json.loads(SCHEMA_JSON.read_text())
    for block in _BLOCKS:
        assert schema["properties"][block].get("additionalProperties") is False, (
            f"{block} must set additionalProperties=false to match _check()"
        )


def test_docstring_schema_documents_every_key():
    """The human-readable copy names every key the code accepts, and invents
    none. This is the check the drifted copy would have failed on `use_depth`."""
    doc = S.__doc__ or ""
    start, end = doc.index("Schema"), doc.index("Sampled model")
    block_txt = doc[start:end]
    accepted = _block_allowlists()
    every = set().union(*accepted.values())

    undocumented = sorted(k for k in every if f'"{k}"' not in block_txt)
    assert not undocumented, (
        f"sim_setup docstring does not document {undocumented}; a key nobody "
        f"documents is a key nobody uses"
    )

    import re

    quoted = set(re.findall(r'"([A-Za-z_0-9]+)"', block_txt))
    # Values and nested pore keys are legitimately quoted but are not top-level
    # params keys, so they are not candidates for the ghost check. Grouped by
    # which key they are a value OF, because the flat list was the thing that
    # rotted: six value spellings were added to the docstring and this set was
    # not, so the check reported them as keys the code would reject.
    values = {
        # potential.main
        "hard_core",
        "lennard_jones",
        # auxiliary.type
        "pmf",
        "short_range",
        # sampler.type
        "metropolis",
        "rosenbluth",
        "branch",
        "channel",
        "reweight",
        # sampler.channel_length: "cyclic" is current, the rest legacy synonyms
        "cyclic",
        "full",
        "full_box",
        "span",
        # ensemble.type
        "muVT",
        "NVT",
        "QA",
        "confined",
        # run.kind
        "single",
        "parallel",
        # run.initial_config
        "grid",
        # ensemble.pore, a nested block with its own keys
        "slit",
        "cylinder",
        "geometry",
        "H",
        "R",
        "L_z",
    }
    # Any key starting with an underscore is the annotate-your-own-params
    # convention: _check() ignores it by prefix, so it is accepted rather than
    # rejected and cannot be a ghost. Matched by shape, not by listing "_" and
    # "_comment", so a params file inventing "_why" needs no test change.
    quoted = {q for q in quoted if not q.startswith("_")}
    ghosts = sorted(quoted - every - values - set(_BLOCKS))
    assert not ghosts, (
        f"sim_setup docstring documents {ghosts}, which from_params would "
        f"reject as unknown keys"
    )
