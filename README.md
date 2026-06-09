Install : ``` pip install -e . ```

## Compatibility table (enforced; each ✗ has its own error message)

| Sampler | Main potential | Auxiliary | depth | trials | trial_box_side | 2D batch | 3D batch | 2D parallel | 3D parallel |
|---|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| metropolis | hard_core, LJ | (pmf pending) | ✗ | ✗ | ✗ | ✓ | ✓ | ✓\* | ✗ |
| rosenbluth | hard_core, LJ | pmf (hard_core) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| branch | hard_core | none | ✓ | ✓ | ✓ | ✓ | ✓ | ✓\* | ✗ |
| reweight | hard_core | psi6 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓\* | ✗ |

