# Using `adaptive_cs`

`adaptive_cs` tracks posterior predictive class probabilities and stops sampling
when a selected condition is met. Install the tracker as described in the
[root README](../README.md#python-setup) before running these examples. The
same Python API works with a prebuilt wheel or a local source build.

## A first tracker

This runnable example uses independent synthetic probability draws. In an
application, replace the Dirichlet draw with a fresh probability vector from
your model for the same input.

```python
import numpy as np
import adaptive_cs as cs

rng = np.random.default_rng(42)
tracker = cs.ClassSelection(alpha=0.05, max_samples=200)

while not tracker.done:
    probabilities = rng.dirichlet([9.0, 1.0])  # [class]
    result = tracker.update(probabilities)

if bool(result.certified[0]):
    print("Selected class:", int(result.prediction[0]))
else:
    print("Stopping condition not met:", result.status[0])
```

`max_samples` limits the number of draws. Reaching that limit ends the loop
even if the class selection condition has not been met. Check
`result.certified` before using the prediction as a stopped decision.

## Inputs and batches

Pass NumPy arrays or PyTorch tensors containing **probabilities**, not logits,
class labels, or probabilities averaged across draws. Each vector needs at
least two classes, finite entries in `[0, 1]`, and entries that sum to one.
The paper's guarantees require independent posterior predictive draws.

| Shape passed to `update` | Meaning |
| --- | --- |
| `[class]` | One draw for one input. |
| `[input, class]` | One draw for several inputs. |
| `[draw, input, class]` | Several draws for several inputs. |

The first update fixes the input batch size and number of classes. For later
updates, use `tracker.active_indices` to sample only inputs whose stopping
condition has not been met:

```python
tracker = cs.ClassSelection(alpha=0.05, max_samples=200)
while not tracker.done:
    active = tracker.active_indices
    probabilities = predict_probabilities(x_batch[active])  # [active input, class]
    result = tracker.update(probabilities, indices=active)
```

`predict_probabilities` is a placeholder for your model. Before the first
update, `active_indices` is `slice(None)`; afterward it contains indices into
the original input batch. Every `result` retains that original order, including
inputs omitted from a later update.

To provide several draws at once, pass an array of shape
`[draw, active input, class]` to the same `update` call. Do not average over
the draw axis. Without a custom inspection schedule, the tracker processes
each draw and checks bounds at the end of the call. If the array extends past
`max_samples`, only the draws up to the limit are consumed.

## Choose a stopping rule

| Tracker | Main argument | Stopping condition |
| --- | --- | --- |
| `cs.ClassSelection` | None | The lower bound of the class with the largest running mean exceeds every other class's upper bound. |
| `cs.DistributionAccuracy` | `epsilon` | Each class interval lies within `epsilon` of its running estimate. |
| `cs.Threshold` | `threshold` | The leading class's lower bound reaches the threshold, or every upper bound falls below it. |
| `cs.Uncertainty` | `component`, `stopping`, and `epsilon` or `threshold` | The selected entropy uncertainty meets an accuracy or threshold condition. |

For example:

```python
class_tracker = cs.ClassSelection(alpha=0.05, max_samples=200)
distribution_tracker = cs.DistributionAccuracy(
    epsilon=0.02, max_samples=200
)
threshold_tracker = cs.Threshold(threshold=0.7, max_samples=200)
uncertainty_tracker = cs.Uncertainty(
    component="epistemic", stopping="accuracy", epsilon=0.1,
    max_samples=500,
)
```

For `Threshold`, `threshold` is a probability in `[0, 1]`.
`result.decision` is a Boolean tensor: `True` means a class reached the
threshold; `False` means all classes fell below it. Interpret a decision only
where `result.certified` is `True`.

`Uncertainty` reports total, aleatoric, and epistemic entropy uncertainty in
nats. Choose `component="total"`, `"aleatoric"`, or `"epistemic"` and one of
these stopping modes:

| `stopping` | Required argument | Condition |
| --- | --- | --- |
| `"accuracy"` | `epsilon` | Both distances from the estimate to its interval endpoints are at most `epsilon`. |
| `"threshold"` | `threshold` | The interval is entirely at or above the threshold, or entirely below it. |

The uncertainty threshold is a nonnegative value in nats. Specify exactly
one of `epsilon` and `threshold`, according to the selected mode.
`result.total`, `result.aleatoric`, and `result.epistemic` each provide
`estimate`, `lower`, `upper`, and `width`. With natural logarithms, total
uncertainty is `H(E[p])`, aleatoric uncertainty is `E[H(p)]`, and epistemic
uncertainty is their difference. These intervals describe Monte Carlo
uncertainty under the supplied posterior, not model misspecification.

## Read the update result

Every `update` returns a `CSResult` snapshot. Its tensor fields retain the
original input axis, even for one input. The main fields are:

| Field | Meaning |
| --- | --- |
| `mean` | Running mean probabilities, shape `[input, class]`. |
| `lower`, `upper` | Confidence sequence bounds, shape `[input, class]`. |
| `prediction` | Class with the largest running mean, shape `[input]`; the stopping condition may still be unmet. |
| `n_samples` | Draws consumed for each original input, shape `[input]`. |
| `certified` | Whether the selected stopping condition was met for each input. |
| `resolved`, `newly_resolved` | Whether the tracker marked an input resolved, and whether that happened on this update. |
| `status` | Per-input status: `running`, `stopped`, `empty`, `abstained`, or `timeout`. |
| `decision` | Threshold decision where applicable; otherwise `None`. |

`status="stopped"` means the selected condition was met. An input can instead
end at the sample limit with `status="timeout"` and `certified=False`.
`abstained` means the optional early abstention rule stopped sampling; `empty`
indicates incompatible bounds. The tracker property `done` means that no
further draws will be consumed, not that every input met the stopping
condition.

For one input, `tracker.mean`, `tracker.lower`, `tracker.upper`, and
`tracker.prediction` omit the input axis. `tracker.t` gives that input's draw
count as an integer. For a batch, `tracker.n_samples` gives the counts for all
original inputs.

## Optional controls

`stop_on_resolve=False` keeps all inputs active, which can be useful when
recording full trajectories. Set a finite `max_samples` if you use this option
in a loop driven by `tracker.done`.

To check bounds only at specified absolute draw counts, pass
`inspection_times` with a finite `max_samples`:

```python
tracker = cs.ClassSelection(
    max_samples=200,
    inspection_times=(16, 24, 36, 54, 81, 122, 183, 200),
)
```

Every draw still updates the tracker. Bounds and stopping are checked at the
listed counts; the sample limit is added to the schedule if omitted.

`ClassSelection`, `DistributionAccuracy`, and `Threshold` also support
`early_abstention=True` when `max_samples` is finite and
`stop_on_resolve=True`. This may stop an input before the limit if the
selected condition can no longer be met by then. Its status is `abstained`,
and `certified` remains `False`. `Uncertainty` does not support this option.

For numerical settings, pass a `ConfidenceSequenceSettings` object through
`config=` (or `settings=`):

```python
settings = cs.ConfidenceSequenceSettings(
    alpha=0.05,
    root_finder="boost",
    endpoint_root_mode="componentwise",
    simplex_intersection=True,
)
tracker = cs.ClassSelection(config=settings, max_samples=200)
```

The available root finders are `"boost"`, `"brent"`, and `"bisection"`.
`"boost"`, componentwise endpoints, and simplex intersection are the defaults.
When passing `config=`, set `alpha` in that object.
