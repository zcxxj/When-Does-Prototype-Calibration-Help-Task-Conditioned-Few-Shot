"""External scoring only: query targets never feed calibration."""
import numpy as np


DIAGNOSTIC_KEYS = ('corrected_count', 'damaged_count', 'group_size', 'group_purity',
                   'group_class_accuracy', 'anchor_fraction', 'displacement',
                   'soft_mass', 'initial_accuracy')


def score_diagnostics(result, targets):
    """Per-episode counts (query-count denominator), class statistics and NA.

    Purity is majority-label fraction; class accuracy is agreement with the
    prototype class, not majority purity. Empty groups are NaN, never perfect.
    """
    targets = np.asarray(targets)
    before = np.asarray(result['logits_before']).argmax(-1)
    after = np.asarray(result['logits_after']).argmax(-1)
    size = np.asarray(result['group_size'])
    purity = np.full(len(size), np.nan)
    accuracy = np.full(len(size), np.nan)
    for c, count in enumerate(size):
        if count:
            labels = targets[before == c]
            purity[c] = np.unique(labels, return_counts=True)[1].max() / count
            accuracy[c] = np.mean(labels == c)
    old_correct, new_correct = before == targets, after == targets
    return dict(corrected_count=np.count_nonzero(~old_correct & new_correct),
                damaged_count=np.count_nonzero(old_correct & ~new_correct),
                initial_accuracy=np.mean(old_correct), group_size=size,
                group_purity=purity, group_class_accuracy=accuracy,
                anchor_fraction=np.asarray(result['anchor_fraction']),
                displacement=np.asarray(result['displacement']),
                soft_mass=np.asarray(result['soft_mass']))


EMPIRICAL_SCALAR_KEYS = ('initial_macro_accuracy', 'final_macro_accuracy', 'final_micro_accuracy',
                         'corrected_micro', 'damaged_micro', 'corrected_macro', 'damaged_macro')
EMPIRICAL_CLASS_KEYS = ('class_query_count', 'class_corrected_count', 'class_damaged_count',
                        'initial_class_accuracy', 'final_class_accuracy')
EMPIRICAL_QUERY_KEYS = ('soft_entropy', 'soft_max_probability')
EMPIRICAL_DIAGNOSTIC_KEYS = DIAGNOSTIC_KEYS + EMPIRICAL_SCALAR_KEYS + EMPIRICAL_CLASS_KEYS + EMPIRICAL_QUERY_KEYS


def accuracy_metrics(predictions, targets, way):
    """Both denominators are explicit; every episode class must have queries."""
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    counts = np.bincount(targets, minlength=way)
    if (predictions.shape != targets.shape or targets.ndim != 1 or len(counts) != way or
            np.any(counts == 0)):
        raise ValueError('Expected aligned predictions and queries from every episode class')
    correct = np.bincount(targets[predictions == targets], minlength=way)
    return dict(micro_accuracy=float(np.mean(predictions == targets)),
                macro_accuracy=float(np.mean(correct / counts)), class_accuracy=correct / counts)


def extended_score_diagnostics(result, targets):
    """Recomputable raw mechanism statistics and macro/micro flip contributions.

    SOFTK responsibilities are softmax of the initial logits, at the method's
    frozen unit temperature. These uncertainty arrays describe that assignment
    distribution; they are not probabilities fitted using query truth.
    """
    values = score_diagnostics(result, targets)
    targets = np.asarray(targets)
    logits = np.asarray(result['logits_before'], dtype=np.float64)
    before, after = logits.argmax(-1), np.asarray(result['logits_after']).argmax(-1)
    way = logits.shape[1]
    initial, final = accuracy_metrics(before, targets, way), accuracy_metrics(after, targets, way)
    counts = np.bincount(targets, minlength=way)
    corrected = np.bincount(targets[(before != targets) & (after == targets)], minlength=way)
    damaged = np.bincount(targets[(before == targets) & (after != targets)], minlength=way)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probs)
    values.update(initial_macro_accuracy=initial['macro_accuracy'],
                  final_macro_accuracy=final['macro_accuracy'], final_micro_accuracy=final['micro_accuracy'],
                  corrected_micro=corrected.sum() / len(targets), damaged_micro=damaged.sum() / len(targets),
                  corrected_macro=np.mean(corrected / counts), damaged_macro=np.mean(damaged / counts),
                  class_query_count=counts, class_corrected_count=corrected, class_damaged_count=damaged,
                  initial_class_accuracy=initial['class_accuracy'], final_class_accuracy=final['class_accuracy'],
                  soft_entropy=-(probabilities * log_probs).sum(axis=1),
                  soft_max_probability=probabilities.max(axis=1))
    return values


def empirical_diagnostic_shapes(count, way=5, queries=75):
    scalar = {'corrected_count', 'damaged_count', 'initial_accuracy', *EMPIRICAL_SCALAR_KEYS}
    return {key: (count,) if key in scalar else (count, queries) if key in EMPIRICAL_QUERY_KEYS
            else (count, way) for key in EMPIRICAL_DIAGNOSTIC_KEYS}
