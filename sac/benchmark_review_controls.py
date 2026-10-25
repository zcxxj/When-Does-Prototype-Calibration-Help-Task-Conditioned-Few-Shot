"""Fixed dev efficiency check; --review1 restricts execution to P0/E2."""
import time

import numpy as np
import torch

from .review_evaluate import predict_methods
from .euclidean import review_scores
from .train import model_inputs

METHODS = ('P0', 'E2', 'SOFTK')
REVIEW1_METHODS = ('P0', 'E2')
EPISODES, WARMUP, REPEATS = 20, 5, 3


def timed_backend(features, shot, method, device):
    """Time only the existing scoring rule on device-resident shared features."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = review_scores(features['proto'], features['query'], shot, (method,))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    del output
    return dict(forward_seconds=elapsed, peak_allocated=allocated, peak_reserved=reserved)


def timed_review1(operation, device):
    """Production operation only; retain outputs until the synchronized stop."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = operation()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    del output
    return dict(forward_seconds=elapsed, peak_allocated=allocated, peak_reserved=reserved)


def benchmark_review1_setting(model, episodes, clouds, methods):
    from . import euclidean
    if tuple(methods) != REVIEW1_METHODS:
        raise ValueError('review1 requires exactly P0, E2 in that order')
    device = next(model.parameters()).device
    rows = []
    for mode in ('end_to_end', 'backend_only'):
        features = None
        if mode == 'backend_only':
            with torch.no_grad():
                features = [model.episode_features(**model_inputs(ep, points, device))
                            for ep, points in zip(episodes, clouds)]
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        measurements = {method: [] for method in methods}
        warmup_orders = []
        for call_index in range(WARMUP + REPEATS * EPISODES):
            warming = call_index < WARMUP
            sample_index = call_index if warming else call_index - WARMUP
            index = sample_index % EPISODES
            order = list(methods if call_index % 2 == 0 else methods[::-1])
            if warming:
                warmup_orders.append(order)
            for position, method in enumerate(order):
                def operation():
                    if mode == 'end_to_end':
                        return model(**model_inputs(episodes[index], clouds[index], device),
                                     backend='euler', refine=method == 'E2')
                    return euclidean.anchor_scores(features[index]['proto'],
                        features[index]['query'], model.shot, refine=method == 'E2')
                result = timed_review1(operation, device)
                if not warming:
                    measurements[method].append(dict(repeat=sample_index // EPISODES,
                        episode_id=index, method_order=order, order_position=position,
                        seconds=float(result['forward_seconds']),
                        allocated_bytes=int(result['peak_allocated']),
                        reserved_bytes=int(result['peak_reserved'])))
        for method in methods:
            samples = measurements[method]
            seconds = [item['seconds'] for item in samples]
            rows.append(dict(method=method, mode=mode, episodes=EPISODES,
                warmup=WARMUP, repeats=REPEATS, samples=len(samples),
                warmup_orders=warmup_orders, measurements=samples,
                median_seconds=float(np.median(seconds)),
                p95_seconds=float(np.percentile(seconds, 95)),
                peak_allocated_bytes=max(item['allocated_bytes'] for item in samples),
                peak_reserved_bytes=max(item['reserved_bytes'] for item in samples)))
        del features
    return rows


def benchmark_setting(model, episodes, clouds, methods=METHODS, *, review1=False):
    """Prepared clouds exclude disk I/O/corruption; each method gets its own run."""
    if len(episodes) != EPISODES or len(clouds) != EPISODES:
        raise ValueError('Exactly the first 20 dev episodes are required')
    model.eval()
    if review1:
        return benchmark_review1_setting(model, episodes, clouds, methods)
    device = next(model.parameters()).device
    rows = []
    features = None
    for mode in ('end_to_end', 'backend_only'):
        if mode == 'backend_only':
            with torch.no_grad():
                features = [model.episode_features(**model_inputs(ep, points, device))
                            for ep, points in zip(episodes, clouds)]
        for method in methods:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

            def run(index):
                if mode == 'end_to_end':
                    records, _ = predict_methods(model, episodes[index], clouds[index],
                                                 (method,), diagnostics=False)
                    return records[method]
                return timed_backend(features[index], model.shot, method, device)

            for index in range(WARMUP):
                run(index)
            measurements = []
            for repeat in range(REPEATS):
                for index in range(EPISODES):
                    result = run(index)
                    measurements.append(dict(repeat=repeat, episode_id=index,
                        seconds=float(result['forward_seconds']),
                        allocated_bytes=int(result['peak_allocated']),
                        reserved_bytes=int(result['peak_reserved'])))
            seconds = [item['seconds'] for item in measurements]
            rows.append(dict(method=method, mode=mode, episodes=EPISODES,
                warmup=WARMUP, repeats=REPEATS, samples=len(measurements),
                median_seconds=float(np.median(seconds)),
                p95_seconds=float(np.percentile(seconds, 95)),
                peak_allocated_bytes=max(item['allocated_bytes'] for item in measurements),
                peak_reserved_bytes=max(item['reserved_bytes'] for item in measurements),
                measurements=measurements))
    return rows


SETTINGS = ((1, 'clean', None, 0), (1, 'query', 'Add-Global', 5),
            (5, 'clean', None, 0), (5, 'query', 'Add-Global', 5))


def save_report(out, report):
    import csv
    import json
    from pathlib import Path
    from .train import write_json
    root = Path(out) if report.get('review1') else Path(out) / 'reports' / 'efficiency'
    root.mkdir(parents=True, exist_ok=True)
    paths = dict(json=root / 'benchmark.json', csv=root / 'benchmark.csv',
                 markdown=root / 'REPORT.md')
    write_json(paths['json'], report)
    fields = ('shot', 'side', 'operator', 'severity', 'method', 'mode', 'episodes',
              'warmup', 'repeats', 'samples', 'median_seconds', 'p95_seconds',
              'peak_allocated_bytes', 'peak_reserved_bytes')
    with paths['csv'].open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['rows'])
    lines = ['# Fixed dev efficiency measurement', '',
        'ModelNet40 fold0 seed7 best-validation; dev first 20 episodes; 5-way, 15 queries/class.',
        'Only clean and query Add-Global severity5, 1/5-shot; not all corruption conditions.',
        '5 warmup calls per method/mode/setting; 3 rounds × 20 episodes (60 samples).',
        'End-to-end includes input transfer, PCIA and one backend; excludes file I/O, corruption and CPU output copies.',
        'Backend-only reuses identical cached device features; no PCIA or input transfer.',
        'Memory is absolute CUDA peak, including model and (backend-only) the 20-episode feature cache.',
        ('Method order alternates every episode pair, including warmups; allocator cache cleared once per mode, '
         'peaks reset per call; CUDA synchronized.' if report.get('review1') else
         'Allocator cache is cleared before each method/mode; peaks reset per call; CUDA synchronized.'),
        'Median/p95 pool the 60 episode calls, not three whole-round durations.', '',
        '## Hardware and versions', '', '```json', json.dumps(report['hardware'], indent=2), '```', '',
        '|shot|condition|method|mode|median ms|p95 ms|allocated bytes|reserved bytes|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in report['rows']:
        lines.append(f"|{row['shot']}|{row['side']} {row['operator'] or ''} {row['severity']}|"
                     f"{row['method']}|{row['mode']}|{1000*row['median_seconds']:.4f}|"
                     f"{1000*row['p95_seconds']:.4f}|{row['peak_allocated_bytes']}|{row['peak_reserved_bytes']}|")
    lines.extend(['', 'Per-episode support/query point counts, checkpoint/manifest provenance and raw timings: benchmark.json.'])
    paths['markdown'].write_text('\n'.join(lines) + '\n')
    return paths

