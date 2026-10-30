#!/usr/bin/env python3
"""Recover original inclusive-tier circuit denominators using exact arithmetic.

Preserves all original G3 inputs and the historical setFinal publication set.
These denominators reproduce the ORIGINAL masks. They do not implement fixed
five-percent selection or identify which designs are tied at a threshold.

The per-topology growth counts are not individually identifiable. Only binary
context-presence sums, proved invariant to the complete null space, are emitted.
The original binary circuit-presence counts are smoothed with Beta(1,1):
(n_part + 1) / (N_circuits + 2). No MLP, simulator, or tie selection is rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sympy import Matrix
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
VF_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(REPO_ROOT / "topology" / "scripts"))
from _presence_enrichment import (PRESENCE_METADATA, PRESENCE_SCHEMA,
                                 presence_statistics)

DEFAULT_INPUT = REPO_ROOT / "data" / "topology_g3" / "l2_top05"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def zero(matrix: Matrix) -> bool:
    return all(value == 0 for value in matrix)


def exact_context_totals(C: Matrix, B: Matrix, observations: Matrix):
    """Prove binary context totals are unique despite unidentified topology counts."""
    n = C.cols
    require(C.rows == observations.rows and B.cols == n,
            'Incompatible count-equation matrix shapes.')
    reduced, pivots = C.row_join(observations).rref()
    require(all(p < n for p in pivots), 'Inconsistent observed count equations.')
    free = [j for j in range(n) if j not in pivots]
    particular = Matrix.zeros(n, observations.cols)
    null = Matrix.zeros(n, len(free))
    for i, pivot in enumerate(pivots):
        for k in range(observations.cols):
            particular[pivot, k] = reduced[i, n + k]
    for k, variable in enumerate(free):
        null[variable, k] = 1
        for i, pivot in enumerate(pivots):
            null[pivot, k] = -reduced[i, variable]
    require(zero(C * particular - observations),
            'Exact per-target count reconstruction failed.')
    require(zero(C * null), 'Invalid exact null-space basis.')
    require(zero(B * null),
            'Some unique-circuit context totals are not identifiable.')
    denominators = B * particular
    require(all(value.q == 1 and value > 0 for value in denominators),
            'Recovered denominators must be positive integers.')
    return denominators, particular, pivots, free


def recover_context_denominators(input_dir: Path) -> tuple[pd.DataFrame, dict]:
    names = ['l2_summary.json', 'l2_enrichment.csv',
             'l2_enrichment_by_target.csv']
    paths = {name: input_dir / name for name in names}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    summary = json.loads(paths['l2_summary.json'].read_text())
    require(summary.get('tier_cutoffs') ==
            {'top_quantile': 0.95, 'bot_quantile': 0.05},
            'Expected the archived inclusive 95th/5th percentile run.')
    topologies = summary['per_topology']
    aggregate = pd.read_csv(paths['l2_enrichment.csv'])
    per_target = pd.read_csv(paths['l2_enrichment_by_target.csv'])
    tasks = ['circuit_log', 'toxicity']
    require(set(aggregate.task) == set(tasks), 'Unexpected aggregate tasks.')
    require(set(per_target.task) == set(tasks), 'Unexpected per-target tasks.')
    for frame in [aggregate, per_target]:
        require(all(frame[c].dtype.kind in 'iu'
                    for c in ['top_count', 'bot_count']),
                'Raw counts must be stored as exact integers.')
        require((frame[['top_count', 'bot_count']].to_numpy() >= 0).all(),
                'Negative raw counts.')
    contexts = sorted(aggregate.fp_key.unique())
    context_index = {key: i for i, key in enumerate(contexts)}
    target_tables = {
        task: per_target[per_target.task == task].groupby(
            ['source', 'fp_key'])[['top_count', 'bot_count']].sum()
        for task in tasks
    }
    rows = sorted(target_tables[tasks[0]].index)
    for task, table in target_tables.items():
        require(set(table.index) == set(rows),
                f'Per-target context rows differ for {task}.')
    row_index = {key: i for i, key in enumerate(rows)}
    n = len(topologies)
    require(n == 215, 'This migration is pinned to the archived 215-topology cohort.')
    C = Matrix.zeros(len(rows), n)
    A = Matrix.zeros(len(contexts), n)
    B = Matrix.zeros(len(contexts), n)
    ids = [t['topology_id'] for t in topologies]
    require(len(ids) == len(set(ids)), 'Duplicate topology IDs in summary.')
    source_lists = []
    for j, topo in enumerate(topologies):
        raw = topo.get('source', 'unknown')
        sources = raw if isinstance(raw, list) else [raw]
        sources = sources or ['unknown']
        require(all(isinstance(s, str) and s for s in sources),
                'Non-string or empty source name.')
        require(len(sources) == len(set(sources)),
                'Repeated target names would double-count per-target data.')
        source_lists.append(sources)
        fingerprints = Counter(topo['fingerprint_keys'])
        require(sum(fingerprints.values()) == topo['n_slots'],
                'Fingerprint multiplicities do not equal topology slot count.')
        for context, multiplicity in fingerprints.items():
            require(context in context_index, 'Summary context absent from CSV.')
            k = context_index[context]
            A[k, j] = multiplicity
            B[k, j] = 1
            for source in sources:
                require((source, context) in row_index,
                        f'Missing exact source/context key: {source}/{context}')
                C[row_index[(source, context)], j] += multiplicity
    require(all(any(C[i, j] for j in range(n)) for i in range(C.rows)),
            'A per-target CSV row has no generating topology.')
    observations = Matrix.hstack(*[
        Matrix(target_tables[task].reindex(rows).values.tolist())
        for task in tasks
    ])
    denominators, particular, pivots, free = exact_context_totals(C, B, observations)
    known_circuit = Matrix([
        [t['n_top_designs'], t['n_bot_designs']] for t in topologies
    ])
    require(zero(C * known_circuit - observations[:, :2]),
            'Published circuit summary fails exact per-target control.')
    require(zero(B * known_circuit - denominators[:, :2]),
            'Recovered circuit denominators differ from known summary.')
    for k, task in enumerate(tasks):
        totals = aggregate[aggregate.task == task].groupby('fp_key')[
            ['top_count', 'bot_count']].sum().reindex(contexts)
        require(zero(A * particular[:, 2*k:2*k+2] -
                     Matrix(totals.values.tolist())),
                f'Exact aggregate slot-count reconstruction failed: {task}.')
        target_part_rows = per_target[per_target.task == task]
        require(target_part_rows.groupby(['source', 'fp_key']).part_idx.nunique().eq(20).all(),
                'Each source/context must include all 20 library parts.')
        require(aggregate[aggregate.task == task].groupby('fp_key').part_idx.nunique().eq(20).all(),
                'Each aggregate context must include all 20 library parts.')
    records = []
    for k, task in enumerate(tasks):
        for i, context in enumerate(contexts):
            top, bottom = map(int, denominators[i, 2*k:2*k+2])
            part_rows = aggregate[(aggregate.task == task) &
                                  (aggregate.fp_key == context)]
            require((part_rows.top_count <= top).all() and
                    (part_rows.bot_count <= bottom).all(),
                    'A part count exceeds its recovered circuit denominator.')
            records.append({
                'task': task, 'fp_key': context,
                'top_circuits': top, 'bottom_circuits': bottom,
                'topology_count': int(sum(B[i, j] for j in range(n))),
            })
    proof = {
        'purpose': 'Exact recovery of unique-circuit denominators for the original inclusive percentile masks.',
        'scope': 'Original G3 inputs preserved; derived v2 circuit-presence data only.',
        'input_directory': str(input_dir), 'input_sha256': hashes,
        'source_names': sorted({s for sources in source_lists for s in sources}),
        'n_topologies': n, 'n_contexts': len(contexts),
        'n_task_contexts': len(records),
        'n_source_context_equations': C.rows,
        'exact_rank': len(pivots), 'nullity': len(free),
        'free_parameter_topology_ids': [ids[j] for j in free],
        'exact_checks': {
            'per_target_equations_reproduced': True,
            'null_space_valid': True,
            'binary_context_presence_annihilates_null_space': True,
            'all_denominators_positive_integers': True,
            'known_circuit_per_topology_control_matches': True,
            'known_circuit_context_denominators_match': True,
            'aggregate_slot_counts_reproduced': True,
            'all_part_counts_within_circuit_denominators': True,
        },
        'limitations': [
            'Individual per-topology growth counts remain underdetermined and are not exported.',
            'These denominators retain all cutoff ties from the original run.',
            'Fixed-size five-percent selection requires revisiting saved prediction arrays.',
            'Denominator recovery itself is independent of the chosen pseudocount.',
            'Source target strings are matched exactly; no aliases or normalization were applied.',
        ],
    }
    return pd.DataFrame(records), proof


def eligibility(frame: pd.DataFrame) -> pd.Series:
    """Keep the original support filter; only the measured statistic changes."""
    return ((frame.n_topologies >= 5) & (frame.n_targets >= 3) &
            (frame.part_tier_support >= 1000))


def compare_with_count_metric(frame: pd.DataFrame) -> dict:
    comparison = {}
    for task, rows in frame[eligibility(frame)].groupby('task'):
        require(len(rows) == 1300 and rows.fp_key.nunique() == 65 and
                rows.part_idx.nunique() == 20,
                f'Eligibility changed unexpectedly for {task}.')
        require(rows.groupby('part_idx').fp_key.nunique().eq(65).all(),
                f'Parts no longer share 65 eligible contexts for {task}.')
        old = rows.raw_log2_count_ratio_reference
        new = rows.log2_presence_enrichment
        medians = rows.groupby('part_name')[[
            'raw_log2_count_ratio_reference', 'log2_presence_enrichment'
        ]].median()
        by_part = []
        for part, cells in rows.groupby('part_name'):
            by_part.append({
                'part_name': part,
                'old_median': float(medians.loc[part].iloc[0]),
                'new_median': float(medians.loc[part].iloc[1]),
                'old_positive_contexts': int((cells.raw_log2_count_ratio_reference > 0).sum()),
                'new_positive_contexts': int((cells.log2_presence_enrichment > 0).sum()),
                'old_negative_contexts': int((cells.raw_log2_count_ratio_reference < 0).sum()),
                'new_negative_contexts': int((cells.log2_presence_enrichment < 0).sum()),
            })
        exposure_ratio = rows.drop_duplicates('fp_key').eval(
            'top_design_count / bottom_design_count')
        comparison[task] = {
            'eligible_part_context_cells': len(rows),
            'eligible_contexts': int(rows.fp_key.nunique()),
            'sign_changes': int((old * new < 0).sum()),
            'median_sign_changes': int((medians.iloc[:, 0] * medians.iloc[:, 1] < 0).sum()),
            'maximum_absolute_change': float((old - new).abs().max()),
            'minimum_circuit_tier_size_ratio': float(exposure_ratio.min()),
            'maximum_circuit_tier_size_ratio': float(exposure_ratio.max()),
            'part_comparison': by_part,
        }
        if task == 'circuit_log':
            rows = rows.copy()
            depth = rows.depth_to_output.astype(int)
            rows['bucket'] = rows.node_type + '-' + np.where(
                depth >= 3, 'early', np.where(depth == 2, 'middle', 'late'))
            bucket = rows.groupby(['part_name', 'bucket'])[[
                'raw_log2_count_ratio_reference', 'log2_presence_enrichment'
            ]].mean()
            changed = bucket[bucket.iloc[:, 0] * bucket.iloc[:, 1] < 0]
            comparison[task]['table_s3'] = {
                'n_cells': len(bucket),
                'sign_changes': len(changed),
                'maximum_absolute_change': float((bucket.iloc[:, 0] - bucket.iloc[:, 1]).abs().max()),
                'changed_sign_cells': changed.reset_index().to_dict('records'),
                'all_cells': bucket.reset_index().to_dict('records'),
            }
    return comparison


def build_presence_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    denominators, proof = recover_context_denominators(input_dir)
    source = pd.read_csv(input_dir / 'l2_enrichment.csv')
    require(source.shape[0] == 20 * len(denominators),
            'Expected all 20 parts in every task/context combination.')
    require(not source.duplicated(['task', 'fp_key', 'part_idx']).any(),
            'Duplicate source part/context rows.')
    frame = source.rename(columns={
        'top_count': 'top_part_design_count',
        'bot_count': 'bottom_part_design_count',
        'total_designs': 'part_tier_support',
        'log_odds': 'raw_log2_count_ratio_reference',
    }).merge(denominators.rename(columns={
        'top_circuits': 'top_design_count',
        'bottom_circuits': 'bottom_design_count',
        'topology_count': 'recovered_topology_count',
    }), on=['task', 'fp_key'], how='left', validate='many_to_one', sort=False)
    require(frame.top_design_count.notna().all() and
            frame.bottom_design_count.notna().all(),
            'A raw count row has no recovered denominator.')
    require(frame.n_topologies.eq(frame.recovered_topology_count).all(),
            'Recovered topology membership differs from source metadata.')
    require(frame.part_tier_support.eq(
        frame.top_part_design_count + frame.bottom_part_design_count).all(),
        'Support filter field differs from original summed part counts.')
    measured = presence_statistics(
        frame.top_part_design_count.to_numpy(),
        frame.bottom_part_design_count.to_numpy(),
        frame.top_design_count.to_numpy(),
        frame.bottom_design_count.to_numpy())
    for name, values in measured.items():
        frame[name] = values
    require(np.isfinite(frame.log2_presence_enrichment).all(),
            'Nonfinite migrated enrichment values.')
    frame = frame.drop(columns='recovered_topology_count')
    proof.update({
        **PRESENCE_METADATA,
        'metric': PRESENCE_METADATA,
        'weighting': 'pooled_design_presence_separately_within_each_tier',
        'sampled_rows_deduplicated': False,
        'tier_cutoffs': {'top_quantile': 0.95, 'bottom_quantile': 0.05},
        'output_rows': len(frame),
        'selection_policy': 'Original per-topology inclusive >=q95 and <=q05 masks, with all cutoff ties retained.',
        'counting_unit': 'Circuit design containing the part in at least one slot with this structural context.',
        'denominator_description': 'Circuit designs containing this context, counted once even when multiple slots share it.',
        'part_presence_basis': 'Family-unique gate assignments prevent the same library part occurring twice in one design; original per-part counts therefore equal binary circuit-presence counts within each context.',
        'context_policy': 'Keep context-specific dots and original context eligibility; do not pool all contexts into one value per part.',
        'support_filter': {'n_topologies_min': 5, 'n_targets_min': 3,
                           'part_tier_support_min': 1000},
        'comparison_with_original_count_metric': compare_with_count_metric(frame),
    })
    require(all(hashlib.sha256((input_dir / name).read_bytes()).hexdigest() == digest
                for name, digest in proof['input_sha256'].items()),
            'Source inputs changed during recovery; refusing mixed-version output.')
    return frame, denominators, proof


def save_data(input_dir: Path, output_dir: Path) -> dict:
    frame, denominators, proof = build_presence_data(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        'l2_presence_enrichment_v2.csv': frame,
        'l2_presence_denominators_v2.csv': denominators,
    }
    hashes = {}
    for name, data in outputs.items():
        destination = output_dir / name
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        data.to_csv(temporary, index=False)
        hashes[name] = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(destination)
    proof['output_sha256'] = hashes
    destination = output_dir / 'l2_presence_summary_v2.json'
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(proof, indent=2) + '\n')
    temporary.replace(destination)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=VF_ROOT / 'data')
    args = parser.parse_args()
    report = save_data(args.input_dir, args.output_dir)
    print(json.dumps({
        'schema': report['schema'], 'output_rows': report['output_rows'],
        'exact_checks': report['exact_checks'],
        'task_diagnostics': {
            task: {key: value for key, value in details.items()
                   if key not in ['part_comparison', 'table_s3']}
            for task, details in report['comparison_with_original_count_metric'].items()
        },
    }, indent=2))
