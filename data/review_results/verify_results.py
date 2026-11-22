"""Recalculate reported numerical results from the supplied prediction records."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
MODELS = {'gnn_transformer': 'AlphaApta D2', 'clip': 'CLIP', 'mamba': 'Mamba',
          'evoformer': 'Evoformer', 'concat_mlp': 'Concat-MLP',
          'film_gmu': 'FiLM-GMU', 'coattn': 'Co-Attention', 'qformer': 'Q-Former'}
SOURCES = ['AGS', 'GES-1', 'MKN-28', 'NCI-N87', 'MKN-1', 'CRL-3621']


def main():
    records = []
    for source in SOURCES:
        reference = None
        for arch, model in MODELS.items():
            loss = 'infonce_main' if arch == 'clip' else 'ce_main'
            path = ROOT / 'LOCLO/predictions' / f'{arch}_{loss}_cellsource_loco_{source}_s42_predictions.npz'
            with np.load(path, allow_pickle=False) as values:
                ids, labels, scores = (values[x] for x in ('selected_event_indices', 'labels', 'scores'))
                if reference is None:
                    reference = (ids.copy(), labels.copy())
                else:
                    assert np.array_equal(ids, reference[0])
                    assert np.array_equal(labels, reference[1])
                records.append({'model': model, 'source': source, 'n_test': len(labels),
                                'roc_auc': roc_auc_score(labels, scores),
                                'pr_auc': average_precision_score(labels, scores)})
    calculated = pd.DataFrame(records).set_index(['model', 'source']).sort_index()
    expected = pd.read_csv(ROOT / 'LOCLO/LOCLO_verified_48.csv').set_index(['model', 'source']).sort_index()
    assert np.allclose(calculated[['roc_auc', 'pr_auc']], expected[['roc_auc', 'pr_auc']], rtol=0, atol=1e-12)
    assert np.array_equal(calculated.n_test, expected.n_test)


    summary = pd.read_csv(ROOT / "LOCLO/LOCLO_verified_summary.csv").set_index("model")
    per_cell = pd.read_csv(ROOT / "LOCLO/D2_LOCLO_per_fold_paired_cell_bootstrap.csv")
    for model, rows in calculated.reset_index().groupby("model"):
        for attr, field in [("roc_auc", "roc"), ("pr_auc", "pr")]:
            assert abs(np.average(rows[attr], weights=rows.n_test) - summary.loc[model, field + "_event_weighted"]) < 1e-12
            assert abs(rows[attr].mean() - summary.loc[model, field + "_macro"]) < 1e-12
    assert len(per_cell) == 42
    with np.load(ROOT / "LOCLO/paired_cell_ROC_draws.npz", allow_pickle=False) as draws:
        for r in per_cell.itertuples(index=False):
            reference = MODELS[r.reference]
            delta = (calculated.loc[("AlphaApta D2", r.source), "roc_auc"]
                     - calculated.loc[(reference, r.source), "roc_auc"])
            assert abs(delta - r.D2_minus_reference) < 1e-12
            sample = draws[r.source + "_gnn_transformer"] - draws[r.source + "_" + r.reference]
            assert sample.shape == (1000,)
            assert np.allclose(np.quantile(sample, [.025, .975]),
                [r.cell_cluster_percentile95_low, r.cell_cluster_percentile95_high], rtol=0, atol=1e-12)

    values = pd.read_csv(ROOT / 'R3_8_100_sequences/scores_100_five_contexts.csv')
    assert values.source_record.nunique() == 100
    expected = pd.read_csv(ROOT / 'R3_8_100_sequences/correlation_top20_summary.csv').set_index('condition')
    score_columns = [c for c in values if c.startswith('score_')]
    calculated_correlations = []
    abundance_top = set(values.sort_values(['R15_frequency', 'source_record'], ascending=[False, True]).head(20).source_record)
    for column in score_columns:
        condition = column.removeprefix('score_')
        statistic = spearmanr(values[column], values.R15_frequency)
        model_top = set(values.sort_values([column, 'source_record'], ascending=[False, True]).head(20).source_record)
        overlap = len(abundance_top & model_top)
        saved = expected.loc[condition]
        assert abs(statistic.statistic - saved.spearman_rho) < 1e-12
        assert abs(statistic.pvalue - saved.two_sided_p) < 1e-12
        assert overlap == saved.top20_overlap_n
        assert len(values) == saved.n_sequences
        calculated_correlations.append({'condition': condition, 'n': len(values), 'rho': statistic.statistic,
                                       'two_sided_p': statistic.pvalue, 'top20_overlap': overlap})
    result = {'LOCLO_prediction_sets_verified': len(records),
              'R3_8_five_context_records_verified': int(len(values)),
              'correlations': calculated_correlations}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
