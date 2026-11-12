# Reproducing the paper

All numbers come from the released code and the Zenodo data. The scripts take their data location
from `--data` / `--mm` arguments or the `JUMC_DATA` / `JUMC_MV` environment variables.

```bash
export JUMC_DATA=/path/to/ju_m3c            # the Zenodo download
export JUMC_MV=/path/to/ju_m3c/multimodal   # figures, mesh, etc.
pip install -e .[benchmark]
```

| Paper item | Command |
|---|---|
| Corpus construction, splits | `python -m ju_m3c.corpus.normalize`, `dedup`, `make_splits`, `build_tasks` |
| Detector validation (Table 3) | `python scripts/audit/id_validation.py --data $JUMC_DATA` |
| Cross-benchmark contamination (Table / Fig) | `python scripts/audit/contamination_map.py --data $JUMC_DATA` |
| Rank flip (Table / Fig) | `python scripts/audit/ranking_flip.py` |
| Fine-tuning leakage effect (Fig) | `python scripts/audit/leakage_experiment.py` |
| Multi-layer network, link prediction | `python -m ju_m3c.network.multilayer`, `link_prediction` |
| Similar-patient retrieval | `python -m ju_m3c.network.similar_cases` |
| Benchmark runs / scores | `scripts/benchmark/run_bench.py`, `score_bench.py`, `analyze_bench.py` |
| Reliability, prompt ablation, few-shot, 9B | `scripts/experiments/*` |
| Image tasks | `scripts/mm/*` |

The audit tool itself (`jumc audit`) needs no data and runs on `tests/fixtures/`.
