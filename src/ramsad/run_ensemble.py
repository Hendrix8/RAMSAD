from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ramsad.ensemble import run_ensemble


@hydra.main(version_base=None, config_path="conf", config_name="ensemble_config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    e = cfg.ensemble
    if e.selection_csv in (None, "", "???"):
        raise ValueError("Set ensemble.selection_csv=/path/to/model_selection.csv (from ramsad-select with select.n=N)")
    decimals = e.get("round_decimals")
    run_ensemble(
        selection_csv=Path(to_absolute_path(str(e.selection_csv))),
        series_dirs=[Path(to_absolute_path(str(d))) for d in e.series_dirs],
        scores_dir=Path(to_absolute_path(str(e.scores_dir))),
        out_dir=Path(HydraConfig.get().runtime.output_dir),
        decimals=None if decimals is None else int(decimals),
        run_missing=bool(e.run_missing),
        evaluate=bool(e.evaluate),
        vus_thresholds=int(e.vus_thresholds),
        vus_window_factor=float(e.vus_window_factor),
        save_scores=bool(e.save_scores),
        verbose=bool(e.get("verbose", True)),
    )


if __name__ == "__main__":
    main()
