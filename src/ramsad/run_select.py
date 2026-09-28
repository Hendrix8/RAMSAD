from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ramsad.select import run_select


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    out = Path(HydraConfig.get().runtime.output_dir)
    rc_override = cfg.select.get("retrieval_csv")
    if (
        rc_override is not None
        and str(rc_override).strip() not in ("", "~", "null", "None")
    ):
        retrieval = Path(to_absolute_path(str(rc_override)))
    else:
        retrieval = out / str(cfg.retrieve.output_csv_name)
    if not retrieval.is_file():
        raise FileNotFoundError(
            f"Retrieval CSV not found: {retrieval}. Run ramsad-retrieve first or ramsad-pipeline."
        )
    run_select(cfg, out, retrieval_path=retrieval)


if __name__ == "__main__":
    main()
