from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ramsad.retrieve import run_retrieve


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    out = Path(HydraConfig.get().runtime.output_dir)
    run_retrieve(cfg, out)


if __name__ == "__main__":
    main()
