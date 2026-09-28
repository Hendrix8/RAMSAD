from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    from ramsad.embed import run_embed

    run_embed(cfg)


if __name__ == "__main__":
    main()
