from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from ramsad.embed_desc import run_embed_desc


@hydra.main(version_base=None, config_path="conf", config_name="embed_desc")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    run_embed_desc(cfg)


if __name__ == "__main__":
    main()
