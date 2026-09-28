from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ramsad.embed import run_embed
from ramsad.embed_desc import run_embed_desc
from ramsad.retrieve import run_retrieve
from ramsad.select import run_select


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    out = Path(HydraConfig.get().runtime.output_dir)
    print(f"[pipeline] output_dir={out}")

    if bool(cfg.pipeline.get("embed_desc", False)):
        print("[pipeline] embed_desc=true (sentence-transformer text embeddings on segment parquets)")
        run_embed_desc(cfg)

    if not cfg.pipeline.skip_embed:
        run_embed(cfg)
    else:
        print("[pipeline] skip_embed=true, assuming parquets already contain embeddings")

    ret_path = run_retrieve(cfg, out)
    run_select(cfg, out, retrieval_path=ret_path)
    print("[pipeline] done.")


if __name__ == "__main__":
    main()
