from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_embed_parquet_desc_writes_column(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("sentence_transformers")
    import sentence_transformers

    from ramsad.embed_desc import embed_parquet_desc

    class FakeModel:
        def encode(self, texts, **kwargs):
            n = len(texts)
            return np.arange(n * 768, dtype=np.float32).reshape(n, 768)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *a, **k: FakeModel())

    pq = tmp_path / "seg.parquet"
    pd.DataFrame({"desc": ["a", "bb"]}).to_parquet(pq, index=False)

    embed_parquet_desc(
        pq,
        model_id="fake",
        text_column="desc",
        out_column="all-mpnet-base-v2_desc",
        batch_size=8,
        device=None,
        skip_if_column_present=False,
    )
    out = pd.read_parquet(pq)
    assert "all-mpnet-base-v2_desc" in out.columns
    assert len(out.iloc[0]["all-mpnet-base-v2_desc"]) == 768
