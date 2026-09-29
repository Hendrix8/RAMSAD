from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ramsad.ensemble import combine_scores, run_ensemble, train_index_from_name, tsbad_name


def test_combine_scores_is_mean_of_minmax() -> None:
    a = np.array([0.0, 5.0, 10.0])
    b = np.array([2.0, 2.0, 4.0, 99.0])  # normalised over its full length, then truncated
    out = combine_scores([a, b], decimals=None)
    np.testing.assert_allclose(out, [0.0, 0.25, 0.5 + 1 / 97])


def test_combine_scores_rounding_and_constant_score() -> None:
    out = combine_scores([np.array([1.0, 2.0, 3.0]), np.ones(3)], decimals=3)
    np.testing.assert_allclose(out, [0.0, 0.25, 0.5])


def test_name_helpers() -> None:
    assert tsbad_name("Sub-PCA") == "Sub_PCA"
    assert tsbad_name("MOMENT (FT)") == "MOMENT_FT"
    assert tsbad_name("CNN") == "CNN"
    assert train_index_from_name("002_NAB_id_2_WebService_tr_1500_1st_4106") == 1500


def test_run_ensemble_from_cached_scores(tmp_path: Path) -> None:
    series = "001_X_id_1_Test_tr_2_1st_3"
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"Data": np.arange(6.0), "Label": [0, 0, 0, 1, 1, 0]}).to_csv(raw / f"{series}.csv", index=False)
    scores = tmp_path / "scores"
    for name, s in {"Sub_PCA": np.arange(6.0), "CNN": np.arange(6.0)[::-1]}.items():
        (scores / name).mkdir(parents=True)
        np.save(scores / name / f"{series}.npy", s)
    sel = tmp_path / "model_selection.csv"
    pd.DataFrame(
        {"time_series": [series], "selected_model": ["Sub-PCA"], "selected_models": ["Sub-PCA;CNN;LOF"]}
    ).to_csv(sel, index=False)

    out = run_ensemble(
        selection_csv=sel, series_dirs=[raw], scores_dir=scores, out_dir=tmp_path / "out",
        evaluate=False, verbose=False,
    )
    res = pd.read_csv(out)
    assert res.loc[0, "n_selected"] == 3 and res.loc[0, "n_used"] == 2  # LOF has no cached score
    np.testing.assert_allclose(np.load(tmp_path / "out" / "ensemble_scores" / f"{series}.npy"), 0.5)
