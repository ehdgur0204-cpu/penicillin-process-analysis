from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from outputs.strate_comparison.strategy_comparison import (  # noqa: E402
    build_outputs,
)


def test_build_outputs_creates_expected_files(tmp_path):
    data_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    pd.DataFrame(
        {
            "Batch_ID": [1, 31, 61, 91],
            "Time (h)": [10.0, 11.0, 12.0, 13.0],
            "Penicllin_yield_total (kg)": [100.0, 110.0, 120.0, 90.0],
            "Penicillin concentration(P:g/L)": [10.0, 11.0, 12.0, 9.0],
        }
    ).to_csv(data_path, index=False)

    html_path, png_path = build_outputs(
        output_dir=output_dir,
        data_path=data_path,
    )

    assert html_path.exists()
    assert png_path.exists()
    assert (output_dir / "batch_summary.csv").exists()
