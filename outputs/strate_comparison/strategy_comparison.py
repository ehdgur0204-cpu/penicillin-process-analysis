"""RC / OC / APC / Fault 그룹별 성능 비교 차트를 생성한다."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "interim" / "merged_data.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "strate_comparison"

STRATEGY_ORDER = ["RC", "OC", "APC", "Fault"]
COLORS = {
    "RC": "#5DCAA5",
    "OC": "#7F77DD",
    "APC": "#378ADD",
    "Fault": "#E24B4A",
}
METRICS = ["total_yield", "final_conc", "productivity"]


def assign_strategy(batch_id: int) -> str:
    """Batch_ID 범위에 따라 제어 전략을 반환한다."""
    if 1 <= batch_id <= 30:
        return "RC"
    if 31 <= batch_id <= 60:
        return "OC"
    if 61 <= batch_id <= 90:
        return "APC"
    if 91 <= batch_id <= 100:
        return "Fault"
    return "Unknown"


def build_batch_summary(data_path: Path | str) -> pd.DataFrame:
    """원본 시계열 데이터를 배치당 한 행의 KPI 데이터로 요약한다."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"입력 데이터를 찾을 수 없습니다: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {
        "Batch_ID",
        "Time (h)",
        "Penicllin_yield_total (kg)",
        "Penicillin concentration(P:g/L)",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing_columns)}")

    df = df.copy()
    df["Strategy"] = df["Batch_ID"].apply(assign_strategy)

    batch_summary = (
        df.sort_values("Time (h)")
        .groupby(["Batch_ID", "Strategy"], as_index=False)
        .agg(
            total_yield=("Penicllin_yield_total (kg)", "max"),
            duration_h=("Time (h)", "max"),
            final_conc=("Penicillin concentration(P:g/L)", "last"),
        )
        .sort_values("Batch_ID")
        .reset_index(drop=True)
    )
    batch_summary["productivity"] = (
        batch_summary["total_yield"] / batch_summary["duration_h"]
    )
    return batch_summary


def _build_interactive_figure(batch_summary: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "총회수량 (kg)",
            "최종 페니실린 농도 (g/L)",
            "시간당 생산성 (kg/h)",
        ),
    )

    for column, metric in enumerate(METRICS, start=1):
        for strategy in STRATEGY_ORDER:
            subset = batch_summary[batch_summary["Strategy"] == strategy]
            figure.add_trace(
                go.Box(
                    y=subset[metric],
                    name=strategy,
                    marker_color=COLORS[strategy],
                    showlegend=(column == 1),
                    legendgroup=strategy,
                ),
                row=1,
                col=column,
            )

    figure.update_layout(
        title="제어전략(RC/OC/APC/Fault)별 성과 비교",
        height=450,
        width=1100,
        boxmode="group",
    )
    return figure


def _write_static_figure(batch_summary: pd.DataFrame, png_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_titles = {
        "total_yield": "Total yield (kg)",
        "final_conc": "Final penicillin conc. (g/L)",
        "productivity": "Productivity (kg/h)",
    }

    for axis, metric in zip(axes, METRICS):
        data_by_group = [
            batch_summary[batch_summary["Strategy"] == strategy][metric].dropna()
            for strategy in STRATEGY_ORDER
        ]
        boxplot = axis.boxplot(
            data_by_group,
            tick_labels=STRATEGY_ORDER,
            patch_artist=True,
        )
        for patch, strategy in zip(boxplot["boxes"], STRATEGY_ORDER):
            patch.set_facecolor(COLORS[strategy])
            patch.set_alpha(0.6)
        axis.set_title(metric_titles[metric])
        axis.grid(axis="y", alpha=0.3)

    figure.suptitle("Strategy comparison: RC / OC / APC / Fault", fontsize=14)
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)
    plt.close(figure)


def build_outputs(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> tuple[Path, Path]:
    """요약 CSV와 HTML·PNG 전략 비교 차트를 생성하고 차트 경로를 반환한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_summary = build_batch_summary(data_path)
    batch_summary.to_csv(output_dir / "batch_summary.csv", index=False)

    html_path = output_dir / "strategy_comparison.html"
    png_path = output_dir / "strategy_comparison.png"

    _build_interactive_figure(batch_summary).write_html(html_path)
    _write_static_figure(batch_summary, png_path)
    return html_path, png_path


def main() -> None:
    html_path, png_path = build_outputs()
    print(f"HTML saved to: {html_path}")
    print(f"PNG saved to: {png_path}")


if __name__ == "__main__":
    main()
