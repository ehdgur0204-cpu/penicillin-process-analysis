from pathlib import Path
import random


# Sampling settings
TARGET_ROWS = 1_000
RANDOM_SEED = 42

# Resolve paths from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "100_Batches_IndPenSim_V3.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "100_Batches_IndPenSim_V3_sample_1000.csv"
)


def count_data_rows(csv_path: Path) -> int:
    """Count CSV data rows without loading the entire file into memory."""
    with csv_path.open("rb") as csv_file:
        header = csv_file.readline()
        if not header:
            raise ValueError(f"The CSV file is empty: {csv_path}")

        return sum(1 for _ in csv_file)


def sample_csv(
    input_path: Path,
    output_path: Path,
    target_rows: int,
    random_seed: int,
) -> None:
    """Randomly sample exactly target_rows and preserve the original header."""
    total_rows = count_data_rows(input_path)

    if total_rows < target_rows:
        raise ValueError(
            f"The source contains {total_rows:,} rows, "
            f"which is less than the requested {target_rows:,} rows."
        )

    random_generator = random.Random(random_seed)
    selected_indices = set(
        random_generator.sample(range(total_rows), target_rows)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as source, output_path.open("wb") as destination:
        destination.write(source.readline())

        for row_index, row in enumerate(source):
            if row_index in selected_indices:
                destination.write(row)

    print(f"Source rows: {total_rows:,}")
    print(f"Sampled rows: {target_rows:,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    sample_csv(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        target_rows=TARGET_ROWS,
        random_seed=RANDOM_SEED,
    )
