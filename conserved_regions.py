
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "results" / "alignment_report.txt"
OUTPUT_FILE = BASE_DIR / "results" / "conserved_regions.png"


def main():
    if not REPORT_FILE.exists():
        print("Error: alignment_report.txt not found.")
        print("Run python smith_waterman.py first.")
        return

    report = REPORT_FILE.read_text(encoding="utf-8")

    regions = []

    in_regions = False
    for line in report.splitlines():
        if line.strip() == "Exact conserved regions:":
            in_regions = True
            continue

        if in_regions and "|" in line:
            parts = [part.strip() for part in line.split("|")]
            sequence = parts[0]

            if "Length:" not in line:
                continue

            length = int(parts[1].replace("Length:", "").strip())
            regions.append((sequence, length))

    if not regions:
        print("No conserved regions found in the report.")
        return

    labels = [
        f"{sequence} (length {length})"
        for sequence, length in regions
    ]

    lengths = [length for _, length in regions]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_positions = list(range(len(regions)))

    ax.barh(y_positions, lengths)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Conserved region length (bases)")
    ax.set_ylabel("Exact matching DNA region")
    ax.set_title("Exact Conserved Regions in Local Alignment")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.show()
    plt.close(fig)

    print(f"Conserved regions plotted: {len(regions)}")
    print(f"Chart saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
