
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_PENALTY = -2

VALID_DNA = set("ACGTNRYKMSWBDHV")


# --------------------------------------------------
# 2. READ AND VALIDATE FASTA FILES
# --------------------------------------------------

def read_fasta(file_path):
    """Read a single DNA sequence from a FASTA file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {file_path}\n"
            "Check the filename and data folder."
        )

    header = None
    sequence_parts = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    raise ValueError(
                        "This program expects one FASTA record per file."
                    )
                header = line[1:]
            else:
                if header is None:
                    raise ValueError(
                        "FASTA sequence must start with a > header."
                    )
                sequence_parts.append(line.upper())

    if header is None or not sequence_parts:
        raise ValueError(f"Empty or invalid FASTA file: {file_path}")

    sequence = "".join(sequence_parts)

    invalid_bases = set(sequence) - VALID_DNA
    if invalid_bases:
        raise ValueError(
            f"Invalid DNA characters in {file_path.name}: "
            f"{sorted(invalid_bases)}"
        )

    if not any(base in "ACGT" for base in sequence):
        raise ValueError(
            f"No unambiguous DNA bases found in {file_path.name}"
        )

    return header, sequence


# --------------------------------------------------
# 3. SMITH-WATERMAN MATRIX CONSTRUCTION
# --------------------------------------------------

def smith_waterman(seq1, seq2):
    rows = len(seq1) + 1
    cols = len(seq2) + 1

    # H stores alignment scores.
    H = np.zeros((rows, cols), dtype=int)

    # P stores traceback directions:
    # 0 = stop, 1 = diagonal, 2 = up, 3 = left
    P = np.zeros((rows, cols), dtype=np.int8)

    max_score = 0
    max_i = 0
    max_j = 0

    for i in range(1, rows):
        for j in range(1, cols):

            if seq1[i - 1] == seq2[j - 1]:
                substitution_score = MATCH_SCORE
            else:
                substitution_score = MISMATCH_SCORE

            diagonal = (
                H[i - 1, j - 1] + substitution_score
            )
            up = H[i - 1, j] + GAP_PENALTY
            left = H[i, j - 1] + GAP_PENALTY

            # Local alignment cannot have a negative score.
            candidates = [
                (0, 0),
                (diagonal, 1),
                (up, 2),
                (left, 3)
            ]

            best_score, direction = max(
                candidates, key=lambda item: item[0]
            )

            H[i, j] = best_score
            P[i, j] = direction

            if best_score > max_score:
                max_score = best_score
                max_i = i
                max_j = j

    return H, P, max_score, max_i, max_j


# --------------------------------------------------
# 4. TRACEBACK AND LOCAL ALIGNMENT
# --------------------------------------------------

def traceback(seq1, seq2, H, P, max_score, max_i, max_j):

    i = max_i
    j = max_j

    aligned1 = []
    aligned2 = []
    alignment_columns = []

    while i > 0 and j > 0 and H[i, j] > 0:

        direction = P[i, j]

        if direction == 1:
            a = seq1[i - 1]
            b = seq2[j - 1]

            aligned1.append(a)
            aligned2.append(b)

            alignment_columns.append(
                (a, b, i - 1, j - 1)
            )

            i -= 1
            j -= 1

        elif direction == 2:
            a = seq1[i - 1]

            aligned1.append(a)
            aligned2.append("-")

            alignment_columns.append(
                (a, "-", i - 1, None)
            )

            i -= 1

        elif direction == 3:
            b = seq2[j - 1]

            aligned1.append("-")
            aligned2.append(b)

            alignment_columns.append(
                ("-", b, None, j - 1)
            )

            j -= 1

        else:
            break

    aligned1.reverse()
    aligned2.reverse()
    alignment_columns.reverse()

    alignment1 = "".join(aligned1)
    alignment2 = "".join(aligned2)

    # Coordinates are 1-based, as commonly reported
    # in biological sequence analysis.
    start1 = i + 1
    start2 = j + 1
    end1 = max_i
    end2 = max_j

    return (
        alignment1,
        alignment2,
        alignment_columns,
        start1,
        end1,
        start2,
        end2
    )


# --------------------------------------------------
# 5. IDENTIFY EXACT CONSERVED REGIONS
# --------------------------------------------------

def find_conserved_regions(columns):

    regions = []
    current = []

    for a, b, pos1, pos2 in columns:

        # Count only unambiguous, exact DNA matches.
        is_match = (
            a == b
            and a in "ACGT"
            and pos1 is not None
            and pos2 is not None
        )

        if not is_match:
            if current:
                regions.append(current)
                current = []
            continue

        if current:
            previous = current[-1]

            consecutive = (
                pos1 == previous[2] + 1
                and pos2 == previous[3] + 1
            )

            if not consecutive:
                regions.append(current)
                current = []

        current.append((a, b, pos1, pos2))

    if current:
        regions.append(current)

    return [
        {
            "sequence": "".join(item[0] for item in region),
            "start1": region[0][2] + 1,
            "end1": region[-1][2] + 1,
            "start2": region[0][3] + 1,
            "end2": region[-1][3] + 1,
            "length": len(region)
        }
        for region in regions
    ]


# --------------------------------------------------
# 6. VISUALIZE AND SAVE THE SCORING MATRIX
# --------------------------------------------------

def visualize_matrix(H, max_i, max_j):

    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.imshow(
        H[1:, 1:],
        cmap="viridis",
        origin="upper",
        aspect="auto",
        interpolation="nearest"
    )

    # Highlight the maximum-scoring cell.
    if max_i > 0 and max_j > 0:
        ax.add_patch(
            Rectangle(
                (max_j - 2, max_i - 2),
                1, 1,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
        )

    ax.set_title("Smith-Waterman Scoring Matrix")
    ax.set_xlabel("Sequence 2 position")
    ax.set_ylabel("Sequence 1 position")

    ax.set_xlim(-0.5, H.shape[1] - 1.5)
    ax.set_ylim(H.shape[0] - 2 + 0.5, -0.5)

    fig.colorbar(image, ax=ax, label="Alignment score")
    fig.tight_layout()

    output_path = RESULTS_DIR / "alignment_matrix.png"
    fig.savefig(output_path, dpi=200)
    print(f"\nMatrix image saved: {output_path.name}")

    plt.show()
    plt.close(fig)


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 60)
    print("SMITH-WATERMAN LOCAL ALIGNMENT")
    print("=" * 60)

    file1 = DATA_DIR / "human_mtDNA.fasta"
    file2 = DATA_DIR / "chimp_mtDNA.fasta"

    try:
        header1, seq1 = read_fasta(file1)
        header2, seq2 = read_fasta(file2)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nInput error: {error}")
        return

    print("\nINPUT VALIDATION")
    print(f"Sequence 1: {header1}")
    print(f"Sequence 1 length: {len(seq1)} bases")
    print(f"Sequence 2: {header2}")
    print(f"Sequence 2 length: {len(seq2)} bases")

    print("\nSCORING PARAMETERS")
    print(f"Match: {MATCH_SCORE}")
    print(f"Mismatch: {MISMATCH_SCORE}")
    print(f"Gap: {GAP_PENALTY}")

    H, P, score, max_i, max_j = smith_waterman(seq1, seq2)

    print("\nMATRIX CONSTRUCTION")
    print(f"Matrix dimensions: {H.shape[0]} x {H.shape[1]}")
    print(f"Maximum alignment score: {score}")

    # Save the numeric matrix for inspection.
    np.savetxt(
        RESULTS_DIR / "scoring_matrix.csv",
        H,
        delimiter=",",
        fmt="%d"
    )
    print("Numeric matrix saved: scoring_matrix.csv")

    (
        alignment1,
        alignment2,
        columns,
        start1,
        end1,
        start2,
        end2
    ) = traceback(seq1, seq2, H, P, score, max_i, max_j)

    if score == 0:
        print("\nNo positive-scoring local alignment found.")
        return

    matches = sum(
        a == b and a in "ACGT"
        for a, b in zip(alignment1, alignment2)
    )
        # STEP 3: Validate traceback and alignment consistency
    if len(alignment1) != len(alignment2):
        raise ValueError("Alignment strings have different lengths.")

    consumed1 = sum(base != "-" for base in alignment1)
    consumed2 = sum(base != "-" for base in alignment2)

    expected1 = end1 - start1 + 1
    expected2 = end2 - start2 + 1

    print("\nSTEP 3: ALIGNMENT VALIDATION")
    print(f"Alignment strings have equal length: {len(alignment1) == len(alignment2)}")
    print(f"Alignment columns: {len(alignment1)}")
    print(f"Sequence 1 bases consumed: {consumed1}")
    print(f"Sequence 2 bases consumed: {consumed2}")
    print(f"Sequence 1 coordinate span: {expected1}")
    print(f"Sequence 2 coordinate span: {expected2}")

    if consumed1 != expected1 or consumed2 != expected2:
        raise ValueError(
            "Traceback coordinates do not match the aligned sequences."
        )

    print("Traceback coordinate validation: PASSED") 
    alignment_length = len(alignment1)
    identity = 100 * matches / alignment_length

    print("\nOPTIMAL LOCAL ALIGNMENT")
    print(f"Sequence 1 positions: {start1}-{end1}")
    print(f"Sequence 2 positions: {start2}-{end2}")
    print(f"Alignment score: {score}")
    print(f"Alignment length: {alignment_length}")
    print(f"Exact matches: {matches}")
    print(f"Percentage identity: {identity:.2f}%")
    print(f"\nSequence 1: {alignment1}")
    print(f"            {''.join('|' if a == b and a in 'ACGT' else ' ' for a, b in zip(alignment1, alignment2))}")
    print(f"Sequence 2: {alignment2}")

    regions = find_conserved_regions(columns)

    print("\nEXACT CONSERVED REGIONS IN THE ALIGNMENT")

    if not regions:
        print("No exact conserved region found.")
    else:
        for number, region in enumerate(regions, start=1):
            print(
                f"{number}. {region['sequence']} | "
                f"Length: {region['length']} | "
                f"Sequence 1: {region['start1']}-{region['end1']} | "
                f"Sequence 2: {region['start2']}-{region['end2']}"
            )

    # Write a report for your RTE.
    report_path = RESULTS_DIR / "alignment_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("SMITH-WATERMAN ALIGNMENT REPORT\n")
        report.write("=" * 40 + "\n")
        report.write(f"Sequence 1: {header1}\n")
        report.write(f"Sequence 2: {header2}\n")
        report.write(f"Length 1: {len(seq1)}\n")
        report.write(f"Length 2: {len(seq2)}\n")
        report.write(f"Scoring matrix: {H.shape}\n")
        report.write(f"Maximum score: {score}\n")
        report.write(f"Sequence 1 positions: {start1}-{end1}\n")
        report.write(f"Sequence 2 positions: {start2}-{end2}\n")
        report.write(f"Alignment length: {alignment_length}\n")
        report.write(f"Exact matches: {matches}\n")
        report.write(f"Percentage identity: {identity:.2f}%\n\n")
        report.write(f"Sequence 1: {alignment1}\n")
        report.write(f"Sequence 2: {alignment2}\n\n")
        report.write("Exact conserved regions:\n")

        for region in regions:
            report.write(
                f"{region['sequence']} | "
                f"Length: {region['length']} | "
                f"Seq1: {region['start1']}-{region['end1']} | "
                f"Seq2: {region['start2']}-{region['end2']}\n"
            )

    print(f"\nReport saved: {report_path.name}")

    visualize_matrix(H, max_i, max_j)

    print("\nPROJECT STEP 2 COMPLETED")


if __name__ == "__main__":
    main()
