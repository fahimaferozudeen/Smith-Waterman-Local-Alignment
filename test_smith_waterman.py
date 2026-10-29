
import unittest

from smith_waterman import smith_waterman, traceback


class TestSmithWaterman(unittest.TestCase):

    def run_alignment(self, seq1, seq2):
        H, P, score, max_i, max_j = smith_waterman(seq1, seq2)

        result = traceback(
            seq1, seq2, H, P, score, max_i, max_j
        )

        alignment1 = result[0]
        alignment2 = result[1]

        return score, alignment1, alignment2

    # Test 1: Identical sequences
    def test_identical_sequences(self):
        seq1 = "ACGTACGT"
        seq2 = "ACGTACGT"

        score, alignment1, alignment2 = self.run_alignment(
            seq1, seq2
        )

        self.assertEqual(score, 16)
        self.assertEqual(alignment1, seq1)
        self.assertEqual(alignment2, seq2)

    # Test 2: Partial matching sequences
    def test_partial_matching_sequences(self):
        seq1 = "TTACGTAA"
        seq2 = "GGACGTCC"

        score, alignment1, alignment2 = self.run_alignment(
            seq1, seq2
        )

        self.assertGreater(score, 0)
        self.assertEqual(len(alignment1), len(alignment2))
        self.assertIn("ACGT", alignment1)
        self.assertIn("ACGT", alignment2)

    # Test 3: No matching bases
    def test_no_matching_sequences(self):
        seq1 = "AAAA"
        seq2 = "CCCC"

        score, alignment1, alignment2 = self.run_alignment(
            seq1, seq2
        )

        self.assertEqual(score, 0)
        self.assertEqual(alignment1, "")
        self.assertEqual(alignment2, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
