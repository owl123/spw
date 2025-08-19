#!/usr/bin/env python3
# test_tdict.py
# A simple test program for the `tdict` package.

import os
import tempfile
from pprint import pprint

# If your class is in a file named tdict.py with class tdict inside it:
from tdict import tdict


def write_wordlist(words) -> str:
    """Write a temporary word list file and return its path."""
    fd, path = tempfile.mkstemp(prefix="tdict_words_", suffix=".txt", text=True)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")
    return path


def main():
    # A small, mixed dictionary (uppercase expected by loader; we’ll upper() here)
    words = [
        "APPLE", "APRON", "APOGEE", "APE", "APEX",
        "PEAR", "PEEL", "PEEK", "PEST", "POST", "PAST", "MOSS",
        "BAT", "BET", "BIT", "BOT", "BUT",
        "ZOO", "ZOOLOGY", "QUIET", "QUEUE",
        "ALPHA", "ALPHABET", "ALP",
    ]
    words = [w.upper() for w in words]

    path = write_wordlist(words)
    print(f"[info] temp wordlist at: {path}")

    # available_letters can prune aggressively; pick something lenient first
    available = "AAAABBBBCCDDDDEEEEEEEEFFGGHHIIIJJKKLLMMNNOOPPQQRRRSSTTTTUUVVWWXXYYZZ"

    # Build dictionary with length filter (3..6 letters)
    lex = tdict(path, available_letters=available, len_min=3, len_max=6)

    # ---------- Basic smoke tests ----------
    print("\n[TEST] contains via match()")
    print("AP..E ->", lex.match("AP..E"))  # Expect ['APPLE'] (APRON fails last letter)
    print("P.ST  ->", sorted(lex.match("P.ST")))  # Expect ['PAST','PEST','POST'] intersecting our list
    print("...   ->", sorted(lex.match("...")))    # All 3-letter words allowed by available and length limits
    print("..R.. ->", sorted(lex.match("..R..")))
    print("..P.. ->", sorted(lex.match("..P..")))

    # ---------- Counting tests ----------
    print("\n[TEST] count()")
    print("AP..E ->", lex.count("AP..E"))  # Expect 1
    print("P.ST  ->", lex.count("P.ST"))   # Expect 3 (PAST, PEST, POST)
    print("....  ->", lex.count("...."))   # Count all 4-letter words in our subset
    print("..R.. ->", lex.count("..R.."))
    print("..P.. ->", lex.count("..P.."))

    # ---------- Assert a few invariants ----------
    # Your numbers may vary if you change the word list above.
    # These asserts are tailored to the list we wrote.
    got_ap = set(lex.match("AP..E"))
    assert "APPLE" in got_ap and "APRON" not in got_ap, "AP..E should match APPLE but not APRON"
    assert lex.count("AP..E") == 1, "AP..E should count to 1"

    got_ap = set(lex.match("..R.."))
    assert "APPLE" not in got_ap and "ALPHA" not in got_ap and "APRON" in got_ap, "..R.. should match APRON but not APPLE and ALPHA"
    assert lex.count("..R..") == 1, "..R.. should count to 1"

    got_ap = set(lex.match("..P.."))
    assert "APPLE" in got_ap and "ALPHA" in got_ap and "APRON" not in got_ap, "..P.. should match ALPHA and APPLE, but not APRON"
    assert lex.count("..P..") == 2, "..P.. should count to 2"


    got_pst = set(lex.match("P.ST"))
    expect_pst = {"PAST", "PEST", "POST"}
    assert expect_pst.issubset(got_pst), "P.ST should include PAST/PEST/POST"
    assert lex.count("P.ST") >= 3, "P.ST count should be at least 3"

    # Check length-partitioning works (no 7+ letters since len_max=6)
    assert lex.match(".......") == [], "No 7-letter matches due to len_max=6"

    print("\n[OK] basic assertions passed.")

    # ---------- Demo of tighter available_letters ----------
    # Only letters that can form PEEL/PEEK
    tight_available = "PEEKL"
    # We provide 1 of each; that allows: P, E, E, L/K (4 letters max)
    # (Note: multiplicity matters; to allow both PEEL and PEEK simultaneously,
    # you’d need enough E’s and both L and K.)
    lex_tight = tdict(path, available_letters=tight_available, len_min=4, len_max=4)
    print("\n[TEST] with tight available letters:", tight_available)
    print("PEE. ->", sorted(lex_tight.match("PEE.")))  # Expect subset of {'PEEL','PEEK'} based on counts

    # ---------- Print a small summary ----------
    def count_by_len(d):
        return {L: len(ws) for L, ws in d.words.items()}

    print("\n[SUMMARY] words loaded by length (len_min=3, len_max=6):")
    pprint(count_by_len(lex))

    print("\nDone.")


if __name__ == "__main__":
    main()