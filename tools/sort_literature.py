"""Sort downloaded papers into Tespit / Sınıflandırma / Karakterizasyon / Derin Öğrenme.

Run it on the computer that has the PDFs.  It does a DRY RUN by default (only prints what it
would move); add --apply to really move the files.  It never deletes or overwrites anything.

    pip install pypdf
    python sort_literature.py                      # dry run on ~/Desktop/Lessons/EEE493/Literature
    python sort_literature.py --apply
    python sort_literature.py "D:\\path\\to\\Literature" --apply

How a file is matched: the title (or arXiv id / IEEE document number) of each known paper is looked
for in the file name and in the text of the first two pages.  Files that match nothing, or match
papers from two different folders, are left where they are and listed at the end.

Rule used for the folders: classical methods go to Tespit, Sınıflandırma or Karakterizasyon;
learning-based papers go to Derin Öğrenme.  Edit RULES below to change an assignment.
"""
from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from pathlib import Path

DEFAULT_ROOT = Path.home() / "Desktop" / "Lessons" / "EEE493" / "Literature"

# folder -> phrases (lower case; punctuation and hyphens do not matter).  One phrase per paper variant.
RULES = {
    "Tespit": [
        "energy detection of unknown deterministic signals",             # Urkowitz 1967
        "survey of spectrum sensing algorithms for cognitive radio",      # Yucek & Arslan 2009
        "adaptive detection mode with threshold control",                 # Finn & Johnson 1968
        "radar cfar thresholding in clutter and multiple target",         # Rohling 1983
        "exploitation of spectral redundancy in cyclostationary signals", # Gardner 1991
    ],
    "Sınıflandırma": [
        "maximum likelihood classification for digital amplitude phase modulations",  # Wei & Mendel 2000
        "likelihood based approach to modulation classification",                    # Hameed et al. 2009
        "automatic identification of digital modulation types",                      # Azzouz & Nandi 1995
        "hierarchical digital modulation classification using cumulants",            # Swami & Sadler 2000
        "survey of automatic modulation classification techniques",                  # Dobre et al. 2007
    ],
    "Karakterizasyon": [
        "comparison of snr estimation techniques for the awgn channel",              # Pauluzzi & Beaulieu 2000
        "handbook on spectrum monitoring",                                           # ITU-R 2011
        "spectrum monitoring handbook",
        "use of fast fourier transform for the estimation of power spectra",         # Welch 1967
    ],
    "Derin Öğrenme": [
        "convolutional radio modulation recognition networks", "1602 04105",         # O'Shea et al. 2016
        "deep neural network architectures for modulation classification", "1712 00443",  # Liu et al. 2017
        "deep architectures for modulation recognition", "1703 09197",               # West & O'Shea 2017
        "deep learning models for wireless signal classification", "1707 08908", "8357902",  # Rajendran 2018
        "over the air deep learning based radio signal classification", "1712 04578",  # O'Shea et al. 2018
        "automatic modulation classification with deep neural networks", "2301 11773",  # Harper et al. 2023
        "robust and fast automatic modulation classification with cnn", "1911 04970",   # Tekbiyik et al. 2020
        "improved automatic modulation recognition using deep learning with additive attention",  # ICRNNA
        # further deep-learning papers from the project presentation
        "large scale radio frequency signal classification", "2207 09918",
        "large scale radio frequency wideband signal detection", "2211 10335",
        "joint signal detection and automatic modulation classification", "2405 00736",
        "rf signal classification with synthetic training data", "2206 12967",
        "spectral detection and localization of radio events",
        "joint detection and classification of rf signals using deep learning",
        "end to end learning from spectrum data",
        "mcnet an efficient cnn architecture", "mcformer",
        "spatiotemporal multi channel learning framework",
        "data augmentation for deep learning based radio modulation classification",
        "adversarial attacks on deep learning based radio signal classification",
        "real time and embedded deep learning on fpga",
        "dfn yolo", "vt mcnet",
        "spectrum sensing and signal identification with deep learning",
        "recurrent neural radio anomaly detection", "saife",
    ],
}


def norm(text: str) -> str:
    """Lower case, remove accents/ligatures, turn every non-alphanumeric run into one space."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"-\s*\n\s*", "", text)  # re-join words hyphenated at a line break
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def first_pages_text(pdf: Path, pages: int = 2) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(pdf))
        return " ".join((reader.pages[i].extract_text() or "") for i in range(min(pages, len(reader.pages))))
    except Exception:
        return ""


def classify(pdf: Path) -> set[str]:
    haystack = norm(pdf.stem) + norm(first_pages_text(pdf))
    return {folder for folder, phrases in RULES.items() for p in phrases if norm(p) in haystack}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--apply", action="store_true", help="really move the files (default: dry run)")
    args = ap.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"Folder not found: {args.root}")
    try:
        import pypdf  # noqa: F401
    except ImportError:
        print("note: pypdf is not installed, so only file names are used (pip install pypdf)\n")

    pdfs = sorted(p for p in args.root.glob("*.pdf") if p.is_file())
    moves, unmatched, ambiguous = [], [], []
    for pdf in pdfs:
        hits = classify(pdf)
        if len(hits) == 1:
            moves.append((pdf, args.root / next(iter(hits)) / pdf.name))
        elif hits:
            ambiguous.append((pdf, sorted(hits)))
        else:
            unmatched.append(pdf)

    print(f"{len(pdfs)} PDF files directly in {args.root}\n")
    for src, dst in moves:
        print(f"  {src.name}  ->  {dst.parent.name}/")
    moved = 0
    if args.apply:
        for src, dst in moves:
            if dst.exists():
                print(f"  skipped (already exists): {dst}")
                continue
            dst.parent.mkdir(exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1
    print(f"\n{'moved' if args.apply else 'would move'}: {moved if args.apply else len(moves)}")
    if ambiguous:
        print("\nmatch more than one folder, left in place:")
        for p, h in ambiguous:
            print(f"  {p.name}: {', '.join(h)}")
    if unmatched:
        print("\nnot recognised, left in place (sort by hand or add the title to RULES):")
        for p in unmatched:
            print(f"  {p.name}")
    if not args.apply and moves:
        print("\nThis was a dry run. Run again with --apply to move the files.")


if __name__ == "__main__":
    main()
