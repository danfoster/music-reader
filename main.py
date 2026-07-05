#!/usr/bin/env python3
"""Transcribe sheet music images using the Sheet Music Transformer (SMT)."""

import argparse
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

SMT_DIR = Path(__file__).parent / ".smt"
SMT_REPO = "https://github.com/antoniorv6/SMT"

MODELS = {
    "grandstaff": "PRAIG/smt-fp-grandstaff",
    "polish": "PRAIG/smt-fp-polish-scores",
    "mozarteum": "PRAIG/smt-fp-mozarteum",
}


def ensure_smt() -> None:
    if not SMT_DIR.exists():
        print(f"Cloning SMT to {SMT_DIR} ...", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth=1", SMT_REPO, str(SMT_DIR)],
            check=True,
        )
    smt_str = str(SMT_DIR)
    if smt_str not in sys.path:
        sys.path.insert(0, smt_str)


def img_to_tensor(image):
    from torchvision import transforms

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])(image)


def token_duration(token: str) -> Fraction:
    """Return duration of a kern token in quarter-note beats, or 0 for null/barline."""
    first = token.split()[0]  # first note of chord; all notes in a chord share duration
    if first in (".", "*", "") or first.startswith("=") or first.startswith("*"):
        return Fraction(0)
    m = re.match(r"^(\d+)(\.{0,3})", first)
    if not m or m.group(1) == "0":
        return Fraction(0)
    base = Fraction(4, int(m.group(1)))  # quarter notes
    dot_value = base
    for _ in m.group(2):
        dot_value /= 2
        base += dot_value
    return base


def validate_kern(kern: str) -> list[str]:
    """Check each measure has the correct beat count per spine. Returns warning strings."""
    warnings = []
    beats_per_measure: Fraction | None = None
    spine_beats: list[Fraction] = []
    measure = 0
    first_measure = True

    for lineno, line in enumerate(kern.splitlines(), 1):
        if not line.strip():
            continue

        tokens = line.split("\t")

        # Spine declarations
        if tokens[0].startswith("**"):
            spine_beats = [Fraction(0)] * len(tokens)
            continue

        # Spine terminators / other interpretations
        if tokens[0].startswith("*"):
            for tok in tokens:
                m = re.match(r"^\*M(\d+)/(\d+)$", tok)
                if m:
                    beats_per_measure = Fraction(int(m.group(1)) * 4, int(m.group(2)))
            continue

        # Barline
        if tokens[0].startswith("="):
            if beats_per_measure is not None and any(b > 0 for b in spine_beats):
                for i, beats in enumerate(spine_beats):
                    if beats == 0:
                        continue
                    # Allow pickup measure (first measure, beats < expected)
                    if first_measure and beats < beats_per_measure:
                        continue
                    if beats != beats_per_measure:
                        warnings.append(
                            f"line {lineno}: measure {measure}, spine {i + 1}: "
                            f"{beats} beats (expected {beats_per_measure})"
                        )
            measure += 1
            first_measure = False
            spine_beats = [Fraction(0)] * len(spine_beats)
            continue

        # Data row
        for i, tok in enumerate(tokens):
            if i < len(spine_beats):
                spine_beats[i] += token_duration(tok)

    return warnings


def transcribe(image_path: str, model_key: str = "grandstaff") -> str:
    import cv2
    import torch
    from smt_model import SMTModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ref = MODELS[model_key]

    print(f"Loading {model_ref} on {device}...", file=sys.stderr)
    model = SMTModelForCausalLM.from_pretrained(model_ref).to(device)
    model.eval()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    tensor = img_to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions, _ = model.predict(tensor, convert_to_str=True)

    body = (
        "".join(predictions)
        .replace("<b>", "\n")
        .replace("<s>", " ")
        .replace("<t>", "\t")
    )
    n_spines = body.split("\n")[0].count("\t") + 1
    header = "\t".join(["**kern"] * n_spines)
    return header + "\n" + body


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe sheet music from an image using SMT"
    )
    parser.add_argument("image", help="Path to sheet music image (PNG/JPG)")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="grandstaff",
        help=(
            "grandstaff: piano scores (default); "
            "polish: Polish scores; "
            "mozarteum: Mozarteum scores"
        ),
    )
    parser.add_argument(
        "--update-smt",
        action="store_true",
        help="Pull latest SMT code before running",
    )
    args = parser.parse_args()

    ensure_smt()

    if args.update_smt:
        subprocess.run(["git", "-C", str(SMT_DIR), "pull"], check=True)

    result = transcribe(args.image, args.model)
    print(result)

    warnings = validate_kern(result)
    if warnings:
        print(f"\n{len(warnings)} rhythmic error(s) detected:", file=sys.stderr)
        for w in warnings:
            print(f"  {w}", file=sys.stderr)
    else:
        print("Validation passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
