#!/usr/bin/env python3
"""Transcribe sheet music images using the Sheet Music Transformer (SMT)."""

import argparse
import subprocess
import sys
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

    print(transcribe(args.image, args.model))


if __name__ == "__main__":
    main()
