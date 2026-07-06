#!/usr/bin/env python3
"""Transcribe sheet music images using the Sheet Music Transformer (SMT)."""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

SMT_DIR = Path(__file__).parent / ".smt"
SMT_REPO = "https://github.com/antoniorv6/SMT"

MODELS = {
    "grandstaff": "PRAIG/smt-fp-grandstaff",
    "polish": "PRAIG/smt-fp-polish-scores",
    "mozarteum": "PRAIG/smt-fp-mozarteum",
}

# Hard positional-encoding bounds (configuration_smt.py)
_MAX_H = 3508
_MAX_W = 2480
# Training min-pad (batch_preparation_img2seq in .smt/data.py)
_MIN_H = 256
_MIN_W = 128

SCAN_SCALES = [0.35, 0.5, 0.65, 1.0]

_model_cache: dict = {}


def ensure_smt() -> None:
    if not SMT_DIR.exists():
        print(f"Cloning SMT to {SMT_DIR} ...", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth=1", SMT_REPO, str(SMT_DIR)],
            check=True,
        )
    _patch_smt()
    smt_str = str(SMT_DIR)
    if smt_str not in sys.path:
        sys.path.insert(0, smt_str)


def _patch_smt() -> None:
    """Apply fixes to the SMT clone that haven't landed upstream."""
    cfg_path = SMT_DIR / "smt_model" / "configuration_smt.py"
    text = cfg_path.read_text()
    # SMTConfig.__init__ never calls super().__init__(**kwargs), leaving
    # PretrainedConfig base attributes uninitialised.
    needle = "        self.architectures = [\"SMT\"]"
    replacement = "        super().__init__(**kwargs)\n" + needle
    if "super().__init__" not in text:
        cfg_path.write_text(text.replace(needle, replacement))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalize_scale(image, ratio: float):
    """
    Resize image by ratio, clamp to positional-encoding bounds, white-pad to
    training minimum size — matching prepare_fp_data + batch_preparation_img2seq
    in .smt/data.py.
    """
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    new_w = math.ceil(w * ratio)
    new_h = math.ceil(h * ratio)

    # Clamp to maxh/maxw (positional encoding bounds), preserving aspect ratio
    if new_h > _MAX_H or new_w > _MAX_W:
        clamp = min(_MAX_H / new_h, _MAX_W / new_w)
        new_h = math.ceil(new_h * clamp)
        new_w = math.ceil(new_w * clamp)
        print(
            f"Warning: image clamped to positional bounds ({new_h}×{new_w})",
            file=sys.stderr,
        )

    resized = cv2.resize(image, (new_w, new_h))  # bilinear, matches training

    # White-pad to at least _MIN_H × _MIN_W, image anchored top-left
    pad_h = max(new_h, _MIN_H)
    pad_w = max(new_w, _MIN_W)
    if pad_h > new_h or pad_w > new_w:
        canvas = np.full(
            (pad_h, pad_w, resized.shape[2]), 255, dtype=resized.dtype
        )
        canvas[:new_h, :new_w] = resized
        resized = canvas

    return resized


def img_to_tensor(image):
    from torchvision import transforms

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])(image)


# ---------------------------------------------------------------------------
# Model loading (cached by model_ref)
# ---------------------------------------------------------------------------

def load_model(model_ref: str):
    if model_ref not in _model_cache:
        import torch
        from smt_model import SMTModelForCausalLM

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Loading {model_ref} on {device}...", file=sys.stderr)
        model = SMTModelForCausalLM.from_pretrained(model_ref).to(device)
        model.eval()
        _model_cache[model_ref] = (model, device)
    return _model_cache[model_ref]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_model(model, device: str, image, scale: float) -> str:
    """Scale, preprocess and run one inference pass. Returns kern string."""
    import torch

    scaled = normalize_scale(image, scale)
    tensor = img_to_tensor(scaled).unsqueeze(0).to(device)

    # Cap maxlen to avoid O(N²) attention OOM on long sequences.
    # 1200 tokens fits a full grandstaff page without memory pressure.
    orig_maxlen = model.maxlen
    model.maxlen = min(model.maxlen, 1200)
    with torch.no_grad():
        predictions, _ = model.predict(tensor, convert_to_str=True)
    model.maxlen = orig_maxlen

    body = (
        "".join(predictions)
        .replace("<b>", "\n")
        .replace("<s>", " ")
        .replace("<t>", "\t")
    )
    n_spines = body.split("\n")[0].count("\t") + 1
    header = "\t".join(["**kern"] * n_spines)
    return header + "\n" + body


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def token_duration(token: str) -> Fraction:
    """Return duration of a kern token in quarter-note beats, or 0 for null/barline."""
    first = token.split()[0]
    if first in (".", "*", "") or first.startswith("=") or first.startswith("*"):
        return Fraction(0)
    m = re.match(r"^(\d+)(\.{0,3})", first)
    if not m or m.group(1) == "0":
        return Fraction(0)
    base = Fraction(4, int(m.group(1)))
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
        if tokens[0].startswith("**"):
            spine_beats = [Fraction(0)] * len(tokens)
            continue
        if tokens[0].startswith("*"):
            for tok in tokens:
                m = re.match(r"^\*M(\d+)/(\d+)$", tok)
                if m:
                    beats_per_measure = Fraction(int(m.group(1)) * 4, int(m.group(2)))
            continue
        if tokens[0].startswith("="):
            if beats_per_measure is not None and any(b > 0 for b in spine_beats):
                for i, beats in enumerate(spine_beats):
                    if beats == 0:
                        continue
                    if first_measure and beats < beats_per_measure:
                        continue  # pickup measure allowed
                    if beats != beats_per_measure:
                        warnings.append(
                            f"line {lineno}: measure {measure}, spine {i + 1}: "
                            f"{beats} beats (expected {beats_per_measure})"
                        )
            measure += 1
            first_measure = False
            spine_beats = [Fraction(0)] * len(spine_beats)
            continue
        for i, tok in enumerate(tokens):
            if i < len(spine_beats):
                spine_beats[i] += token_duration(tok)

    return warnings


def _capture_verovio(kern: str):
    """Run verovio.toolkit().loadData(kern), capturing stderr. Returns (toolkit, stderr_text)."""
    import verovio

    old_fd = os.dup(2)
    tmp = tempfile.TemporaryFile()
    os.dup2(tmp.fileno(), 2)
    try:
        tk = verovio.toolkit()
        tk.loadData(kern)
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        tmp.seek(0)
        err_text = tmp.read().decode("utf-8", errors="replace")
        tmp.close()
    return tk, err_text


def verovio_validate(kern: str) -> list[str]:
    """Validate kern using verovio. Returns error strings (empty = valid)."""
    try:
        import verovio  # noqa: F401
    except ImportError:
        return []
    _, err_text = _capture_verovio(kern)
    return [l.strip() for l in err_text.splitlines() if l.strip()]


def is_truncated(kern: str) -> bool:
    """True if the model hit maxlen without emitting *- spine terminators."""
    return "*-" not in kern


def is_degenerate(kern: str) -> bool:
    """
    True if the model is stuck in a repetition loop.
    Heuristic: >50% of data rows are identical, or fewer barlines than expected
    for the number of data rows (indicating one measure repeating endlessly).
    """
    data_rows = [
        l for l in kern.splitlines()
        if l.strip() and not l.startswith("*") and not l.startswith("=")
    ]
    if not data_rows:
        return False
    barlines = sum(1 for l in kern.splitlines() if l.strip().startswith("="))
    # Expect at least 1 barline per 8 data rows on average; fewer = suspicious
    if len(data_rows) > 16 and barlines < len(data_rows) / 8:
        return True
    # More than half the data rows are identical
    from collections import Counter
    most_common_count = Counter(data_rows).most_common(1)[0][1]
    return most_common_count > len(data_rows) * 0.5


def score_kern(kern: str) -> tuple[int, int, bool, bool]:
    """Score a kern candidate — lower is better.
    (degenerate, verovio_errors, rhythm_errors, truncated)"""
    return (
        is_degenerate(kern),
        len(verovio_validate(kern)),
        len(validate_kern(kern)),
        is_truncated(kern),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_kern(kern: str, fmt: str) -> bytes | str:
    """Export kern to another format. fmt: 'mei' or 'midi'."""
    import base64

    tk, _ = _capture_verovio(kern)
    if fmt == "mei":
        return tk.getMEI()
    elif fmt == "midi":
        return base64.b64decode(tk.renderToMIDI())
    raise ValueError(f"Unsupported export format: {fmt!r}")


# ---------------------------------------------------------------------------
# Repair (opt-in, last resort)
# ---------------------------------------------------------------------------

def _rest_token(beats: Fraction) -> str | None:
    """Return a kern rest token for the given duration, or None if not representable."""
    for denom in [1, 2, 4, 8, 16, 32]:
        base = Fraction(4, denom)
        if base == beats:
            return f"{denom}r"
        if base * Fraction(3, 2) == beats:
            return f"{denom}.r"
        if base * Fraction(7, 4) == beats:
            return f"{denom}..r"
    return None


def repair_kern(kern: str) -> tuple[str, list[str]]:
    """
    Minimal opt-in rhythm repair: fills beat deficits with rests, reports overflows
    (overflows are not auto-corrected as that would truncate musical content).
    Returns (repaired_kern, list_of_repair_descriptions).
    """
    repairs: list[str] = []
    beats_per_measure: Fraction | None = None
    n_spines = 0

    for line in kern.splitlines():
        toks = line.split("\t")
        if toks[0].startswith("**"):
            n_spines = len(toks)
        for tok in toks:
            m = re.match(r"^\*M(\d+)/(\d+)$", tok)
            if m:
                beats_per_measure = Fraction(int(m.group(1)) * 4, int(m.group(2)))

    if beats_per_measure is None or n_spines == 0:
        return kern, repairs

    out_lines: list[str] = []
    measure_data: list[str] = []
    first_measure = True
    measure_num = 0

    def flush() -> None:
        nonlocal first_measure, measure_num
        totals = [Fraction(0)] * n_spines
        for ln in measure_data:
            for i, tok in enumerate(ln.split("\t")):
                if i < n_spines:
                    totals[i] += token_duration(tok)

        out_lines.extend(measure_data)

        for i, total in enumerate(totals):
            if total == 0:
                continue
            if first_measure and total < beats_per_measure:
                continue  # pickup measure, skip
            deficit = beats_per_measure - total
            if deficit > 0:
                rest = _rest_token(deficit)
                if rest:
                    out_lines.append(
                        "\t".join(rest if j == i else "." for j in range(n_spines))
                    )
                    repairs.append(
                        f"measure {measure_num}, spine {i+1}: "
                        f"inserted {rest} (deficit {deficit} beats)"
                    )
                else:
                    repairs.append(
                        f"measure {measure_num}, spine {i+1}: "
                        f"{deficit} beat deficit — cannot represent as single rest"
                    )
            elif deficit < 0:
                repairs.append(
                    f"measure {measure_num}, spine {i+1}: "
                    f"{-deficit} beat overflow — not auto-repaired (would alter content)"
                )

        measure_data.clear()
        first_measure = False
        measure_num += 1

    for line in kern.splitlines():
        toks = line.split("\t")
        if toks[0].startswith("**") or toks[0].startswith("*"):
            out_lines.append(line)
        elif toks[0].startswith("="):
            flush()
            out_lines.append(line)
        else:
            measure_data.append(line)

    if measure_data:
        flush()

    return "\n".join(out_lines), repairs


# ---------------------------------------------------------------------------
# Transcription entry points
# ---------------------------------------------------------------------------

def transcribe(image_path: str, model_key: str = "grandstaff", scale: float = 0.5) -> str:
    """Single-attempt transcription."""
    import cv2

    ensure_smt()
    model, device = load_model(MODELS[model_key])
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return run_model(model, device, image, scale)


def transcribe_best(
    image_path: str,
    model_keys: list[str],
    scales: list[float],
) -> str:
    """Sweep scales × models, return the candidate with the best (lowest) score."""
    import cv2

    ensure_smt()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    candidates = []
    for model_key in model_keys:
        model, device = load_model(MODELS[model_key])
        for scale in scales:
            print(f"  model={model_key} scale={scale}...", file=sys.stderr)
            kern = run_model(model, device, image, scale)
            sc = score_kern(kern)
            candidates.append((sc, model_key, scale, kern))
            print(
                f"    degenerate={sc[0]} verovio={sc[1]} rhythm={sc[2]} truncated={sc[3]}",
                file=sys.stderr,
            )

    candidates.sort(key=lambda x: x[0])
    sc, best_model, best_scale, best_kern = candidates[0]
    print(
        f"\nBest: model={best_model} scale={best_scale} "
        f"degenerate={sc[0]} verovio={sc[1]} rhythm={sc[2]} truncated={sc[3]}",
        file=sys.stderr,
    )
    return best_kern


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe sheet music from an image using SMT"
    )
    parser.add_argument("image", help="Path to sheet music image (PNG/JPG)")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="grandstaff",
        help="Model to use (default: grandstaff)",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated models to sweep when using --scan (e.g. grandstaff,polish)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Resize ratio applied to input before inference (default: 0.5, matching training)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help=f"Sweep scales {SCAN_SCALES} × models and keep the best result",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: input stem + .kern alongside input)",
    )
    parser.add_argument(
        "--export",
        choices=["mei", "midi"],
        help="Also export to MEI or MIDI (written alongside output with .mei/.mid extension)",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Last-resort: insert rests to fix beat deficits. "
            "WARNING: alters musical content. Original printed to stderr."
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

    # Inference
    if args.scan:
        model_keys = (
            [k.strip() for k in args.models.split(",")]
            if args.models
            else [args.model]
        )
        result = transcribe_best(args.image, model_keys, SCAN_SCALES)
    else:
        result = transcribe(args.image, args.model, args.scale)

    # Validation
    degenerate = is_degenerate(result)
    v_warnings = verovio_validate(result)
    r_warnings = validate_kern(result)
    n_errors = len(v_warnings) + len(r_warnings) + int(degenerate)

    if n_errors or degenerate:
        print(
            f"\ndegenerate={degenerate} {len(v_warnings)} verovio error(s), "
            f"{len(r_warnings)} rhythm error(s):",
            file=sys.stderr,
        )
        for w in v_warnings:
            print(f"  [verovio] {w}", file=sys.stderr)
        for w in r_warnings:
            print(f"  [rhythm]  {w}", file=sys.stderr)

        if args.repair:
            print("\n--- ORIGINAL (pre-repair) ---", file=sys.stderr)
            for line in result.splitlines():
                print(line, file=sys.stderr)
            print("--- END ORIGINAL ---\n", file=sys.stderr)
            print(
                "WARNING: --repair active. Musical content may be altered to satisfy "
                "the rhythmic validator.",
                file=sys.stderr,
            )
            result, repair_log = repair_kern(result)
            if repair_log:
                print("Repairs:", file=sys.stderr)
                for r in repair_log:
                    print(f"  {r}", file=sys.stderr)
            remaining = len(verovio_validate(result)) + len(validate_kern(result))
            print(f"After repair: {remaining} error(s) remaining.", file=sys.stderr)
    else:
        print("Validation passed.", file=sys.stderr)

    # Output
    out_path = Path(args.output) if args.output else Path(args.image).with_suffix(".kern")
    out_path.write_text(result)
    print(f"Written: {out_path}", file=sys.stderr)

    if args.export:
        ext = ".mei" if args.export == "mei" else ".mid"
        export_path = out_path.with_suffix(ext)
        exported = export_kern(result, args.export)
        if isinstance(exported, bytes):
            export_path.write_bytes(exported)
        else:
            export_path.write_text(exported)
        print(f"Exported: {export_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
