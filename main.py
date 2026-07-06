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
    # Determine spine count from first non-empty line
    n_spines = 1
    for line in body.split("\n"):
        if line.strip():
            n_spines = line.count("\t") + 1
            break

    # Strip spine-split (*^) and spine-merge (*v) interpretation lines —
    # the model sometimes emits them but never follows with correct field counts.
    # Normalize every data row to exactly n_spines fields so verovio doesn't crash.
    normalized = []
    for line in body.split("\n"):
        toks = line.split("\t")
        if any(t.strip() in ("*^", "*v") for t in toks):
            continue
        if line and not line.startswith("*"):
            # Data rows and barlines: normalize field count to n_spines
            if len(toks) > n_spines:
                toks = toks[:n_spines]
            elif len(toks) < n_spines:
                fill = "=" if line.startswith("=") else "."
                toks += [fill] * (n_spines - len(toks))
            line = "\t".join(toks)
        normalized.append(line)
    body = "\n".join(normalized)
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
    # Run verovio in a subprocess so a crash (e.g. std::length_error on
    # malformed spine ops) doesn't kill the parent process.
    script = (
        "import sys, verovio, os, tempfile\n"
        "kern = sys.stdin.read()\n"
        "old = os.dup(2); tmp = tempfile.TemporaryFile()\n"
        "os.dup2(tmp.fileno(), 2)\n"
        "try:\n"
        "    tk = verovio.toolkit(); tk.loadData(kern)\n"
        "finally:\n"
        "    os.dup2(old, 2); os.close(old)\n"
        "    tmp.seek(0); errs = tmp.read().decode('utf-8', errors='replace'); tmp.close()\n"
        "print(errs, end='')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=kern,
        capture_output=True,
        text=True,
    )
    err_text = result.stdout + result.stderr
    # If subprocess crashed, report that as an error
    if result.returncode not in (0, 1):
        err_text = f"verovio subprocess exited {result.returncode}: {err_text[:200]}"
    return err_text


def verovio_validate(kern: str) -> list[str]:
    """Validate kern using verovio. Returns error strings (empty = valid)."""
    try:
        import verovio  # noqa: F401
    except ImportError:
        return []
    err_text = _capture_verovio(kern)
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
    """Export kern to another format. fmt: 'mei' or 'midi'. Runs in a subprocess to isolate verovio crashes."""
    import base64

    if fmt not in ("mei", "midi"):
        raise ValueError(f"Unsupported export format: {fmt!r}")

    # renderToMIDI already returns base64 text, so we keep everything in text mode
    # and decode on the parent side for MIDI.
    script = """
import sys, verovio
kern = sys.stdin.read()
tk = verovio.toolkit()
tk.loadData(kern)
fmt = sys.argv[1]
if fmt == 'mei':
    sys.stdout.write(tk.getMEI())
else:
    sys.stdout.write(tk.renderToMIDI())  # already base64 text
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, fmt],
        input=kern,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"verovio export failed (exit {proc.returncode}): {proc.stderr[:200]}")
    if fmt == "mei":
        return proc.stdout
    return base64.b64decode(proc.stdout)


# ---------------------------------------------------------------------------
# WAV synthesis (MIDI → WAV via additive synthesis, no external deps)
# ---------------------------------------------------------------------------

def kern_to_wav(kern: str, out_path: Path, sample_rate: int = 44100) -> None:
    """Render kern to a WAV file via verovio MIDI + numpy additive synthesis."""
    import io, wave
    import mido
    import numpy as np

    midi_bytes = export_kern(kern, "midi")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    tempo = 500000
    tpb = mid.ticks_per_beat
    active: dict[tuple[int, int], tuple[float, int]] = {}
    notes: list[tuple[float, float, int, int]] = []
    abs_sec = 0.0

    for msg in mido.merge_tracks(mid.tracks):
        abs_sec += mido.tick2second(msg.time, tpb, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (abs_sec, msg.velocity)
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active:
                start, vel = active.pop(key)
                notes.append((start, abs_sec, msg.note, vel))

    for (_, note), (start, vel) in active.items():
        notes.append((start, abs_sec + 0.5, note, vel))

    if not notes:
        raise RuntimeError("No MIDI notes found in rendered output")

    total = max(end for _, end, _, _ in notes) + 0.5
    n_samples = int(total * sample_rate)
    buf = np.zeros(n_samples, dtype=np.float64)

    for start, end, pitch, velocity in notes:
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        dur = max(end - start, 0.01)
        n = min(int(dur * sample_rate), n_samples - int(start * sample_rate))
        if n <= 0:
            continue
        t = np.linspace(0.0, dur, n, endpoint=False)
        # Additive synthesis: fundamental + harmonics (piano-like timbre)
        sig = (
            np.sin(2 * np.pi * freq * t) * 0.50
            + np.sin(2 * np.pi * 2 * freq * t) * 0.20
            + np.sin(2 * np.pi * 3 * freq * t) * 0.10
            + np.sin(2 * np.pi * 4 * freq * t) * 0.05
        )
        # ADSR envelope
        a = min(int(0.005 * sample_rate), n // 4)
        d = min(int(0.06 * sample_rate), n // 3)
        r = min(int(0.08 * sample_rate), n // 4)
        s_len = max(n - a - d - r, 0)
        sustain = 0.65
        env = np.concatenate([
            np.linspace(0, 1, a),
            np.linspace(1, sustain, d),
            np.full(s_len, sustain),
            np.linspace(sustain, 0, r),
        ])[:n]
        sig = sig * env * (velocity / 127.0) * 0.6

        i0 = int(start * sample_rate)
        buf[i0 : i0 + n] += sig

    peak = np.max(np.abs(buf))
    if peak > 0:
        buf = buf / peak * 0.9

    pcm = (buf * 32767).astype(np.int16)
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


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


def _realign_measure(rows: list[str], n_spines: int) -> list[str]:
    """
    Reconstruct measure rows so every row contains only events that share the same
    timepoint. When the model emits two spines' events on the same row despite them
    belonging to different timepoints, verovio produces "Inconsistent rhythm analysis"
    errors. This collects all (timepoint, spine, token) events, sorts by timepoint,
    then re-emits one row per distinct timepoint.
    """
    events: list[tuple[Fraction, int, str]] = []
    spine_time = [Fraction(0)] * n_spines
    for row in rows:
        toks = (row.split("\t") + ["."] * n_spines)[:n_spines]
        for i, tok in enumerate(toks):
            if tok not in (".", ""):
                events.append((spine_time[i], i, tok))
                spine_time[i] += token_duration(tok)
    if not events:
        return rows
    events.sort(key=lambda e: e[0])
    out = []
    i = 0
    while i < len(events):
        t = events[i][0]
        row_evts: dict[int, str] = {}
        while i < len(events) and events[i][0] == t:
            row_evts[events[i][1]] = events[i][2]
            i += 1
        out.append("\t".join(row_evts.get(j, ".") for j in range(n_spines)))
    return out


def _rest_sequence(beats: Fraction) -> list[str]:
    """Decompose any duration into a minimal list of kern rest tokens (greedy)."""
    result = []
    for denom in [1, 2, 4, 8, 16, 32]:
        base = Fraction(4, denom)
        for mult, suffix in [(Fraction(7, 4), ".."), (Fraction(3, 2), "."), (Fraction(1), "")]:
            val = base * mult
            while beats >= val:
                result.append(f"{denom}{suffix}r")
                beats -= val
    return result


def repair_kern(kern: str) -> tuple[str, list[str]]:
    """
    Best-effort rhythm repair: normalises mismatched time signatures, fills beat
    deficits with rests, and trims overflows by nulling/shortening the last token(s)
    in the offending spine. Always produces rhythmically valid output; content near
    measure boundaries may be very slightly wrong.
    Returns (repaired_kern, list_of_repair_descriptions).
    """
    repairs: list[str] = []
    beats_per_measure: Fraction | None = None
    n_spines = 0
    out_lines: list[str] = []
    measure_data: list[str] = []
    first_measure = True
    measure_num = 0

    def _trim_overflow(spine_idx: int, overflow: Fraction) -> None:
        for li in range(len(measure_data) - 1, -1, -1):
            if overflow <= 0:
                break
            row = measure_data[li].split("\t")
            if spine_idx >= len(row) or row[spine_idx] in (".", ""):
                continue
            dur = token_duration(row[spine_idx])
            if dur <= 0:
                continue
            if dur <= overflow:
                overflow -= dur
                row[spine_idx] = "."
                measure_data[li] = "\t".join(row)
            else:
                # Partial: keep only (dur - overflow) beats; insert rest sequence
                allowed = dur - overflow
                rests = _rest_sequence(allowed)
                row[spine_idx] = rests[0] if rests else "."
                measure_data[li] = "\t".join(row)
                for extra in rests[1:]:
                    new_row = [extra if j == spine_idx else "." for j in range(n_spines)]
                    measure_data.insert(li + 1, "\t".join(new_row))
                overflow = Fraction(0)

    def flush() -> None:
        nonlocal first_measure, measure_num
        if beats_per_measure is None or n_spines == 0:
            out_lines.extend(measure_data)
            measure_data.clear()
            return

        # Realign cross-spine temporal ordering before any per-spine corrections
        realigned = _realign_measure(measure_data, n_spines)
        measure_data[:] = realigned

        def totals_now() -> list[Fraction]:
            t = [Fraction(0)] * n_spines
            for ln in measure_data:
                for i, tok in enumerate(ln.split("\t")):
                    if i < n_spines:
                        t[i] += token_duration(tok)
            return t

        # Fix overflows first (mutates measure_data before appending to out_lines)
        for i, total in enumerate(totals_now()):
            if total == 0 or (first_measure and total < beats_per_measure):
                continue
            if total > beats_per_measure:
                _trim_overflow(i, total - beats_per_measure)
                repairs.append(
                    f"measure {measure_num}, spine {i+1}: trimmed {total - beats_per_measure} beat overflow"
                )

        out_lines.extend(measure_data)

        # Fill deficits after trimming
        for i, total in enumerate(totals_now()):
            if total == 0 or (first_measure and total < beats_per_measure):
                continue
            deficit = beats_per_measure - total
            if deficit > 0:
                rests = _rest_sequence(deficit)
                for rest in rests:
                    out_lines.append(
                        "\t".join(rest if j == i else "." for j in range(n_spines))
                    )
                if rests:
                    repairs.append(
                        f"measure {measure_num}, spine {i+1}: inserted {' '.join(rests)} (deficit {deficit} beats)"
                    )
                else:
                    repairs.append(
                        f"measure {measure_num}, spine {i+1}: {deficit} beat deficit — unresolvable"
                    )

        measure_data.clear()
        first_measure = False
        measure_num += 1

    for line in kern.splitlines():
        toks = line.split("\t")
        if toks[0].startswith("**"):
            n_spines = len(toks)
            out_lines.append(line)
        elif toks[0].startswith("*"):
            # Normalize mismatched time signatures: pick first *M token seen on the line
            time_sigs = [t for t in toks if re.match(r"^\*M\d+/\d+$", t)]
            if len(set(time_sigs)) > 1:
                canonical = time_sigs[0]
                toks = [canonical if re.match(r"^\*M\d+/\d+$", t) else t for t in toks]
                repairs.append(f"normalized time sig to {canonical} (was {set(time_sigs)})")
                line = "\t".join(toks)
            for tok in toks:
                m = re.match(r"^\*M(\d+)/(\d+)$", tok)
                if m:
                    beats_per_measure = Fraction(int(m.group(1)) * 4, int(m.group(2)))
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
# System detection
# ---------------------------------------------------------------------------

def detect_systems(image) -> list:
    """
    Split a full-page image into individual staff-system crops.

    Strategy: horizontal projection profile → mark rows as "gap" when dark
    pixel count is below a low threshold for a minimum consecutive run.
    Staff lines span the full width so inter-system gaps are reliably near-zero;
    within a system, notes and beams always maintain some dark pixels.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY_INV)
    proj = binary.sum(axis=1).astype(float)

    # A row is a gap if it has fewer dark pixels than this fraction of the peak.
    # 5 % catches true white rows while surviving beams/slurs that cross a row.
    gap_threshold = proj.max() * 0.05

    # A split only fires on a gap run of at least this many consecutive rows.
    # Treble-bass intra-system gaps are ~53px; inter-system gaps are ~87px
    # for a typical 2000px page scan. image.shape[0]//35 scales proportionally.
    min_gap_rows = max(40, image.shape[0] // 35)

    # Build a boolean gap mask
    is_gap = proj < gap_threshold

    # Find gap runs
    split_centers = []
    run_start = None
    for i, g in enumerate(is_gap):
        if g and run_start is None:
            run_start = i
        elif not g and run_start is not None:
            run_len = i - run_start
            if run_len >= min_gap_rows:
                split_centers.append((run_start + i) // 2)
            run_start = None
    # Trailing whitespace: if the last gap reaches the image edge, it's
    # not a system boundary — don't split, let it merge into the last crop.

    # Build crop boundaries from split centres
    boundaries = [0] + split_centers + [image.shape[0]]
    padding = 10
    min_h = max(50, image.shape[0] // 12)
    crops = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= min_h:
            top = max(0, s - padding)
            bot = min(image.shape[0], e + padding)
            crops.append(image[top:bot, :])

    return crops


def concatenate_kern(parts: list[str]) -> str:
    """
    Merge per-system kern strings into a single score.
    Keeps the first system's spine header; strips **kern headers from
    subsequent systems but preserves clef/key/time interpretation changes.
    """
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]

    n_spines = parts[0].splitlines()[0].count("\t") + 1
    terminator = "\t".join(["*-"] * n_spines)

    def strip(kern: str, first: bool) -> list[str]:
        lines = kern.splitlines()
        out = []
        for line in lines:
            toks = line.split("\t")
            # Skip **kern spine header (only keep it for the first segment)
            if all(t.startswith("**") for t in toks):
                if first:
                    out.append(line)
                continue
            # Skip *- terminators — we add one at the very end
            if all(t.strip() == "*-" for t in toks):
                continue
            out.append(line)
        return out

    result = strip(parts[0], first=True)
    for part in parts[1:]:
        # Ensure a barline separates systems
        if result and not result[-1].startswith("="):
            result.append("\t".join(["="] * n_spines))
        result.extend(strip(part, first=False))

    result.append(terminator)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Transcription entry points
# ---------------------------------------------------------------------------

def _load_image(image_path: str):
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def _transcribe_image(image, model_key: str, scale: float) -> str:
    model, device = load_model(MODELS[model_key])
    return run_model(model, device, image, scale)


def _transcribe_best_image(image, model_keys: list[str], scales: list[float]) -> str:
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
        f"  Best: model={best_model} scale={best_scale} "
        f"degenerate={sc[0]} verovio={sc[1]} rhythm={sc[2]} truncated={sc[3]}",
        file=sys.stderr,
    )
    return best_kern


def transcribe(image_path: str, model_key: str = "grandstaff", scale: float = 0.5) -> str:
    ensure_smt()
    return _transcribe_image(_load_image(image_path), model_key, scale)


def transcribe_best(image_path: str, model_keys: list[str], scales: list[float]) -> str:
    ensure_smt()
    return _transcribe_best_image(_load_image(image_path), model_keys, scales)


def _transcribe_page(
    image,
    model_keys: list[str],
    scales: list[float],
    use_scan: bool,
    page_label: str = "",
) -> list[str]:
    """Detect systems on one page image and return a list of per-system kern strings."""
    crops = detect_systems(image)
    n = len(crops)
    label = f"{page_label}: " if page_label else ""
    print(f"{label}Detected {n} system(s)", file=sys.stderr)
    parts = []
    for i, crop in enumerate(crops):
        print(f"\n{label}System {i + 1}/{n}:", file=sys.stderr)
        kern = _transcribe_best_image(crop, model_keys, scales) if use_scan else _transcribe_image(crop, model_keys[0], scales[0])
        parts.append(kern)
    return parts


def pdf_to_page_images(pdf_path: str, dpi: int = 200):
    """Yield each page of a PDF as a numpy BGR image (same format as cv2.imread)."""
    import fitz
    import numpy as np
    scale = dpi / 72.0
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        # fitz gives RGB; cv2 expects BGR
        yield img[:, :, ::-1].copy()
    doc.close()


def transcribe_systems(
    image_path: str,
    model_keys: list[str],
    scales: list[float],
    use_scan: bool = False,
) -> str:
    """
    Detect staff systems on one image, transcribe each, concatenate results.
    When use_scan=True, sweeps scales × models per system.
    """
    ensure_smt()
    image = _load_image(image_path)
    parts = _transcribe_page(image, model_keys, scales, use_scan)
    if not parts:
        raise RuntimeError("No staff systems detected in image")
    return concatenate_kern(parts)


def transcribe_pdf(
    pdf_path: str,
    model_keys: list[str],
    scales: list[float],
    use_scan: bool = False,
    dpi: int = 200,
) -> str:
    """Transcribe all pages of a PDF and concatenate into a single kern string."""
    ensure_smt()
    all_parts: list[str] = []
    for page_num, image in enumerate(pdf_to_page_images(pdf_path, dpi=dpi), start=1):
        print(f"\n=== Page {page_num} ===", file=sys.stderr)
        parts = _transcribe_page(image, model_keys, scales, use_scan, page_label=f"p{page_num}")
        all_parts.extend(parts)
    if not all_parts:
        raise RuntimeError("No staff systems detected in PDF")
    return concatenate_kern(all_parts)


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
        "--systems",
        action="store_true",
        help="Detect individual staff systems and transcribe each separately (recommended for full pages)",
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
        "--no-wav",
        action="store_true",
        help="Skip WAV audio rendering (WAV is written by default alongside the kern output)",
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
    model_keys = (
        [k.strip() for k in args.models.split(",")]
        if args.models
        else [args.model]
    )
    is_pdf = Path(args.image).suffix.lower() == ".pdf"
    if is_pdf:
        result = transcribe_pdf(
            args.image,
            model_keys,
            SCAN_SCALES if args.scan else [args.scale],
            use_scan=args.scan,
        )
    elif args.systems:
        result = transcribe_systems(
            args.image,
            model_keys,
            SCAN_SCALES if args.scan else [args.scale],
            use_scan=args.scan,
        )
    elif args.scan:
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
            v_errs = verovio_validate(result)
            r_errs = validate_kern(result)
            remaining = len(v_errs) + len(r_errs)
            print(f"After repair: {remaining} error(s) remaining.", file=sys.stderr)
            for e in v_errs:
                print(f"  [verovio] {e}", file=sys.stderr)
            for e in r_errs:
                print(f"  [rhythm]  {e}", file=sys.stderr)
    else:
        print("Validation passed.", file=sys.stderr)

    # Output
    out_path = Path(args.output) if args.output else Path(args.image).with_suffix(".kern")
    out_path.write_text(result)
    print(f"Written: {out_path}", file=sys.stderr)

    if not args.no_wav:
        wav_path = out_path.with_suffix(".wav")
        try:
            # Always repair before rendering audio — invalid kern causes verovio SIGABRT
            kern_for_wav, _ = repair_kern(result) if n_errors > 0 else (result, [])
            kern_to_wav(kern_for_wav, wav_path)
            print(f"Written: {wav_path}", file=sys.stderr)
        except Exception as e:
            print(f"WAV render failed: {e}", file=sys.stderr)

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
