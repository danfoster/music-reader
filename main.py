"""
music-reader — sheet music transcription via HOMR → MusicXML → WAV

Pipeline:
  image / PDF page  ──HOMR──▶  .musicxml  ──verovio──▶  MIDI  ──numpy──▶  .wav
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


# ---------------------------------------------------------------------------
# HOMR — image → MusicXML
# ---------------------------------------------------------------------------

def run_homr(image_path: str) -> str:
    """
    Run HOMR on *image_path* and return the resulting MusicXML as a string.

    HOMR writes <stem>.musicxml next to its input file; we stage the image in
    a temp directory so we control where the output lands.
    """
    src = Path(image_path)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / src.name
        shutil.copy2(src, staged)

        homr_bin = shutil.which("homr")
        if homr_bin:
            cmd = [homr_bin, str(staged)]
        else:
            cmd = [sys.executable, "-m", "homr.main", str(staged)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"HOMR failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
            )

        xml_path = staged.with_suffix(".musicxml")
        if not xml_path.exists():
            raise RuntimeError(
                f"HOMR produced no .musicxml for {src.name}.\n"
                f"stdout: {proc.stdout[-1000:]}\nstderr: {proc.stderr[-1000:]}"
            )
        return xml_path.read_text()


def transcribe_image(image_path: str, out_stem: Path) -> Path:
    """
    Run HOMR on *image_path*, save the MusicXML alongside *out_stem*, return its path.
    """
    print(f"Running HOMR on {Path(image_path).name} …", file=sys.stderr)
    xml_text = run_homr(image_path)
    xml_path = out_stem.with_suffix(".musicxml")
    xml_path.write_text(xml_text)
    print(f"Written: {xml_path}", file=sys.stderr)
    return xml_path


# ---------------------------------------------------------------------------
# PDF → page images
# ---------------------------------------------------------------------------

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
        # fitz gives RGB; save as PNG for HOMR (which reads any format)
        yield img[:, :, ::-1].copy()
    doc.close()


# ---------------------------------------------------------------------------
# MusicXML → MIDI (verovio, subprocess-isolated)
# ---------------------------------------------------------------------------

def musicxml_to_midi(xml: str) -> bytes:
    """Convert a MusicXML string to raw MIDI bytes via verovio (subprocess)."""
    import base64

    script = """
import sys, base64, verovio
xml = sys.stdin.read()
tk = verovio.toolkit()
tk.setInputFrom("musicxml")
tk.loadData(xml)
sys.stdout.write(tk.renderToMIDI())  # already base64
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=xml,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"verovio MIDI export failed (exit {proc.returncode}): {proc.stderr[:500]}"
        )
    return base64.b64decode(proc.stdout)


# ---------------------------------------------------------------------------
# MIDI → WAV (numpy additive synthesis, no external deps)
# ---------------------------------------------------------------------------

def _midi_note_events(
    midi_bytes: bytes, time_offset: float = 0.0
) -> list[tuple[float, float, int, int]]:
    """Parse MIDI bytes and return (start_sec, end_sec, pitch, velocity) tuples."""
    import mido

    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    tempo = 500000
    tpb = mid.ticks_per_beat
    active: dict[tuple[int, int], tuple[float, int]] = {}
    notes: list[tuple[float, float, int, int]] = []
    abs_sec = time_offset

    for msg in mido.merge_tracks(mid.tracks):
        abs_sec += mido.tick2second(msg.time, tpb, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (abs_sec, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active:
                start, vel = active.pop(key)
                notes.append((start, abs_sec, msg.note, vel))

    for (_, note), (start, vel) in active.items():
        notes.append((start, abs_sec + 0.5, note, vel))

    return notes


def midi_to_wav(midi_bytes: bytes, out_path: Path, sample_rate: int = 44100) -> None:
    """Synthesise MIDI bytes to a WAV file using additive synthesis."""
    import numpy as np

    notes = _midi_note_events(midi_bytes)
    if not notes:
        raise RuntimeError("No MIDI notes found")

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
        sig = (
            np.sin(2 * np.pi * freq * t) * 0.50
            + np.sin(2 * np.pi * 2 * freq * t) * 0.20
            + np.sin(2 * np.pi * 3 * freq * t) * 0.10
            + np.sin(2 * np.pi * 4 * freq * t) * 0.05
        )
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

    peak = np.abs(buf).max()
    if peak > 0:
        buf = buf / peak * 0.9
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def musicxml_files_to_wav(xml_paths: list[Path], wav_path: Path) -> None:
    """
    Convert a sequence of MusicXML files to a single continuous WAV file.
    Each file's audio is placed after the previous one's last note.
    """
    all_notes: list[tuple[float, float, int, int]] = []
    cursor = 0.0

    for xml_path in xml_paths:
        try:
            midi_bytes = musicxml_to_midi(xml_path.read_text())
        except Exception as e:
            print(f"MIDI render failed for {xml_path.name}: {e}", file=sys.stderr)
            continue
        page_notes = _midi_note_events(midi_bytes, time_offset=cursor)
        if page_notes:
            all_notes.extend(page_notes)
            cursor = max(end for _, end, _, _ in page_notes) + 0.5

    if not all_notes:
        raise RuntimeError("No renderable notes across all MusicXML files")

    # Synthesise combined buffer
    import numpy as np

    sample_rate = 44100
    total = max(end for _, end, _, _ in all_notes) + 0.5
    n_samples = int(total * sample_rate)
    buf = np.zeros(n_samples, dtype=np.float64)

    for start, end, pitch, velocity in all_notes:
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        dur = max(end - start, 0.01)
        n = min(int(dur * sample_rate), n_samples - int(start * sample_rate))
        if n <= 0:
            continue
        t = np.linspace(0.0, dur, n, endpoint=False)
        sig = (
            np.sin(2 * np.pi * freq * t) * 0.50
            + np.sin(2 * np.pi * 2 * freq * t) * 0.20
            + np.sin(2 * np.pi * 3 * freq * t) * 0.10
            + np.sin(2 * np.pi * 4 * freq * t) * 0.05
        )
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

    peak = np.abs(buf).max()
    if peak > 0:
        buf = buf / peak * 0.9
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe sheet music from an image or PDF using HOMR"
    )
    parser.add_argument("image", help="Path to sheet music image (PNG/JPG) or PDF")
    parser.add_argument(
        "--output", "-o",
        help="Output base path (default: input stem alongside input). "
             "Extensions .musicxml and .wav are appended automatically.",
    )
    parser.add_argument(
        "--no-wav",
        action="store_true",
        help="Write MusicXML only; skip WAV rendering",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF page rendering (default: 200)",
    )
    args = parser.parse_args()

    src = Path(args.image)
    out_stem = Path(args.output) if args.output else src.with_suffix("")
    is_pdf = src.suffix.lower() == ".pdf"

    if is_pdf:
        xml_paths: list[Path] = []
        with tempfile.TemporaryDirectory() as tmp:
            import numpy as np
            import cv2

            for page_num, img_bgr in enumerate(pdf_to_page_images(str(src), dpi=args.dpi), start=1):
                tmp_png = Path(tmp) / f"page_{page_num:03d}.png"
                cv2.imwrite(str(tmp_png), img_bgr)
                page_stem = out_stem.parent / f"{out_stem.name}-p{page_num}"
                try:
                    xml_path = transcribe_image(str(tmp_png), page_stem)
                    xml_paths.append(xml_path)
                except Exception as e:
                    print(f"Page {page_num} failed: {e}", file=sys.stderr)

        if not xml_paths:
            print("No pages transcribed successfully.", file=sys.stderr)
            sys.exit(1)

        if not args.no_wav:
            wav_path = out_stem.with_suffix(".wav")
            try:
                musicxml_files_to_wav(xml_paths, wav_path)
                print(f"Written: {wav_path}", file=sys.stderr)
            except Exception as e:
                print(f"WAV render failed: {e}", file=sys.stderr)

    else:
        try:
            xml_path = transcribe_image(str(src), out_stem)
        except Exception as e:
            print(f"Transcription failed: {e}", file=sys.stderr)
            sys.exit(1)

        if not args.no_wav:
            wav_path = out_stem.with_suffix(".wav")
            try:
                musicxml_files_to_wav([xml_path], wav_path)
                print(f"Written: {wav_path}", file=sys.stderr)
            except Exception as e:
                print(f"WAV render failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
