#!/usr/bin/env python3
"""Render a Create 3 melody YAML entry to a WAV preview file."""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

import yaml


NOTE_OFFSETS = {
    "C": -9,
    "C#": -8,
    "DB": -8,
    "D": -7,
    "D#": -6,
    "EB": -6,
    "E": -5,
    "F": -4,
    "F#": -3,
    "GB": -3,
    "G": -2,
    "G#": -1,
    "AB": -1,
    "A": 0,
    "A#": 1,
    "BB": 1,
    "B": 2,
}


def note_name_to_frequency(note_name: str) -> int | None:
    normalized = note_name.strip().upper()
    if normalized in {"R", "REST", "SILENCE"}:
        return None

    if len(normalized) >= 3 and normalized[1] in {"#", "B"}:
        note = normalized[:2]
        octave_text = normalized[2:]
    else:
        note = normalized[:1]
        octave_text = normalized[1:]

    if note not in NOTE_OFFSETS:
        raise ValueError(f"Invalid note name: {note_name!r}")

    octave = int(octave_text)
    semitone_from_a4 = NOTE_OFFSETS[note] + (octave - 4) * 12
    return int(round(440.0 * (2.0 ** (semitone_from_a4 / 12.0))))


def load_sequence(melody_file: Path, melody_name: str) -> list[tuple[int | None, float]]:
    with melody_file.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    melodies = data.get("melodies", {})
    if melody_name not in melodies:
        names = ", ".join(sorted(melodies))
        raise ValueError(f"Unknown melody {melody_name!r}. Available: {names}")

    sequence: list[tuple[int | None, float]] = []
    for entry in melodies[melody_name]:
        duration = float(entry.get("duration", entry.get("seconds", 0.0)))
        if duration <= 0:
            raise ValueError("Durations must be greater than zero")

        if "frequency" in entry:
            frequency = int(entry["frequency"])
        elif "note" in entry:
            frequency = note_name_to_frequency(str(entry["note"]))
        else:
            raise ValueError("Each melody entry needs 'note' or 'frequency'")

        sequence.append((frequency, duration))

    return sequence


def render_wave(
    sequence: list[tuple[int | None, float]],
    output_path: Path,
    sample_rate: int,
    volume: float,
) -> None:
    samples: list[int] = []
    amplitude = int(32767 * volume)
    attack_seconds = 0.005
    release_seconds = 0.012

    for frequency, duration in sequence:
        sample_count = max(1, int(sample_rate * duration))
        attack_count = max(1, int(sample_rate * attack_seconds))
        release_count = max(1, int(sample_rate * release_seconds))

        for i in range(sample_count):
            if frequency is None:
                samples.append(0)
                continue

            t = i / sample_rate
            envelope = 1.0
            if i < attack_count:
                envelope = i / attack_count
            elif i > sample_count - release_count:
                envelope = max(0.0, (sample_count - i) / release_count)

            # Add a small second harmonic so previews feel closer to a tiny robot speaker.
            base = math.sin(2.0 * math.pi * frequency * t)
            harmonic = 0.22 * math.sin(2.0 * math.pi * frequency * 2.0 * t)
            samples.append(int(amplitude * envelope * (base + harmonic) / 1.22))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("melody", help="Melody name from the YAML file")
    parser.add_argument(
        "-m",
        "--melody-file",
        default="config/create3_melodies.yaml",
        help="Path to create3_melodies.yaml",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output WAV path. Defaults to /tmp/create3_<melody>.wav",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--volume", type=float, default=0.35)
    args = parser.parse_args()

    output_path = Path(args.output or f"/tmp/create3_{args.melody}.wav")
    sequence = load_sequence(Path(args.melody_file), args.melody)
    render_wave(sequence, output_path, args.sample_rate, args.volume)
    seconds = sum(duration for _, duration in sequence)
    print(f"Wrote {output_path} ({seconds:.2f} seconds)")


if __name__ == "__main__":
    main()
