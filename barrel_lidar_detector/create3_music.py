#!/usr/bin/env python3
"""Play simple note melodies on the iRobot Create 3 speaker."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import yaml

from irobot_create_msgs.msg import AudioNote, AudioNoteVector

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - only used outside a ROS install
    get_package_share_directory = None


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

    if len(normalized) < 2:
        raise ValueError(f"Invalid note name: {note_name!r}")

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


def seconds_to_duration(seconds: float) -> tuple[int, int]:
    if seconds <= 0:
        raise ValueError("Note duration must be greater than zero")
    whole = int(math.floor(seconds))
    nanos = int(round((seconds - whole) * 1_000_000_000))
    if nanos >= 1_000_000_000:
        whole += 1
        nanos -= 1_000_000_000
    return whole, nanos


def default_melody_file() -> str:
    local_file = Path.cwd() / "config" / "create3_melodies.yaml"
    if local_file.exists():
        return str(local_file)

    if get_package_share_directory is not None:
        share_dir = Path(get_package_share_directory("barrel_lidar_detector"))
        share_file = share_dir / "config" / "create3_melodies.yaml"
        if share_file.exists():
            return str(share_file)

    return str(local_file)


class Create3Music(Node):
    def __init__(self) -> None:
        super().__init__("create3_music")
        self.declare_parameter("melody", "epic_theme")
        self.declare_parameter("melody_file", default_melody_file())
        self.declare_parameter("iterations", 1)
        self.declare_parameter("topic", "/cmd_audio")
        self.declare_parameter("list_melodies", False)

        topic = self.get_parameter("topic").get_parameter_value().string_value
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.create_publisher(AudioNoteVector, topic, qos)

    def run(self) -> None:
        melody_file = Path(
            self.get_parameter("melody_file").get_parameter_value().string_value
        ).expanduser()
        melody_name = self.get_parameter("melody").get_parameter_value().string_value
        iterations = self.get_parameter("iterations").get_parameter_value().integer_value
        list_melodies = (
            self.get_parameter("list_melodies").get_parameter_value().bool_value
        )

        melodies = self._load_melodies(melody_file)
        if list_melodies:
            names = ", ".join(sorted(melodies))
            self.get_logger().info(f"Available melodies: {names}")
            return

        if melody_name not in melodies:
            names = ", ".join(sorted(melodies))
            raise ValueError(f"Unknown melody {melody_name!r}. Available: {names}")

        sequence = self._build_sequence(melodies[melody_name])
        if not sequence:
            raise ValueError(f"Melody {melody_name!r} has no notes")

        self.get_logger().info(
            f"Playing {melody_name!r} on /cmd_audio for {iterations} iteration(s)"
        )
        time.sleep(0.5)

        repeat_forever = iterations < 0
        remaining = iterations
        while rclpy.ok() and (repeat_forever or remaining > 0):
            self._play_once(sequence)
            if not repeat_forever:
                remaining -= 1

    def _load_melodies(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Melody file not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        melodies = data.get("melodies", {})
        if not isinstance(melodies, dict):
            raise ValueError("Melody file must contain a 'melodies' mapping")
        return melodies

    def _build_sequence(self, entries: Any) -> list[tuple[int | None, float]]:
        if not isinstance(entries, list):
            raise ValueError("Melody entries must be a list")

        sequence: list[tuple[int | None, float]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each melody entry must be a mapping")

            duration = float(entry.get("duration", entry.get("seconds", 0.0)))
            if "frequency" in entry:
                frequency = int(entry["frequency"])
            elif "note" in entry:
                frequency = note_name_to_frequency(str(entry["note"]))
            else:
                raise ValueError("Each melody entry needs 'note' or 'frequency'")

            sequence.append((frequency, duration))
        return sequence

    def _play_once(self, sequence: list[tuple[int | None, float]]) -> None:
        pending: list[AudioNote] = []
        pending_duration = 0.0

        for frequency, duration in sequence:
            if frequency is None:
                self._publish_notes(pending)
                time.sleep(pending_duration + duration)
                pending = []
                pending_duration = 0.0
                continue

            note = AudioNote()
            note.frequency = frequency
            note.max_runtime.sec, note.max_runtime.nanosec = seconds_to_duration(duration)
            pending.append(note)
            pending_duration += duration

        self._publish_notes(pending)
        time.sleep(pending_duration + 0.1)

    def _publish_notes(self, notes: list[AudioNote]) -> None:
        if not notes:
            return
        message = AudioNoteVector()
        message.append = False
        message.notes = notes
        self.publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = Create3Music()
    try:
        node.run()
    except Exception as exc:  # noqa: BLE001 - report cleanly through ROS logging
        node.get_logger().error(str(exc))
        raise SystemExit(1) from exc
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
