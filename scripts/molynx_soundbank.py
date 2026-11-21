"""Molynx's own sound bank, built from his sound files (molynx_sounds) on the Bile Demon's bank.

How the game finds a creature's sounds. A creature record's `soundGategory` (char[32], the Bile
Demon's "CREATURE_BILE_DEMON") is copied to a buffer and handed to the sound system's bank loader
(DKII-DX.exe 0x493d8a..0x493dae, the same pattern at every other GLOBAL\\ load), which opens
`data\\sound\\SFX\\GLOBAL\\<category>BANK.map` and `...<category>SFX.map` (0x60f870, 0x60f8ab):
the file names are built from the category string, nothing is listed in the exe, and Windows file
names ignore case. So a record whose category is CREATURE_MOLYNX loads

  Data/Sound/Sfx/Global/creature_molynxBANK.map   the sample banks the events use: the Bile Demon's
                                                  with "Global\\Bile" renamed "Global\\Molynx"; the
                                                  second bank, Global\\MiscCreatures (shared by many
                                                  creatures), is kept
  Data/Sound/Sfx/Global/creature_molynxSFX.map    the events (which sample, volume, chance): the Bile
                                                  Demon's, byte for byte, so every event plays the
                                                  sample at the same place in his bank
  Data/Sound/Sfx/Global/MolynxHD.sdt, MolynxHW.sdt  his samples, in the Bile Demon's order: the bank
                                                  name + "HD.sdt" / "HW.sdt" (exe strings D.sdt,
                                                  W.sdt; which one plays depends on the sound
                                                  settings, so both are written)

The .sdt layout: u32 count, count u32 offsets, then per sample a 40-byte header (u32 header size 40,
u32 data size, char[16] name, u32 format, u32 0, u32 PCM byte length, u32 0) and the data. The
format 0x24105622 is 22050 Hz (0x5622), 16 bit (0x10), compressed (0x24); "Blank" entries have
0x00105622 and no data. The game's samples are MPEG-2 audio at 22050 Hz mono; Quuz_AudioFix's
MaidenHW.sdt (which fixed the Maiden's samples being WAV, not mp2) uses MPEG-2 Layer II at 64
kbit/s, the format written here (ffmpeg's mp2 encoder). The PCM byte length is the sound's own
length in 16-bit samples x 2.

Flame serves all of these from the mod folder like any other data file (Quuz_AudioFix ships .sdt
and SFX.map files the same way).

LICENCE: GPL-3.0 (code). The sounds are molynx_sounds' (partly CC BY-SA 3.0, MegaGlest).
"""
from __future__ import annotations

import struct
import subprocess
import wave
from pathlib import Path

from scripts.molynx_sounds import ROLES

SOUND_PATH = "Data/Sound/Sfx/Global"
CATEGORY = "CREATURE_MOLYNX"
BILE_CATEGORY_FILE = "creature_bile_demon"
MOLYNX_CATEGORY_FILE = "creature_molynx"
BILE_BANK, MOLYNX_BANK = "Global\\Bile", "Global\\Molynx"
HEADER = struct.Struct("<II16sIIII")
FORMAT_MP2_22K = 0x24105622
RATE = 22050
BITRATE = "64k"
BANK_COUNT_AT, BANK_ENTRY = 0x18, 11


# ---- reading and writing ---------------------------------------------------------------------------

def read_sdt(data: bytes) -> list[tuple[bytes, bytes]]:
    """[(40-byte header, sample data)] in file order."""
    (n,) = struct.unpack_from("<I", data, 0)
    offsets = struct.unpack_from(f"<{n}I", data, 4)
    out = []
    for off in offsets:
        size_h, size = struct.unpack_from("<II", data, off)
        if size_h != HEADER.size:
            raise ValueError(f"sample header of {size_h} bytes at {off}, expected {HEADER.size}")
        out.append((data[off:off + HEADER.size], data[off + HEADER.size:off + HEADER.size + size]))
    return out


def write_sdt(entries: list[tuple[bytes, bytes]]) -> bytes:
    n = len(entries)
    pos = 4 + 4 * n
    offsets, body = [], []
    for header, payload in entries:
        offsets.append(pos)
        body += [header, payload]
        pos += len(header) + len(payload)
    return struct.pack(f"<I{n}I", n, *offsets) + b"".join(body)


def sample_name(header: bytes) -> str:
    return HEADER.unpack(header)[2].split(b"\0")[0].decode("latin-1")


def is_blank(header: bytes) -> bool:
    return HEADER.unpack(header)[1] == 0 and sample_name(header) == "Blank"


def molynx_bank_map(bile_bank: bytes) -> bytes:
    """The Bile Demon's BANK.map with its own bank renamed (the strings are u32-length-prefixed,
    NUL included)."""
    (n,) = struct.unpack_from("<I", bile_bank, BANK_COUNT_AT)
    at = BANK_COUNT_AT + 4 + n * BANK_ENTRY
    head, names = bile_bank[:at], []
    for _ in range(n):
        (length,) = struct.unpack_from("<I", bile_bank, at)
        names.append(bile_bank[at + 4:at + 4 + length - 1].decode("latin-1"))
        at += 4 + length
    if at != len(bile_bank):
        raise ValueError("BANK.map has bytes after its bank names")
    if names.count(BILE_BANK) != 1:
        raise ValueError(f"BANK.map names {names}, expected {BILE_BANK!r} once")
    names = [MOLYNX_BANK if name == BILE_BANK else name for name in names]
    return head + b"".join(struct.pack("<I", len(s) + 1) + s.encode("latin-1") + b"\0" for s in names)


def encode_mp2(wav: Path) -> tuple[bytes, int]:
    """(MPEG-2 Layer II, 22050 Hz mono, of the file; its length in samples)."""
    with wave.open(str(wav)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"{wav.name} is not 22050 Hz mono 16-bit")
        frames = w.getnframes()
    data = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-c:a", "mp2", "-ar", str(RATE),
                           "-ac", "1", "-b:a", BITRATE, "-f", "mp2", "-"],
                          capture_output=True, check=True).stdout
    if data[:2] != b"\xff\xf5":
        raise ValueError(f"{wav.name}: the encoder did not write MPEG-2 Layer II")
    return data, frames


def molynx_sdt(bile_sdt: bytes, wav_dir: Path) -> bytes:
    """The Bile Demon's .sdt with every sample replaced by Molynx's for the same role (ROLES, in
    file order with the blanks left out); blanks are kept as they are."""
    entries = read_sdt(bile_sdt)
    samples = [i for i, (h, _d) in enumerate(entries) if not is_blank(h)]
    roles = list(ROLES.items())
    if len(samples) != len(roles):
        raise ValueError(f"the bank has {len(samples)} samples, ROLES {len(roles)}")
    out = list(entries)
    for i, (bile_name, role) in zip(samples, roles):
        header = entries[i][0]
        size_h, _size, name, fmt, z1, _pcm, z2 = HEADER.unpack(header)
        if sample_name(header) not in ((bile_name + ".mp2")[:15], bile_name[:15]):
            raise ValueError(f"sample {i} is {sample_name(header)!r}, ROLES expects {bile_name!r}")
        if fmt != FORMAT_MP2_22K:
            raise ValueError(f"sample {i} has format {fmt:#x}, expected {FORMAT_MP2_22K:#x}")
        data, frames = encode_mp2(Path(wav_dir) / f"{role}.wav")
        out[i] = (HEADER.pack(size_h, len(data), name, fmt, z1, frames * 2, z2), data)
    return write_sdt(out)


def sound_files(game_global: Path, wav_dir: Path) -> dict[str, bytes]:
    """Every sound file of the mod folder (relative path -> bytes), from the game's Bile Demon files
    in `game_global` (Data/Sound/Sfx/Global) and Molynx's WAVs in `wav_dir`."""
    g = Path(game_global)
    files = {
        f"{SOUND_PATH}/{MOLYNX_CATEGORY_FILE}BANK.map": molynx_bank_map((g / f"{BILE_CATEGORY_FILE}BANK.map").read_bytes()),
        f"{SOUND_PATH}/{MOLYNX_CATEGORY_FILE}SFX.map": (g / f"{BILE_CATEGORY_FILE}SFX.map").read_bytes(),
    }
    for kind in ("HD", "HW"):
        files[f"{SOUND_PATH}/Molynx{kind}.sdt"] = molynx_sdt((g / f"Bile{kind}.sdt").read_bytes(), wav_dir)
    return files
