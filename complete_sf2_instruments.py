#!/usr/bin/env python3
"""
Complete an SF2 sound font's instrument set by filling missing (bank, program)
slots with presets that reuse the most appropriate existing instruments
(e.g. Fiddle -> Violin, Viola -> Violin). By default, percussion/drum presets
(bank >= 128) are left unchanged and output after the 128 melodic presets.

Usage:
  python complete_sf2_instruments.py input.sf2 -o output.sf2
  python complete_sf2_instruments.py input.sf2 -o output.sf2 --bank 0 --programs 0-127
  python complete_sf2_instruments.py input.sf2 -o output.sf2 --dry-run
  python complete_sf2_instruments.py input.sf2 -o output.sf2 --no-keep-percussion  # melodic only
  python complete_sf2_instruments.py input.sf2 -o output.sf2 --allow-cross-bank-fallback  # allow percussion<->melodic fallback
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# --- RIFF / SF2 (standalone, no dependency on trim_soundfont_to_midi) ---

class RiffError(Exception):
    pass


@dataclass
class Chunk:
    fourcc: str
    data: memoryview
    children: Optional[List[Chunk]] = None
    list_type: Optional[str] = None


def _read_u32le(buf: memoryview, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _read_u16le(buf: memoryview, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0]


def _read_s16le(buf: memoryview, off: int) -> int:
    return struct.unpack_from("<h", buf, off)[0]


def _parse_riff(buf: bytes) -> Chunk:
    mv = memoryview(buf)
    if len(mv) < 12:
        raise RiffError("File too small for RIFF")
    if mv[0:4].tobytes() != b"RIFF":
        raise RiffError("Not a RIFF file")
    size = _read_u32le(mv, 4)
    form = mv[8:12].tobytes().decode("ascii", "replace")
    root_data = mv[12 : 8 + size]
    root = Chunk("RIFF", root_data, children=[], list_type=form)
    root.children = _parse_children(root_data, 0, len(root_data))
    return root


def _parse_children(mv: memoryview, start: int, end: int) -> List[Chunk]:
    out: List[Chunk] = []
    off = start
    while off + 8 <= end:
        fourcc = mv[off : off + 4].tobytes().decode("ascii", "replace")
        size = _read_u32le(mv, off + 4)
        data_start = off + 8
        data_end = data_start + size
        if data_end > end:
            break
        data = mv[data_start:data_end]
        ch = Chunk(fourcc, data, children=None, list_type=None)
        if fourcc in ("RIFF", "LIST"):
            if len(data) < 4:
                raise RiffError(f"{fourcc} chunk too small")
            list_type = data[0:4].tobytes().decode("ascii", "replace")
            ch.list_type = list_type
            ch.children = _parse_children(data, 4, len(data))
        out.append(ch)
        off = data_end + (size & 1)
    return out


def _find_list(root: Chunk, list_type: str) -> Optional[Chunk]:
    if root.fourcc in ("RIFF", "LIST") and root.list_type == list_type:
        return root
    if not root.children:
        return None
    for c in root.children:
        got = _find_list(c, list_type)
        if got:
            return got
    return None


def _find_child(parent: Chunk, fourcc: str) -> Optional[Chunk]:
    if not parent.children:
        return None
    for c in parent.children:
        if c.fourcc == fourcc:
            return c
    return None


def _read_zstr(b: bytes) -> str:
    s = b.split(b"\x00", 1)[0]
    return s.decode("ascii", "replace")


def _sf2_normalize_bank(bank: int, bank_style: str) -> int:
    """SF2 wBank is 16-bit; normalize for GM (7-bit) or MMA (14-bit)."""
    if bank_style == "gm":
        return bank & 0x7F
    return bank & 0x7F7F


@dataclass
class SF2Parsed:
    phdr: List[Tuple[str, int, int, int]]  # name, preset, bank, pbag_ndx
    pbag: List[Tuple[int, int]]
    pgen: List[Tuple[int, int]]
    inst: List[Tuple[str, int]]  # name, ibag_ndx
    ibag: List[Tuple[int, int]]
    igen: List[Tuple[int, int]]
    shdr: List[Dict]
    smpl_bytes: bytes
    pdta_chunks_order: List[str]
    info_chunks: Dict[str, bytes]
    raw_pmod: bytes
    raw_imod: bytes


def _parse_sf2_raw(path: str) -> SF2Parsed:
    data = open(path, "rb").read()
    root = _parse_riff(data)
    if root.list_type != "sfbk":
        raise RiffError("Not an SF2 (RIFF sfbk)")

    sdta = _find_list(root, "sdta")
    pdta = _find_list(root, "pdta")
    if not sdta or not pdta:
        raise RiffError("Malformed SF2: missing sdta/pdta")

    smpl_chunk = _find_child(sdta, "smpl")
    if not smpl_chunk:
        raise RiffError("Malformed SF2: missing sdta/smpl")
    smpl_bytes = smpl_chunk.data.tobytes()

    def need(name: str) -> Chunk:
        c = _find_child(pdta, name)
        if not c:
            raise RiffError(f"Malformed SF2: missing pdta/{name}")
        return c

    phdr_d = need("phdr").data
    pbag_d = need("pbag").data
    pgen_d = need("pgen").data
    inst_d = need("inst").data
    ibag_d = need("ibag").data
    igen_d = need("igen").data
    shdr_d = need("shdr").data
    pmod_chunk = _find_child(pdta, "pmod")
    imod_chunk = _find_child(pdta, "imod")
    raw_pmod = pmod_chunk.data.tobytes() if pmod_chunk else b""
    raw_imod = imod_chunk.data.tobytes() if imod_chunk else b""

    rec_sz = 38
    phdr_list: List[Tuple[str, int, int, int]] = []
    for off in range(0, len(phdr_d), rec_sz):
        if off + rec_sz > len(phdr_d):
            break
        name = _read_zstr(phdr_d[off : off + 20].tobytes())
        preset = _read_u16le(phdr_d, off + 20)
        bank = _read_u16le(phdr_d, off + 22)
        pbag_ndx = _read_u16le(phdr_d, off + 24)
        phdr_list.append((name, preset, bank, pbag_ndx))

    rec_sz = 22
    inst_list: List[Tuple[str, int]] = []
    for off in range(0, len(inst_d), rec_sz):
        if off + rec_sz > len(inst_d):
            break
        name = _read_zstr(inst_d[off : off + 20].tobytes())
        ibag_ndx = _read_u16le(inst_d, off + 20)
        inst_list.append((name, ibag_ndx))

    rec_sz = 46
    shdr_list: List[Dict] = []
    for off in range(0, len(shdr_d), rec_sz):
        if off + rec_sz > len(shdr_d):
            break
        name = _read_zstr(shdr_d[off : off + 20].tobytes())
        start = _read_u32le(shdr_d, off + 20)
        end = _read_u32le(shdr_d, off + 24)
        start_loop = _read_u32le(shdr_d, off + 28)
        end_loop = _read_u32le(shdr_d, off + 32)
        rate = _read_u32le(shdr_d, off + 36)
        orig_pitch = int(shdr_d[off + 40])
        pitch_corr = struct.unpack_from("<b", shdr_d, off + 41)[0]
        shdr_list.append({
            "name": name, "start": start, "end": end,
            "start_loop": start_loop, "end_loop": end_loop, "rate": rate,
            "orig_pitch": orig_pitch, "pitch_corr": pitch_corr,
            "raw": shdr_d[off : off + rec_sz].tobytes(),
        })

    def parse_bag(bag_d: memoryview) -> List[Tuple[int, int]]:
        out = []
        for off in range(0, len(bag_d), 4):
            if off + 4 > len(bag_d):
                break
            out.append((_read_u16le(bag_d, off), _read_u16le(bag_d, off + 2)))
        return out

    def parse_gen(gen_d: memoryview) -> List[Tuple[int, int]]:
        out = []
        for off in range(0, len(gen_d), 4):
            if off + 4 > len(gen_d):
                break
            out.append((_read_u16le(gen_d, off), _read_s16le(gen_d, off + 2)))
        return out

    pbag_list = parse_bag(pbag_d)
    pgen_list = parse_gen(pgen_d)
    ibag_list = parse_bag(ibag_d)
    igen_list = parse_gen(igen_d)

    info_chunks: Dict[str, bytes] = {}
    info_list = _find_list(root, "INFO")
    if info_list and info_list.children:
        for c in info_list.children:
            if c.fourcc not in ("RIFF", "LIST") and len(c.data) >= 4:
                info_chunks[c.fourcc] = c.data.tobytes()

    return SF2Parsed(
        phdr=phdr_list,
        pbag=pbag_list,
        pgen=pgen_list,
        inst=inst_list,
        ibag=ibag_list,
        igen=igen_list,
        shdr=shdr_list,
        smpl_bytes=smpl_bytes,
        pdta_chunks_order=["phdr", "pbag", "pmod", "pgen", "inst", "ibag", "imod", "igen", "shdr"],
        info_chunks=info_chunks,
        raw_pmod=raw_pmod,
        raw_imod=raw_imod,
    )


def _write_sf2(
    out_path: str,
    sf2: SF2Parsed,
    new_phdr: List[Tuple[str, int, int, int]],
    new_pbag: List[Tuple[int, int]],
    new_pgen: List[Tuple[int, int]],
    new_inst: List[Tuple[str, int]],
    new_ibag: List[Tuple[int, int]],
    new_igen: List[Tuple[int, int]],
    new_shdr: List[Dict],
    new_smpl: bytes,
) -> None:
    def w_u16le(v: int) -> bytes:
        return struct.pack("<H", v & 0xFFFF)

    def w_s16le(v: int) -> bytes:
        return struct.pack("<h", v)

    def w_u32le(v: int) -> bytes:
        return struct.pack("<I", v & 0xFFFFFFFF)

    def chunk(fourcc: str, data: bytes) -> bytes:
        pad = len(data) & 1
        return fourcc.encode("ascii") + w_u32le(len(data)) + data + (b"\x00" * pad)

    info_parts = []
    for k, v in sf2.info_chunks.items():
        info_parts.append(k.encode("ascii")[:4] + w_u32le(len(v)) + v)
    info_data = b"".join(info_parts)
    if not info_data:
        info_data = b"ifil\x04\x00\x00\x00\x02\x00\x01\x00"
    info_list = b"LIST" + w_u32le(4 + len(info_data)) + b"INFO" + info_data
    if len(info_list) & 1:
        info_list += b"\x00"

    smpl_chunk = b"smpl" + w_u32le(len(new_smpl)) + new_smpl
    if len(new_smpl) & 1:
        smpl_chunk += b"\x00"
    sdta_list = b"LIST" + w_u32le(4 + len(smpl_chunk)) + b"sdta" + smpl_chunk
    if len(sdta_list) & 1:
        sdta_list += b"\x00"

    phdr_data = b""
    for name, preset, bank, pbag_ndx in new_phdr:
        phdr_data += (name[:20].encode("ascii") + b"\x00" * 20)[:20]
        phdr_data += w_u16le(preset)
        phdr_data += w_u16le(bank)
        phdr_data += w_u16le(pbag_ndx)
        phdr_data += w_u32le(0) * 3
    pbag_data = b""
    for g, m in new_pbag:
        pbag_data += w_u16le(g) + w_u16le(m)
    pmod_data = sf2.raw_pmod if sf2.raw_pmod else (w_u16le(0) + w_u16le(0) + w_u16le(0) + w_u16le(0))
    pgen_data = b""
    for op, amt in new_pgen:
        pgen_data += w_u16le(op) + w_s16le(amt)
    inst_data = b""
    for name, ibag_ndx in new_inst:
        inst_data += (name[:20].encode("ascii") + b"\x00" * 20)[:20]
        inst_data += w_u16le(ibag_ndx)
    ibag_data = b""
    for g, m in new_ibag:
        ibag_data += w_u16le(g) + w_u16le(m)
    imod_data = sf2.raw_imod if sf2.raw_imod else (w_u16le(0) + w_u16le(0) + w_u16le(0) + w_u16le(0))
    igen_data = b""
    for op, amt in new_igen:
        igen_data += w_u16le(op) + w_s16le(amt)
    shdr_data = b""
    for h in new_shdr:
        name = (h.get("name", "") or "EOS")[:20]
        shdr_data += (name.encode("ascii") + b"\x00" * 20)[:20]
        shdr_data += w_u32le(h["start"]) + w_u32le(h["end"])
        shdr_data += w_u32le(h["start_loop"]) + w_u32le(h["end_loop"])
        shdr_data += w_u32le(h.get("rate", 44100))
        shdr_data += bytes([h.get("orig_pitch", 60) & 0xFF])
        shdr_data += struct.pack("<b", h.get("pitch_corr", 0))
        shdr_data += w_u16le(0)
        is_eos = h.get("end") == h.get("start")
        shdr_data += w_u16le(0 if is_eos else 1)
    if len(new_shdr) == 0 or (new_shdr[-1].get("end") != new_shdr[-1].get("start")):
        shdr_data += b"\x00" * 46
    pdta_body = (
        chunk("phdr", phdr_data) + chunk("pbag", pbag_data) + chunk("pmod", pmod_data) +
        chunk("pgen", pgen_data) + chunk("inst", inst_data) + chunk("ibag", ibag_data) +
        chunk("imod", imod_data) + chunk("igen", igen_data) + chunk("shdr", shdr_data)
    )
    pdta_list = b"LIST" + w_u32le(4 + len(pdta_body)) + b"pdta" + pdta_body
    if len(pdta_list) & 1:
        pdta_list += b"\x00"

    body = info_list + sdta_list + pdta_list
    riff = b"RIFF" + w_u32le(4 + len(body)) + b"sfbk" + body
    if len(riff) & 1:
        riff += b"\x00"
    with open(out_path, "wb") as f:
        f.write(riff)


# SF2 generator IDs
SF2_GEN_INSTRUMENT = 41
SF2_GEN_SAMPLE_ID = 53

# GM percussion: drum kits are typically at bank 128 (0x80)
PERCUSSION_BANK_MIN = 128

# Fallback preset selection strategies
FALLBACK_FIRST = "first"  # first in family order (lowest program number in 8-program family)
FALLBACK_LARGEST_TOTAL_SIZE = "largest_total_size"
FALLBACK_NUM_SAMPLES = "num_samples"
FALLBACK_SIZE_TIMES_COUNT = "size_times_count"  # new default


def _is_percussion_bank(bank: int) -> bool:
    """True if this bank is typically used for percussion/drum kits in GM."""
    return bank >= PERCUSSION_BANK_MIN


# --- GM Level 1 names (programs 0-127) ---
GM_NAMES: List[str] = [
    "Acoustic Grand Piano",
    "Bright Acoustic Piano",
    "Electric Grand Piano",
    "Honky-tonk Piano",
    "Electric Piano 1",
    "Electric Piano 2",
    "Harpsichord",
    "Clavinet",
    "Celesta",
    "Glockenspiel",
    "Music Box",
    "Vibraphone",
    "Marimba",
    "Xylophone",
    "Tubular Bells",
    "Dulcimer",
    "Drawbar Organ",
    "Percussive Organ",
    "Rock Organ",
    "Church Organ",
    "Reed Organ",
    "Accordion",
    "Harmonica",
    "Tango Accordion",
    "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)",
    "Electric Guitar (clean)",
    "Electric Guitar (muted)",
    "Overdriven Guitar",
    "Distortion Guitar",
    "Guitar Harmonics",
    "Acoustic Bass",
    "Electric Bass (finger)",
    "Electric Bass (pick)",
    "Fretless Bass",
    "Slap Bass 1",
    "Slap Bass 2",
    "Synth Bass 1",
    "Synth Bass 2",
    "Violin",
    "Viola",
    "Cello",
    "Contrabass",
    "Tremolo Strings",
    "Pizzicato Strings",
    "Orchestral Harp",
    "Timpani",
    "String Ensemble 1",
    "String Ensemble 2",
    "Synth Strings 1",
    "Synth Strings 2",
    "Choir Aahs",
    "Voice Oohs",
    "Synth Choir",
    "Orchestra Hit",
    "Trumpet",
    "Trombone",
    "Tuba",
    "Muted Trumpet",
    "French Horn",
    "Brass Section",
    "Synth Brass 1",
    "Synth Brass 2",
    "Soprano Sax",
    "Alto Sax",
    "Tenor Sax",
    "Baritone Sax",
    "Oboe",
    "English Horn",
    "Bassoon",
    "Clarinet",
    "Piccolo",
    "Flute",
    "Recorder",
    "Pan Flute",
    "Blown Bottle",
    "Shakuhachi",
    "Whistle",
    "Ocarina",
    "Lead 1 (square)",
    "Lead 2 (sawtooth)",
    "Lead 3 (calliope)",
    "Lead 4 (chiff)",
    "Lead 5 (charang)",
    "Lead 6 (voice)",
    "Lead 7 (fifths)",
    "Lead 8 (bass+lead)",
    "Pad 1 (new age)",
    "Pad 2 (warm)",
    "Pad 3 (polysynth)",
    "Pad 4 (choir)",
    "Pad 5 (bowed)",
    "Pad 6 (metallic)",
    "Pad 7 (halo)",
    "Pad 8 (sweep)",
    "FX 1 (rain)",
    "FX 2 (soundtrack)",
    "FX 3 (crystal)",
    "FX 4 (atmosphere)",
    "FX 5 (brightness)",
    "FX 6 (goblins)",
    "FX 7 (echoes)",
    "FX 8 (sci-fi)",
    "Sitar",
    "Banjo",
    "Shamisen",
    "Koto",
    "Kalimba",
    "Bag pipe",
    "Fiddle",
    "Shanai",
    "Tinkle Bell",
    "Agogo",
    "Steel Drums",
    "Woodblock",
    "Taiko Drum",
    "Melodic Tom",
    "Synth Drum",
    "Reverse Cymbal",
    "Guitar Fret Noise",
    "Breath Noise",
    "Seashore",
    "Bird Tweet",
    "Telephone Ring",
    "Helicopter",
    "Applause",
    "Gunshot",
]


def _gm_name(program: int) -> str:
    if 0 <= program < len(GM_NAMES):
        return GM_NAMES[program]
    return f"Program {program}"


# --- Fallback: for each program, ordered list of preferred substitute programs ---
# When a preset is missing, we clone the first fallback whose preset exists.
# By family: piano 0-7, chrom.perc 8-15, organ 16-23, guitar 24-31, bass 32-39,
# strings 40-47, ensemble 48-55, brass 56-63, reed 64-71, pipe 72-79,
# synth lead 80-87, pad 88-95, fx 96-103, ethnic 104-111, perc 112-119, sfx 120-127.
def _family_rep(program: int) -> int:
    if program <= 7:
        return 0
    if program <= 15:
        return 8
    if program <= 23:
        return 16
    if program <= 31:
        return 24
    if program <= 39:
        return 32
    if program <= 47:
        return 40
    if program <= 55:
        return 48
    if program <= 63:
        return 56
    if program <= 71:
        return 64
    if program <= 79:
        return 72
    if program <= 87:
        return 80
    if program <= 95:
        return 88
    if program <= 103:
        return 96
    if program <= 111:
        # Ethnic: Fiddle->Violin, Shanai->Soprano Sax, etc.
        if program == 110:
            return 40
        if program == 111:
            return 64
        return 48
    if program <= 119:
        return 112
    return 120


def _build_fallback_list(program: int) -> List[int]:
    """Ordered list of program numbers to try when cloning a missing preset."""
    out: List[int] = [program]
    rep = _family_rep(program)
    if rep != program:
        out.append(rep)
    # Same broader category (e.g. all strings 40-47 -> 40)
    for start in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]:
        if start <= program < start + 8 and start != rep and start not in out:
            out.append(start)
    if 0 not in out:
        out.append(0)
    return out


# Precompute fallback for 0-127
GM_FALLBACK: Dict[int, List[int]] = {p: _build_fallback_list(p) for p in range(128)}


def _existing_bank_program_set(sf2: SF2Parsed, bank_style: str = "gm") -> Set[Tuple[int, int]]:
    """Set of (normalized_bank, program) for all non-terminal presets."""
    out: Set[Tuple[int, int]] = set()
    for i in range(len(sf2.phdr) - 1):
        name, preset_num, bank_num, pbag_ndx = sf2.phdr[i]
        nb = _sf2_normalize_bank(bank_num, bank_style)
        out.add((nb, preset_num & 0x7F))
    return out


def _existing_bank_program_set_exact(sf2: SF2Parsed) -> Set[Tuple[int, int]]:
    """Set of (bank, program) for all non-terminal presets (raw bank, no normalization)."""
    out: Set[Tuple[int, int]] = set()
    for i in range(len(sf2.phdr) - 1):
        name, preset_num, bank_num, pbag_ndx = sf2.phdr[i]
        out.add((bank_num, preset_num & 0x7F))
    return out


def _get_preset_zones(sf2: SF2Parsed, preset_index: int) -> List[List[Tuple[int, int]]]:
    """Return list of zone generator lists for the given preset (each zone = list of (oper, amt))."""
    pbag_ndx = sf2.phdr[preset_index][3]
    next_pbag = sf2.phdr[preset_index + 1][3]
    zones: List[List[Tuple[int, int]]] = []
    for bag_i in range(pbag_ndx, next_pbag):
        gen_start = sf2.pbag[bag_i][0]
        gen_end = sf2.pbag[bag_i + 1][0] if bag_i + 1 < len(sf2.pbag) else len(sf2.pgen)
        zones.append(list(sf2.pgen[gen_start:gen_end]))
    return zones


def _preset_has_instrument(sf2: SF2Parsed, preset_index: int) -> bool:
    """True if the preset has at least one zone with an instrument generator."""
    pbag_ndx = sf2.phdr[preset_index][3]
    next_pbag = sf2.phdr[preset_index + 1][3]
    for bag_i in range(pbag_ndx, next_pbag):
        gen_start = sf2.pbag[bag_i][0]
        gen_end = sf2.pbag[bag_i + 1][0] if bag_i + 1 < len(sf2.pbag) else len(sf2.pgen)
        for op, _ in sf2.pgen[gen_start:gen_end]:
            if op == SF2_GEN_INSTRUMENT:
                return True
    return False


def _preset_sample_stats(sf2: SF2Parsed, preset_index: int) -> Tuple[int, int]:
    """Return (total_sample_size_in_words, num_distinct_samples) for the preset.
    Size is the sum of (end - start) over all sample headers used by the preset's instruments."""
    sample_ids: Set[int] = set()
    pbag_ndx = sf2.phdr[preset_index][3]
    next_pbag = sf2.phdr[preset_index + 1][3]
    for bag_i in range(pbag_ndx, next_pbag):
        gen_start = sf2.pbag[bag_i][0]
        gen_end = sf2.pbag[bag_i + 1][0] if bag_i + 1 < len(sf2.pbag) else len(sf2.pgen)
        gens = dict(sf2.pgen[gen_start:gen_end])
        inst_id = gens.get(SF2_GEN_INSTRUMENT)
        if inst_id is None or inst_id < 0 or inst_id >= len(sf2.inst) - 1:
            continue
        ibag_ndx = sf2.inst[inst_id][1]
        next_ibag = sf2.inst[inst_id + 1][1]
        for ib in range(ibag_ndx, next_ibag):
            igen_start = sf2.ibag[ib][0]
            igen_end = sf2.ibag[ib + 1][0] if ib + 1 < len(sf2.ibag) else len(sf2.igen)
            for op, amt in sf2.igen[igen_start:igen_end]:
                if op == SF2_GEN_SAMPLE_ID and 0 <= amt < len(sf2.shdr):
                    sample_ids.add(amt)
    total_size = 0
    for sid in sample_ids:
        if sid < len(sf2.shdr):
            h = sf2.shdr[sid]
            start = int(h.get("start", 0))
            end = int(h.get("end", 0))
            total_size += max(0, end - start)
    return (total_size, len(sample_ids))


def _preset_index_by_bank_program(
    sf2: SF2Parsed,
    existing: Set[Tuple[int, int]],
    bank: int,
    program: int,
    bank_style: str,
    *,
    exact_bank: bool = False,
    allow_cross_bank_fallback: bool = False,
) -> int:
    """Return preset index (into sf2.phdr) for (bank, program), or -1 if not found.
    If exact_bank is True, only match when bank_num == bank.
    If allow_cross_bank_fallback is False, never use percussion for melodic or vice versa."""
    prog = program & 0x7F
    target_is_percussion = _is_percussion_bank(bank)

    def bank_ok(bank_num: int) -> bool:
        if allow_cross_bank_fallback:
            return True
        preset_is_percussion = _is_percussion_bank(bank_num)
        return preset_is_percussion == target_is_percussion

    if exact_bank:
        key = (bank, prog)
        if key not in existing:
            return -1
        for i in range(len(sf2.phdr) - 1):
            name, preset_num, bank_num, pbag_ndx = sf2.phdr[i]
            if bank_num == bank and (preset_num & 0x7F) == prog and bank_ok(bank_num):
                return i
        return -1
    nb = _sf2_normalize_bank(bank, bank_style)
    key = (nb, prog)
    if key not in existing:
        return -1
    for i in range(len(sf2.phdr) - 1):
        name, preset_num, bank_num, pbag_ndx = sf2.phdr[i]
        if (
            _sf2_normalize_bank(bank_num, bank_style) == nb
            and (preset_num & 0x7F) == prog
            and bank_ok(bank_num)
        ):
            return i
    return -1


def _family_range(program: int) -> range:
    """All 8 program numbers in the same GM family as program (0-127)."""
    prog = program & 0x7F
    family_start = (prog // 8) * 8
    return range(family_start, min(family_start + 8, 128))


def _resolve_fallback_preset_index(
    sf2: SF2Parsed,
    existing: Set[Tuple[int, int]],
    bank: int,
    program: int,
    bank_style: str,
    *,
    exact_bank: bool = False,
    allow_cross_bank_fallback: bool = False,
    fallback_strategy: str = FALLBACK_SIZE_TIMES_COUNT,
) -> int:
    """Return preset index to clone. Candidates are all existing presets in the same
    8-program family; if none exist, all presets in the same bank category are used.
    Selection is by fallback_strategy: first, largest_total_size, num_samples, size_times_count."""
    target_is_percussion = _is_percussion_bank(bank)

    def bank_ok(bank_num: int) -> bool:
        if allow_cross_bank_fallback:
            return True
        return _is_percussion_bank(bank_num) == target_is_percussion

    candidates: List[int] = []
    # Phase 1: all existing presets in the same 8-program family (by program number)
    for try_prog in _family_range(program):
        idx = _preset_index_by_bank_program(
            sf2,
            existing,
            bank,
            try_prog,
            bank_style,
            exact_bank=exact_bank,
            allow_cross_bank_fallback=allow_cross_bank_fallback,
        )
        if idx >= 0 and _preset_has_instrument(sf2, idx):
            candidates.append(idx)
    for try_prog in _family_range(program):
        idx = _preset_index_by_bank_program(
            sf2,
            existing,
            bank,
            try_prog,
            bank_style,
            exact_bank=exact_bank,
            allow_cross_bank_fallback=allow_cross_bank_fallback,
        )
        if idx >= 0 and idx not in candidates:
            candidates.append(idx)

    # Phase 2: only if no instruments in family, add all existing presets in bank category
    if not candidates:
        for i in range(len(sf2.phdr) - 1):
            if _preset_has_instrument(sf2, i) and bank_ok(sf2.phdr[i][2]):
                candidates.append(i)
        for i in range(len(sf2.phdr) - 1):
            if bank_ok(sf2.phdr[i][2]) and i not in candidates:
                candidates.append(i)

    if not candidates:
        return -1
    if fallback_strategy == FALLBACK_FIRST:
        return candidates[0]

    # Score each candidate; pick one with highest score
    def score(idx: int) -> int:
        total, num = _preset_sample_stats(sf2, idx)
        if fallback_strategy == FALLBACK_LARGEST_TOTAL_SIZE:
            return total
        if fallback_strategy == FALLBACK_NUM_SAMPLES:
            return num
        if fallback_strategy == FALLBACK_SIZE_TIMES_COUNT:
            return total * num
        return 0

    best_idx = candidates[0]
    best_score = score(best_idx)
    for idx in candidates[1:]:
        s = score(idx)
        if s > best_score:
            best_score = s
            best_idx = idx
    return best_idx


def complete_sf2(
    sf2: SF2Parsed,
    target_bank: int,
    program_lo: int,
    program_hi: int,
    bank_style: str = "gm",
    *,
    keep_percussion: bool = True,
    allow_cross_bank_fallback: bool = False,
    fallback_strategy: str = FALLBACK_SIZE_TIMES_COUNT,
) -> Tuple[
    List[Tuple[str, int, int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    """
    Build new_phdr, new_pbag, new_pgen so that GM bank 0 works like 1mgm.sf2:
    exactly one preset per (target_bank, program) for program in [program_lo, program_hi],
    in order. For each program, use existing preset zones if that preset exists and has
    instruments; otherwise use fallback. Then append any other original presets (e.g.
    percussion at bank 128).     When keep_percussion is True (default), percussion banks
    (bank >= 128) are left completely unchanged and copied through.
    When allow_cross_bank_fallback is False (default), percussion presets
    are never used as fallback for melodic and vice versa.
    fallback_strategy controls how the best candidate is chosen: first, largest_total_size,
    num_samples, or size_times_count (default).
    """
    existing = _existing_bank_program_set(sf2, bank_style)
    existing_exact = _existing_bank_program_set_exact(sf2)
    nb = _sf2_normalize_bank(target_bank, bank_style)

    # Melodic target: only exact (target_bank, program) when keep_percussion so (128,0) is not treated as (0,0)
    if keep_percussion:
        target_set = {(target_bank, p) for p in range(program_lo, min(program_hi + 1, 128))}
        use_exact_bank = True
        lookup_set = existing_exact
    else:
        target_set = {(nb, p) for p in range(program_lo, min(program_hi + 1, 128))}
        use_exact_bank = False
        lookup_set = existing

    # Build canonical GM bank: one preset per program 0-127 in order
    presets_with_zones: List[Tuple[str, int, int, List[List[Tuple[int, int]]]]] = []

    for program in range(program_lo, min(program_hi + 1, 128)):
        idx = _preset_index_by_bank_program(
            sf2,
            lookup_set,
            target_bank,
            program,
            bank_style,
            exact_bank=use_exact_bank,
            allow_cross_bank_fallback=allow_cross_bank_fallback,
        )
        if idx >= 0 and _preset_has_instrument(sf2, idx):
            name, preset_num, bank_num, _ = sf2.phdr[idx]
            zones = _get_preset_zones(sf2, idx)
            presets_with_zones.append((name, program, target_bank, zones))
        else:
            fallback_idx = _resolve_fallback_preset_index(
                sf2,
                lookup_set,
                target_bank,
                program,
                bank_style,
                exact_bank=use_exact_bank,
                allow_cross_bank_fallback=allow_cross_bank_fallback,
                fallback_strategy=fallback_strategy,
            )
            if fallback_idx >= 0:
                zones = _get_preset_zones(sf2, fallback_idx)
                presets_with_zones.append((_gm_name(program), program, target_bank, zones))

    # Append any original presets not in melodic target (e.g. percussion at bank 128)
    # When keep_percussion: use raw (bank_num, preset) so (128,0) is kept
    for i in range(len(sf2.phdr) - 1):
        name, preset_num, bank_num, _ = sf2.phdr[i]
        if keep_percussion:
            key = (bank_num, preset_num & 0x7F)
        else:
            key = (_sf2_normalize_bank(bank_num, bank_style), preset_num & 0x7F)
        if key not in target_set:
            zones = _get_preset_zones(sf2, i)
            presets_with_zones.append((name, preset_num, bank_num, zones))

    # Sort: melodic target first (bank 0 in program order), then percussion (bank >= 128), then others
    def sort_key(x: Tuple[str, int, int, List]) -> Tuple[int, int, int]:
        name, preset_num, bank_num, _ = x
        in_target = (bank_num, preset_num) in target_set if keep_percussion else (
            _sf2_normalize_bank(bank_num, bank_style), preset_num
        ) in target_set
        if in_target and bank_num == target_bank:
            return (0, 0, preset_num)
        if keep_percussion and _is_percussion_bank(bank_num):
            return (1, bank_num, preset_num)
        return (2, bank_num * 128 + preset_num, 0)

    presets_with_zones.sort(key=sort_key)

    new_phdr = []
    new_pbag = []
    new_pgen = []
    pbag_count = 0
    pgen_count = 0
    for name, preset_num, bank_num, zones in presets_with_zones:
        new_phdr.append((name, preset_num, bank_num, pbag_count))
        for zone_gens in zones:
            new_pbag.append((pgen_count, 0))
            new_pgen.extend(zone_gens)
            pgen_count += len(zone_gens)
        pbag_count += len(zones)
    new_phdr.append(("EOP", 0, 0, pbag_count))
    new_pbag.append((pgen_count, 0))

    return (new_phdr, new_pbag, new_pgen)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Complete an SF2 instrument set by filling missing (bank, program) slots with clones of the most appropriate existing presets."
    )
    ap.add_argument("input", help="Input SF2 file")
    ap.add_argument("-o", "--output", required=True, help="Output SF2 file")
    ap.add_argument("--bank", type=int, default=0, help="Target bank to complete (default 0)")
    ap.add_argument(
        "--programs",
        default="0-127",
        metavar="LO-HI",
        help="Program range to fill (default 0-127)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be filled and exit without writing",
    )
    ap.add_argument(
        "--keep-percussion",
        action="store_true",
        default=True,
        help="Preserve percussion/drum presets (bank >= 128) unchanged; output them after melodic (default: on)",
    )
    ap.add_argument(
        "--no-keep-percussion",
        action="store_false",
        dest="keep_percussion",
        help="Do not preserve percussion; only output the 128 melodic presets",
    )
    ap.add_argument(
        "--allow-cross-bank-fallback",
        action="store_true",
        help="Allow using percussion presets as fallback for melodic and vice versa (default: off)",
    )
    ap.add_argument(
        "--fallback",
        dest="fallback_strategy",
        choices=["first", "largest_total_size", "num_samples", "size_times_count"],
        default="size_times_count",
        help="How to choose fallback preset: first (GM order), largest_total_size, num_samples, size_times_count (default)",
    )
    args = ap.parse_args()

    program_lo, program_hi = 0, 127
    if args.programs:
        part = args.programs.strip().split("-")
        if len(part) == 2:
            try:
                program_lo = max(0, int(part[0].strip()))
                program_hi = min(127, int(part[1].strip()))
            except ValueError:
                pass

    try:
        sf2 = _parse_sf2_raw(args.input)
    except RiffError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    existing = _existing_bank_program_set(sf2, "gm")
    nb = _sf2_normalize_bank(args.bank, "gm")
    target_set = set()
    for p in range(program_lo, program_hi + 1):
        target_set.add((nb, p))
    missing = target_set - existing

    if args.dry_run:
        if not missing:
            print("No missing slots; soundfont already has all target programs.")
            return
        print(f"Would fill {len(missing)} missing (bank={args.bank}, programs {program_lo}-{program_hi}):")
        for (b, prog) in sorted(missing):
            idx = _resolve_fallback_preset_index(sf2, existing, args.bank, prog, "gm")
            src = "?"
            if idx >= 0 and idx < len(sf2.phdr) - 1:
                src = sf2.phdr[idx][0]
            print(f"  {prog} {_gm_name(prog)} <- {src}")
        return

    new_phdr, new_pbag, new_pgen = complete_sf2(
        sf2,
        args.bank,
        program_lo,
        program_hi,
        "gm",
        keep_percussion=args.keep_percussion,
        allow_cross_bank_fallback=args.allow_cross_bank_fallback,
        fallback_strategy=args.fallback_strategy,
    )
    # Keep inst, ibag, igen, shdr, smpl unchanged
    new_inst = sf2.inst
    new_ibag = sf2.ibag
    new_igen = sf2.igen
    new_shdr = sf2.shdr
    new_smpl = sf2.smpl_bytes

    _write_sf2(
        args.output,
        sf2,
        new_phdr,
        new_pbag,
        new_pgen,
        new_inst,
        new_ibag,
        new_igen,
        new_shdr,
        new_smpl,
    )
    n_presets = len(new_phdr) - 1
    n_orig = len(sf2.phdr) - 1
    if n_presets >= n_orig:
        print(f"Wrote {args.output}: {n_presets} presets ({n_presets - n_orig} added).")
    else:
        print(f"Wrote {args.output}: {n_presets} presets (canonical GM bank {args.bank}).")


if __name__ == "__main__":
    main()
