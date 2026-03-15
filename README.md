# `complete_sf2_instruments.py`

Fill missing **(bank, program)** preset slots in an SF2 by cloning presets that reuse the most appropriate existing instruments (e.g. “Fiddle” falling back to “Violin”).

By default, percussion/drum presets (typically bank \(\ge 128\)) are preserved and written after the 128 melodic presets.

## Requirements

- Python 3.8+
- No third-party dependencies

## Usage

Complete bank 0 programs 0–127:

```bash
python scripts\complete_sf2_instruments.py "input.sf2" -o "output.sf2"
```

Choose a different bank and/or program range:

```bash
python scripts\complete_sf2_instruments.py "input.sf2" -o "output.sf2" --bank 0 --programs 0-127
```

Dry run (show what would be filled):

```bash
python scripts\complete_sf2_instruments.py "input.sf2" -o "output.sf2" --dry-run
```

Melodic-only output (drop percussion banks):

```bash
python scripts\complete_sf2_instruments.py "input.sf2" -o "output.sf2" --no-keep-percussion
```

Allow cross-category fallback (melodic can borrow from percussion and vice versa):

```bash
python scripts\complete_sf2_instruments.py "input.sf2" -o "output.sf2" --allow-cross-bank-fallback
```

Fallback selection strategy:

```bash
python scripts\complete_sf2_instruments.py "input.sf2" -o "output.sf2" --fallback size_times_count
```

## Notes

- This script modifies the **preset tables** (phdr/pbag/pgen). It keeps instrument/sample data intact.
- “Best” fallback is heuristic-based; use `--dry-run` to preview what it will clone.
