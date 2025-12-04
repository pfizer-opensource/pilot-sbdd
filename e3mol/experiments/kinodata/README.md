## Data Processing

By default, we omit explicit hydrogens when creating the Protein-Ligand complexes.
The distance cutoff is set to 5.0 Angstrom.

```bash
python e3mol/experiments/kinodata/process_data.py --base-dir $BASE_DIR --out-dir $OUT_DIR --no-H --dist-cutoff 5.0
```