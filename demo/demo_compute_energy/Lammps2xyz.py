#!/usr/bin/env python3
"""
Converte file LAMMPS data (output di write_data) in .xyz compatibile con
energia_asio2.py.

Il file xyz prodotto ha:
    riga 1: numero di atomi
    riga 2: box Lx Ly Lz          (letta da energia_asio2.py)
    poi:    Specie x y z          (coordinate in A, origine della cella in 0)

Mappatura tipo -> specie:
  - di default dalle masse nella sezione Masses (~28.09 -> Si, ~16.00 -> O);
  - oppure esplicita con --map "1=Si,2=O" (ha la precedenza).

atom_style: letto dal commento della riga "Atoms # atomic" (scritto da
write_data); se manca si usa --style (default: atomic).
Supportati: atomic, charge, full, molecular, bond, angle.

Celle triclinee (xy, xz, yz diversi da 0) non sono supportate, perche'
energia_asio2.py lavora con celle ortorombiche.

Uso:
    python lammps2xyz.py conf1.data conf2.data          # -> conf1.xyz, conf2.xyz
    python lammps2xyz.py conf.data -o out.xyz --map "1=Si,2=O"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# indici delle colonne (tipo, x, y, z) per ciascun atom_style
STYLE_COLS = {
    "atomic": (1, 2, 3, 4),
    "charge": (1, 3, 4, 5),
    "full": (2, 4, 5, 6),
    "molecular": (2, 3, 4, 5),
    "bond": (2, 3, 4, 5),
    "angle": (2, 3, 4, 5),
}

MASS_TO_SPECIES = [(28.0855, "Si"), (15.999, "O")]


def parse_data(path):
    n_atoms = None
    lo, hi, tilt = np.zeros(3), np.zeros(3), np.zeros(3)
    masses, atoms = {}, []
    style, section = None, None

    with open(path) as f:
        lines = f.read().splitlines()

    for raw in lines[1:]:  # la prima riga e' un titolo
        comment = raw.split("#", 1)[1].strip() if "#" in raw else ""
        s = raw.split("#", 1)[0].strip()
        if not s:
            continue
        t = s.split()

        if t[0][0].isalpha():  # intestazione di sezione
            section = t[0]
            if section == "Atoms" and comment:
                style = comment.split()[0].lower()
            continue

        if section is None:  # header
            if s.endswith("atoms") and len(t) == 2:
                n_atoms = int(t[0])
            elif "xlo" in t:
                lo[0], hi[0] = float(t[0]), float(t[1])
            elif "ylo" in t:
                lo[1], hi[1] = float(t[0]), float(t[1])
            elif "zlo" in t:
                lo[2], hi[2] = float(t[0]), float(t[1])
            elif "xy" in t:
                tilt = np.array(t[:3], dtype=float)
        elif section == "Masses":
            masses[int(t[0])] = float(t[1])
        elif section == "Atoms":
            atoms.append(t)

    return n_atoms, lo, hi, tilt, masses, atoms, style


def species_map(masses, map_arg):
    mapping = {}
    if map_arg:
        for item in map_arg.split(","):
            k, v = item.split("=")
            mapping[int(k)] = v.strip().capitalize() if v.strip().lower() != "o" else "O"
        return mapping
    for typ, m in masses.items():
        name, ref = min(((nm, abs(m - ref_m)) for ref_m, nm in MASS_TO_SPECIES),
                        key=lambda x: x[1])
        if ref > 1.0:
            sys.exit(f"Massa {m} del tipo {typ} non riconosciuta come Si/O: usa --map.")
        mapping[typ] = name
    return mapping


def convert(path, out, style_arg, map_arg):
    n_atoms, lo, hi, tilt, masses, atoms, style = parse_data(path)

    if np.any(np.abs(tilt) > 1e-8):
        sys.exit(f"{path}: cella triclinea non supportata (tilt = {tilt}).")
    style = style or style_arg
    if style not in STYLE_COLS:
        sys.exit(f"{path}: atom_style '{style}' non supportato; usa --style.")
    if n_atoms is not None and n_atoms != len(atoms):
        sys.exit(f"{path}: attesi {n_atoms} atomi, letti {len(atoms)}.")

    mapping = species_map(masses, map_arg)
    ci_t, ci_x, ci_y, ci_z = STYLE_COLS[style]
    atoms.sort(key=lambda t: int(t[0]))  # ordina per id

    box = hi - lo
    types = [int(t[ci_t]) for t in atoms]
    pos = np.array([[float(t[ci_x]), float(t[ci_y]), float(t[ci_z])] for t in atoms])
    pos = (pos - lo) % box  # origine in 0 e riavvolgimento nella cella

    with open(out, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"box {box[0]:.8f} {box[1]:.8f} {box[2]:.8f}\n")
        for typ, p in zip(types, pos):
            f.write(f"{mapping[typ]:2s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

    n_si = sum(mapping[t] == "Si" for t in types)
    print(f"{path} -> {out}: {len(atoms)} atomi (Si: {n_si}, O: {len(atoms) - n_si}), "
          f"box = {box[0]:.4f} {box[1]:.4f} {box[2]:.4f}, style = {style}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="file data di LAMMPS")
    ap.add_argument("-o", "--output", help="file di output (solo con un input)")
    ap.add_argument("--map", help='mappa tipo=specie, es. "1=Si,2=O"')
    ap.add_argument("--style", default="atomic",
                    help="atom_style se non indicato nel file (default: atomic)")
    a = ap.parse_args()

    if a.output and len(a.files) > 1:
        sys.exit("-o si puo' usare solo con un singolo file di input.")
    for p in a.files:
        out = a.output or str(Path(p).with_suffix(".xyz"))
        convert(p, out, a.style, a.map)


if __name__ == "__main__":
    main()