#!/usr/bin/env python3
"""
Genera configurazioni completamente random di SiO2 (posizioni uniformi nella
cella cubica, nessun vincolo di distanza minima) e le scrive in formato
LAMMPS data, nello stile di write_data.

Default: 3000 atomi (1000 Si, 2000 O), cella cubica di lato 36.21762010130505 A,
tipi 1 = Si, 2 = O (stessa convenzione di lammps2xyz.py).

Uso:
    python random_sio2.py                         # un file: random_sio2_0000.data
    python random_sio2.py -n 10 --seed 42         # dieci configurazioni
    python random_sio2.py -n 5 -p conf --outdir configs
    python random_sio2.py --style charge --q-si 2.4 --q-o -1.2   # con le cariche

Poi, per passare all'energia:
    python lammps2xyz.py random_sio2_0000.data
    python energia_asio2.py random_sio2_0000.xyz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

MASS_SI = 28.0855
MASS_O = 15.9994


def write_lammps_data(path, types, pos, L, style, charges=None):
    n = len(types)
    with open(path, "w") as f:
        f.write("LAMMPS data file via write_data, timestep = 0, units = metal\n\n")
        f.write(f"{n} atoms\n2 atom types\n\n")
        for ax in "xyz":
            f.write(f"0 {L!r} {ax}lo {ax}hi\n")
        f.write(f"\nMasses\n\n1 {MASS_SI} # Si\n2 {MASS_O} # O\n\n")
        f.write(f"Atoms # {style}\n\n")
        for i in range(n):
            x, y, z = pos[i]
            if style == "atomic":
                f.write(f"{i+1} {types[i]} {x:.16g} {y:.16g} {z:.16g} 0 0 0\n")
            else:  # charge
                f.write(f"{i+1} {types[i]} {charges[types[i]]} "
                        f"{x:.16g} {y:.16g} {z:.16g} 0 0 0\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--nconf", type=int, default=1, help="numero di configurazioni")
    ap.add_argument("--natoms", type=int, default=3000, help="atomi totali (default 3000)")
    ap.add_argument("--n-oxygen", type=int, default=2000, help="numero di O (default 2000)")
    ap.add_argument("--box", type=float, default=36.21762010130505,
                    help="lato della cella cubica in A")
    ap.add_argument("--seed", type=int, default=None, help="seed del generatore")
    ap.add_argument("-p", "--prefix", default="random_sio2", help="prefisso dei file")
    ap.add_argument("--outdir", default=".", help="cartella di output")
    ap.add_argument("--style", choices=["atomic", "charge"], default="atomic",
                    help="atom_style scritto nel file (default: atomic)")
    ap.add_argument("--q-si", type=float, help="carica di Si (solo con --style charge)")
    ap.add_argument("--q-o", type=float, help="carica di O (solo con --style charge)")
    a = ap.parse_args()

    if a.n_oxygen > a.natoms:
        sys.exit("--n-oxygen non puo' superare --natoms.")
    charges = None
    if a.style == "charge":
        if a.q_si is None or a.q_o is None:
            sys.exit("Con --style charge servono --q-si e --q-o.")
        charges = {1: a.q_si, 2: a.q_o}

    rng = np.random.default_rng(a.seed)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_si = a.natoms - a.n_oxygen
    base_types = np.array([1] * n_si + [2] * a.n_oxygen)

    for k in range(a.nconf):
        types = rng.permutation(base_types)           # specie mescolate
        pos = rng.uniform(0.0, a.box, size=(a.natoms, 3))
        out = outdir / f"{a.prefix}_{k:04d}.data"
        write_lammps_data(out, types, pos, a.box, a.style, charges)
        print(f"{out}: {a.natoms} atomi (Si: {n_si}, O: {a.n_oxygen}), L = {a.box!r}")


if __name__ == "__main__":
    main()