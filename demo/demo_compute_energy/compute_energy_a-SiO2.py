#!/usr/bin/env python3
"""
Energia di una configurazione di a-SiO2.

  E = E_corto_raggio (cutoff 8.0 A) + E_Coulomb (Fennell damped shifted force,
      alpha = 0.25 A^-1, cutoff 8.0 A) + termine di self del DSF

PBC nelle tre direzioni (cella ortorombica, convenzione di minima immagine).

FORMATO FILE DI INPUT (una riga per atomo):
    Si   x  y  z
    O    x  y  z
Sono ammessi: righe vuote, commenti con '#', l'header del formato .xyz
(la riga col numero di atomi e quella di commento vengono ignorate) e una
riga opzionale   box Lx Ly Lz   per le dimensioni della cella.
In alternativa la cella si passa da riga di comando con --box.

Uso:
    python compute_energy_a-SiO2.py config.txt --box 24.0 24.0 24.0
    python compute_energy_a-SiO2.py config.txt            # se il file contiene "box Lx Ly Lz"

!!! I PARAMETRI DEL POTENZIALE DI JAKSE NON SONO INCLUSI (vedi PARAMS) !!!
Vanno copiati dalla tabella del lavoro originale di Jakse et al. (o dal
file di potenziale usato in Yang & Koda 2026). Lo script si ferma finche'
restano None, per non produrre energie calcolate con numeri inventati.
"""
import argparse
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erfc

KE = 14.399645352  # costante di Coulomb in eV*A/e^2

# ---------------------------------------------------------------------------
# PARAMETRI DEL POTENZIALE  -> DA COMPILARE
# Forma assunta del termine a corto raggio (Buckingham, stile BKS):
#     V_ij(r) = A_ij * exp(-r / rho_ij) - C_ij / r^6 + D_ij / r^24
# (D_ij = 0 se il potenziale non ha il termine repulsivo 1/r^24.)
# Unita': A in eV, rho in A, C in eV*A^6, D in eV*A^24, cariche in unita' di e.
# Se la forma funzionale di Jakse e' diversa, modificare solo short_range().
# ---------------------------------------------------------------------------
PARAMS = {
    "charges": {"Si": 2.4, "O": -1.2},
    "pairs": {
        ("O", "O"): dict(A=0.0120, rho=0.2630, s=3.6430, C=85.0840, D=0.0),
        ("Si", "O"):  dict(A=0.0070, rho=0.1560, s=2.5419 , C=46.2930, D=0.0),
        ("Si", "Si"):   dict(A=0.0012, rho=0.0460, s=1.4408, C=25.1873, D=0.0),
    },
}

SPECIES = ["Si", "O"]


def read_config(path):
    species, pos, box = [], [], None
    with open(path) as f:
        for line in f:
            s = line.split("#")[0].strip()
            if not s:
                continue
            t = s.split()
            if t[0].lower() == "box" and len(t) >= 4:
                box = np.array(t[1:4], dtype=float)
                continue
            if len(t) >= 4:
                try:
                    xyz = [float(v) for v in t[1:4]]
                except ValueError:
                    continue  # riga di commento dell'header xyz
                species.append(t[0].capitalize() if t[0].lower() != "o" else "O")
                pos.append(xyz)
    if not species:
        sys.exit("Nessun atomo letto dal file.")
    return species, np.array(pos), box


def check_params(params):
    missing = [k for k, v in params["charges"].items() if v is None]
    for pair, d in params["pairs"].items():
        missing += [f"{pair}:{k}" for k, v in d.items() if v is None]
    if missing:
        sys.exit("Parametri del potenziale mancanti in PARAMS: " + ", ".join(missing))


def short_range(r, A, rho, s, C, D):
    return A * np.exp(-r * s / rho) - C / r**6 + D / r**24


def energy(species, pos, box, params, rc=8.0, alpha=0.25,
           shift_short_range=False, self_term=True):
    check_params(params)
    if rc > box.min() / 2:
        print(f"ATTENZIONE: rc={rc} > L/2={box.min()/2:.3f}: la minima immagine "
              "non basta, l'energia non e' affidabile.", file=sys.stderr)

    idx = np.array([SPECIES.index(s) for s in species])
    q_type = np.array([params["charges"][s] for s in SPECIES], dtype=float)
    q = q_type[idx]

    # matrici di parametri indicizzate per tipo
    P = {k: np.zeros((2, 2)) for k in ("A", "rho", "s", "C", "D")}
    for (a, b), d in params["pairs"].items():
        ia, ib = SPECIES.index(a), SPECIES.index(b)
        for k in P:
            P[k][ia, ib] = P[k][ib, ia] = d[k]

    # coppie entro rc con PBC
    pos = pos % box
    tree = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(rc, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[i] - pos[j]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    ti, tj = idx[i], idx[j]

    # --- corto raggio ---
    A, rho, s, C, D = (P[k][ti, tj] for k in ("A", "rho", "s", "C", "D"))
    e_sr = short_range(r, A, rho, s, C, D)
    if shift_short_range:
        e_sr -= short_range(rc, A, rho, s, C, D)
    E_sr = e_sr.sum()

    # --- Coulomb: Fennell-Gezelter damped shifted force ---
    erfc_rc = erfc(alpha * rc)
    e_shift = erfc_rc / rc
    f_shift = erfc_rc / rc**2 + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * rc) ** 2) / rc
    qq = q[i] * q[j]
    e_c = KE * qq * (erfc(alpha * r) / r - e_shift + f_shift * (r - rc))
    E_coul = e_c.sum()
    E_self = 0.0
    if self_term:
        E_self = -KE * (e_shift / 2 + alpha / np.sqrt(np.pi)) * np.sum(q**2)

    return dict(short_range=E_sr, coulomb_pairs=E_coul, coulomb_self=E_self,
                total=E_sr + E_coul + E_self, n_atoms=len(species), n_pairs=len(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--box", nargs="+", type=float,
                    help="Lx Ly Lz (o un solo valore per cella cubica)")
    ap.add_argument("--rc", type=float, default=8.0, help="cutoff [A] (default 8.0)")
    ap.add_argument("--alpha", type=float, default=0.25, help="damping DSF [1/A]")
    ap.add_argument("--shift-sr", action="store_true",
                    help="trasla il termine a corto raggio a zero al cutoff "
                         "(default: semplice troncamento)")
    ap.add_argument("--no-self", action="store_true",
                    help="esclude il termine di self del DSF")
    a = ap.parse_args()

    species, pos, box = read_config(a.file)
    if a.box:
        box = np.array(a.box * 3 if len(a.box) == 1 else a.box, dtype=float)
    if box is None:
        sys.exit("Dimensioni della cella non fornite: usa --box o la riga 'box Lx Ly Lz'.")

    res = energy(species, pos, box, PARAMS, rc=a.rc, alpha=a.alpha,
                 shift_short_range=a.shift_sr, self_term=not a.no_self)

    n = res["n_atoms"]
    print(f"Atomi: {n}  (Si: {species.count('Si')}, O: {species.count('O')})  "
          f"coppie entro rc: {res['n_pairs']}")
    print(f"E corto raggio   = {res['short_range']:.6f} eV")
    print(f"E Coulomb (coppie)= {res['coulomb_pairs']:.6f} eV")
    print(f"E Coulomb (self) = {res['coulomb_self']:.6f} eV")
    print(f"E totale         = {res['total']:.6f} eV  ({res['total']/n:.6f} eV/atomo)")


if __name__ == "__main__":
    main()