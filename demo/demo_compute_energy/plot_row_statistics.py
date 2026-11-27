#!/usr/bin/env python3
"""Calcola statistiche per riga e crea il grafico della media con barre d'errore.

Il file di input deve contenere una riga per ciascun valore x: la prima colonna
è x e le colonne successive sono i valori da mediare. Sono ammesse righe vuote
e commenti che iniziano con '#'.

Esempio:
    python plot_row_statistics.py train_eVST.dat gen_eVST.dat
"""

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        type=Path,
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="due file di dati separati da spazi",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="file immagine in output (default: <file1>_<file2>_statistics.png)",
    )
    parser.add_argument(
        "--uncertainty",
        choices=("sem", "std"),
        default="sem",
        help="tipo di barre d'errore: errore standard (sem, default) "
        "o deviazione standard campionaria (std)",
    )
    args = parser.parse_args()

    results = []
    for path in args.files:
        try:
            # Accetta anche il segno meno Unicode (−) che può comparire nei file.
            contents = path.read_text(encoding="utf-8").replace("\N{MINUS SIGN}", "-")
            data = np.loadtxt(io.StringIO(contents), comments="#", ndmin=2)
        except OSError as exc:
            parser.error(f"impossibile leggere '{path}': {exc}")
        except ValueError as exc:
            parser.error(f"formato dati non valido in '{path}': {exc}")

        if data.shape[1] < 2:
            parser.error(f"'{path}' deve contenere almeno due colonne")
        if data.shape[0] == 0:
            parser.error(f"'{path}' non contiene righe di dati")
        if data.shape[1] < 3:
            parser.error(f"'{path}': servono almeno due valori per riga "
                         "per calcolare l'incertezza")

        x = data[:, 0]
        samples = data[:, 1:]
        means = np.mean(samples, axis=1)
        stds = np.std(samples, axis=1, ddof=1)
        uncertainties = (
            stds / np.sqrt(samples.shape[1])
            if args.uncertainty == "sem"
            else stds
        )
        results.append((path, x, means, uncertainties))

        print(f"{'file':<30} {'x':>12} {'media':>16} {'incertezza':>16}")
    for path, x, means, uncertainties in results:
        for x_value, mean, uncertainty in zip(x, means, uncertainties):
            print(f"{path.name:<30} {x_value:12g} {mean:16g} {uncertainty:16g}")

    first, second = args.files
    output = args.output or first.with_name(
        f"{first.stem}_{second.stem}_statistics.png"
    )
    for path, x, means, uncertainties in results:
        plt.errorbar(
            x, means, yerr=uncertainties, fmt="o-", capsize=4, label=path.name
        )
    plt.xlabel("Valore della prima colonna")
    plt.ylabel("Media dei valori")
    plt.title(f"Media e incertezza ({args.uncertainty.upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Grafico salvato in: {output}")


if __name__ == "__main__":
    main()
