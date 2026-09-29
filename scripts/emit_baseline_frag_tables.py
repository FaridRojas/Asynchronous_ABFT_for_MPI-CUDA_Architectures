#!/usr/bin/env python3
"""
Genera las dos tablas de anexo con el costo de fragmentar la multiplicación.

Lee el CSV de scripts/run_baseline_frag_felix.sbatch, que mide únicamente la
fase sin protección, y emite dos `longtable` en LaTeX: una para el régimen
cuadrado y otra con la dimensión de contracción fijada.

Cada fila es un tamaño. La primera columna numérica da el tiempo absoluto sin
fragmentar, y las restantes el incremento porcentual que introduce cada valor
de F respecto a ese mismo tiempo. Presentar las columnas como incrementos y no
como tiempos absolutos es deliberado: lo que la tabla debe permitir leer es el
peaje de fragmentar, no el tiempo de la multiplicación, que la sección de
resultados ya recoge.

El CSV registra la F *efectiva*, y por debajo de 1024 la política consciente
del tamaño la colapsa a uno, de modo que las seis ejecuciones de esas filas
miden la misma configuración. Por eso la F solicitada se asigna por el orden en
que el barrido las emite y no por la columna del CSV.

    python3 scripts/emit_baseline_frag_tables.py [--csv RUTA] [--out RUTA]
"""

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

FRAGS = [1, 2, 4, 8, 16, 32]


def cargar(ruta: Path):
    """Devuelve {(fase, M): [ms por F solicitado, en orden]} y el conjunto de
    claves cuyo F efectivo quedó colapsado a uno en todas sus ejecuciones.

    La fase se deduce del orden de las filas y no de las dimensiones, porque el
    barrido cuadrado y el de contracción fija coinciden exactamente en la forma
    $1024^3$: distinguirlas por sus dimensiones fundiría esa medición en un
    solo grupo de doce ejecuciones y la perdería en ambas tablas."""
    filas = list(csv.DictReader(open(ruta)))
    if len(filas) % (2 * len(FRAGS)):
        raise SystemExit(f"{ruta}: {len(filas)} filas no son dos barridos "
                         f"completos de múltiplos de {len(FRAGS)}")
    corte = len(filas) // 2
    grupos, efectivos = OrderedDict(), {}
    for i, r in enumerate(filas):
        k = ("sq" if i < corte else "ns", int(r["M"]))
        grupos.setdefault(k, []).append(float(r["baseline_mean_ms"]))
        efectivos.setdefault(k, []).append(int(r["frags_per_rank"]))
    colapsados = {k for k, v in efectivos.items() if set(v) == {1}}
    return grupos, colapsados


def _tiempo(x):
    """Tiempo en milisegundos con cuatro cifras significativas.

    Un valor fijo de tres decimales daría «114.900» para una medición cuya
    dispersión propia ronda el uno por ciento, lo que atribuiría a la cifra una
    resolución que no tiene."""
    dec = 3 if x < 10 else (2 if x < 100 else 1)
    return f"${x:.{dec}f}$"


def _num(x, dec):
    """Número en modo matemático. El libro usa el punto como separador decimal
    y ningún separador de millares."""
    if dec != 1:
        return _tiempo(x)
    # Un incremento por debajo de la resolución de la columna no lleva signo:
    # "-0.0" se leería como un error de formato antes que como un cero.
    return f"${abs(x):.1f}$" if abs(x) < 0.05 else f"${x:+.1f}$"


def fila(m, tiempos, colapsado):
    marca = "$^{\\dagger}$" if colapsado else ""
    celdas = [str(m) + marca, _num(tiempos[0], 3)]
    for t in tiempos[1:]:
        celdas.append(_num((t / tiempos[0] - 1) * 100, 1))
    return " & ".join(celdas) + " \\\\"


def tabla(grupos, colapsados, fase, etiqueta, caption):
    filas = [(k, v) for k, v in grupos.items() if k[0] == fase]
    filas.sort(key=lambda x: x[0][1])
    cab = " & ".join([f"$F={f}$" for f in FRAGS[1:]])
    out = [
        "\\begin{longtable}{@{}rrrrrrr@{}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{etiqueta}}} \\\\",
        "\\toprule",
        f"$M$ & $F=1$ (ms) & {cab} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{7}{@{}l}{\\footnotesize\\itshape "
        "(continuación de la página anterior)}\\\\",
        "\\toprule",
        f"$M$ & $F=1$ (ms) & {cab} \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{7}{r@{}}{\\footnotesize\\itshape continúa\\ldots}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for k, tiempos in filas:
        out.append(fila(k[1], tiempos, k in colapsados))
    out.append("\\end{longtable}")
    return "\n".join(out), filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/fragsweep/abft_baseline_frag.csv")
    ap.add_argument("--out", default="/tmp/claude-1000/tablas_basefrag.tex")
    a = ap.parse_args()

    grupos, colapsados = cargar(Path(a.csv))
    sq, fsq = tabla(
        grupos, colapsados, "sq", "tab:anx-basefrag-sq",
        "Costo de fragmentar la multiplicación sin protección, régimen "
        "cuadrado. La segunda columna es el tiempo de la ejecución no "
        "fragmentada; las restantes, la variación porcentual que introduce "
        "cada número de fragmentos respecto de ese mismo tiempo, de modo que "
        "un valor negativo indica una ejecución más rápida que sin fragmentar.")
    ns, fns = tabla(
        grupos, colapsados, "ns", "tab:anx-basefrag-ns",
        "Costo de fragmentar la multiplicación sin protección con la dimensión "
        "de contracción fijada en $K = 1024$. Las columnas tienen el mismo "
        "significado que en la Tabla~\\ref{tab:anx-basefrag-sq}.")

    nota = ("\n\\noindent\\footnotesize $^{\\dagger}$ Por debajo de "
            "$\\max(M,K,N) = 1024$ la política consciente del tamaño colapsa "
            "$F$ a uno, de modo que las seis ejecuciones de esas filas miden la "
            "misma configuración y su dispersión corresponde al ruido de "
            "medición.\\normalsize\n")

    Path(a.out).write_text(sq + "\n" + nota + "\n" + ns + "\n" + nota)
    print(f"escrito {a.out}")
    print(f"  cuadradas   : {len(fsq)} tamaños")
    print(f"  no cuadradas: {len(fns)} tamaños")

    for nombre, filas in (("cuadradas", fsq), ("no cuadradas", fns)):
        peor = max(filas, key=lambda x: x[1][3] / x[1][0])
        print(f"  {nombre}: F=8 cuesta hasta "
              f"{(peor[1][3] / peor[1][0] - 1) * 100:+.1f}% en M={peor[0][1]}")


if __name__ == "__main__":
    main()
