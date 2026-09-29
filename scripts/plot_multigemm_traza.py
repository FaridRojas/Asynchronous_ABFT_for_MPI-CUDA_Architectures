#!/usr/bin/env python3
"""
Figura de la línea de tiempo de una secuencia de multiplicaciones
independientes emitidas sin espera intermedia.

Lee las trazas tabulares que produce nvprof --print-gpu-trace y dibuja el
diagrama de ocupación de cada flujo. Se dibuja aquí en lugar de capturar la
pantalla de un perfilador para que la figura quede en el idioma y el estilo del
resto del libro y pueda regenerarse desde los datos.

El color identifica la operación y la fila el flujo, de modo que el
solapamiento se lee directamente: una barra de verificación de un color situada
bajo una barra de multiplicación de otro color es la verificación de una
operación transcurriendo mientras se calcula la siguiente.

    python3 scripts/plot_multigemm_traza.py [--size 2048] [--in DIR] [--out DIR]
"""

import argparse
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})

# Un color por operación, legibles en impresión monocroma por su orden de gris.
COLORES = ["#2a6f9e", "#c2703d", "#5a9367", "#8c6d9e"]
GEMM = "sgemm"
INICIO_OP = "k_col_checksum_A_part"


def leer(ruta: Path):
    txt = [l for l in open(ruta) if not l.startswith("==")]
    filas = list(csv.DictReader(io.StringIO("".join(txt))))[1:]   # 2ª fila = unidades
    ks = []
    for f in filas:
        n = (f.get("Name") or "").strip()
        if not n or "memcpy" in n or "memset" in n:
            continue
        ks.append({"t": float(f["Start"]) * 1000.0,
                   "d": float(f["Duration"]),
                   "s": int(f["Stream"]),
                   "n": n.split("(")[0]})
    ks.sort(key=lambda k: k["t"])
    return ks


def ultima_ronda(ks):
    """El banco ejecuta una ronda descartada y otra registrada; se dibuja la
    segunda, ya con la biblioteca y los relojes en régimen."""
    inicios = [k["t"] for k in ks if k["n"] == INICIO_OP]
    if len(inicios) < 2:
        return ks
    corte = inicios[len(inicios) // 2]
    return [k for k in ks if k["t"] >= corte - 1e-6]


def asignar_operacion(ks):
    # Un solo par de flujos: la operación se deduce del orden, y cada suma de
    # comprobación de A marca el comienzo de una.
    computo = min(k["s"] for k in ks)
    op = -1
    for k in ks:
        k["computo"] = (k["s"] == computo)
        if k["n"] == INICIO_OP:
            op += 1
        k["op"] = max(op, 0)
    # Las multiplicaciones se numeran por su propio orden, que puede adelantar
    # al de las codificaciones cuando no hay espera intermedia.
    g = 0
    for k in ks:
        if k["n"].startswith(GEMM):
            k["op"] = g
            g += 1


def panel(ax, ks, titulo, anotar=False):
    ks = ultima_ronda(ks)
    asignar_operacion(ks)
    t0 = min(k["t"] for k in ks)

    flujos = sorted({k["s"] for k in ks})
    filas = {s: i for i, s in enumerate(flujos)}
    etiquetas = []
    for s in flujos:
        es_comp = any(k["computo"] for k in ks if k["s"] == s)
        base = "cómputo" if es_comp else "verificación"
        etiquetas.append(base)

    for k in ks:
        y = len(filas) - 1 - filas[k["s"]]
        ax.barh(y, max(k["d"], 0.004), left=k["t"] - t0, height=0.62,
                color=COLORES[k["op"] % len(COLORES)], edgecolor="black",
                linewidth=0.25, hatch="" if k["n"].startswith(GEMM) else "///",
                zorder=3)

    # Señala un caso concreto: la comprobación de la salida de una operación
    # transcurriendo mientras se multiplica la siguiente.
    if anotar:
        ver = {k["op"]: k for k in ks if k["n"] == "k_actual_row"}
        gem = {k["op"]: k for k in ks if k["n"].startswith(GEMM)}
        if 0 in ver and 1 in gem:
            v, g = ver[0], gem[1]
            yv = len(filas) - 1 - filas[v["s"]]
            ax.annotate("la comprobación de la salida de la operación 1\n"
                        "transcurre mientras se multiplica la 2",
                        xy=(v["t"] - t0 + v["d"] / 2, yv + 0.34),
                        xytext=(v["t"] - t0 - 5.6, yv + 1.95),
                        fontsize=6.8, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                        color="black",
                                        connectionstyle="arc3,rad=0.2"))

    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels(list(reversed(etiquetas)), fontsize=7)
    ax.set_ylim(-0.6, len(filas) + (1.35 if anotar else -0.25))
    ax.set_xlim(-0.2, max(k["t"] - t0 + k["d"] for k in ks) * 1.02)
    ax.set_title(titulo, fontsize=8.5, pad=4)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4, zorder=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="2048")
    ap.add_argument("--in", dest="src", default="docs/multigemm/trazas")
    ap.add_argument("--out", dest="out", default="docs/presentation/libro/figuras")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)

    # La traza muestra que la comprobación de una operación transcurre bajo la
    # multiplicación de la siguiente.
    ruta = src / f"traza_{a.size}_encadenado.csv"
    if not ruta.exists():
        raise SystemExit(f"falta {ruta}")

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    panel(ax, leer(ruta),
          "Cuatro multiplicaciones independientes emitidas sin espera intermedia",
          anotar=True)
    axes = [ax]
    axes[-1].set_xlabel("Tiempo desde el inicio de la secuencia (ms)", fontsize=8.5)

    leyenda = [Patch(facecolor=COLORES[i], edgecolor="black", linewidth=0.4,
                     label=f"operación {i + 1}") for i in range(4)]
    leyenda += [Patch(facecolor="white", edgecolor="black", linewidth=0.4,
                      label="multiplicación"),
                Patch(facecolor="white", edgecolor="black", linewidth=0.4,
                      hatch="///", label="verificación")]
    fig.legend(handles=leyenda, loc="upper center", ncol=6, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(bottom=0.30)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "res_multigemm_traza.pdf", bbox_inches="tight", pad_inches=0.03)
    print(f"  {out}/res_multigemm_traza.pdf")


if __name__ == "__main__":
    main()
