#!/usr/bin/env python3
"""
Figuras del capítulo de metodología del libro.

El capítulo describe el trabajo realizado sin una sola ilustración, mientras que
el de resultados lleva nueve figuras. Estas cuatro cubren una fase cada una y se
generan como PDF para insertarse con \\includegraphics, de modo que el preámbulo
del documento no necesita ningún paquete adicional.

    python3 scripts/plot_metodologia_figs.py [--out DIR]

Las magnitudes de la figura de inyección se derivan del formato IEEE-754 de
precisión simple y coinciden con los rangos de bits que implementa
src/kernels/swifi.cuh.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})

GRIS   = "#4a4a4a"
CLARO  = "#d9d9d9"
MEDIO  = "#a6a6a6"
ACENTO = "#2a6f9e"
CALIDO = "#c2703d"


def _guardar(fig, out: Path, nombre: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{nombre}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {nombre}.pdf")


# ---------------------------------------------------------------------------
def fig_inyeccion_zonas(out: Path) -> None:
    """Formato IEEE-754 por zonas y magnitud relativa de la perturbación.

    Para un valor normal x = 1.m * 2^(E-127), invertir el bit i del significando
    altera el valor en 2^(i-23)/1.m veces su magnitud, es decir, entre 2^(i-24)
    y 2^(i-23): el significando bajo cubre (2^-24, 2^-11] y el alto (2^-11, 2^-1].
    Invertir el bit e del exponente multiplica o divide el valor por 2^(2^(e-23)),
    con una perturbacion relativa que va de 1/2 (bit 23 de 1 a 0) a 2^128 - 1
    (bit 30 de 0 a 1); si el exponente resultante vale 255, el valor pasa a ser
    +-inf o NaN. Invertir el signo produce una diferencia de dos veces el valor.
    La figura ordena las cuatro zonas por esa magnitud, que es lo que determina
    si la verificacion puede distinguirlas del ruido de redondeo.
    """
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7.0, 3.5), gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.55})

    zonas = [("signo", 31, 31, CALIDO), ("exponente", 30, 23, MEDIO),
             ("significando alto", 22, 13, ACENTO), ("significando bajo", 12, 0, CLARO)]

    ax.set_xlim(-0.5, 32.5); ax.set_ylim(-0.9, 2.35); ax.axis("off")
    for nombre, hi, lo, color in zonas:
        x0, ancho = 31 - hi, hi - lo + 1
        ax.add_patch(Rectangle((x0, 0), ancho, 1, facecolor=color,
                               edgecolor="black", linewidth=0.7))
        if ancho >= 4:
            ax.text(x0 + ancho / 2, 0.5, nombre, ha="center", va="center",
                    fontsize=8,
                    color="black" if color in (CLARO, MEDIO) else "white")
        else:
            # Una zona de un solo bit no admite la etiqueta dentro; va encima,
            # con un trazo que la conecta con su celda.
            ax.text(x0 + ancho / 2, 1.62, nombre, ha="center", va="bottom",
                    fontsize=8, color=color)
            ax.plot([x0 + ancho / 2, x0 + ancho / 2], [1.02, 1.58],
                    color=color, linewidth=0.8)
        etiqueta = f"{hi}" if hi == lo else f"{hi}–{lo}"
        ax.text(x0 + ancho / 2, -0.42, f"bits {etiqueta}", ha="center",
                va="center", fontsize=7.5, color=GRIS)
    for i in range(33):
        ax.plot([i, i], [0, 1], color="black", linewidth=0.25, alpha=0.35)
    ax.text(16, 2.18, "Formato de precisión simple, 32 bits",
            ha="center", va="center", fontsize=9)

    etiquetas = ["significando\nbajo", "significando\nalto", "exponente", "signo"]
    # Cotas exactas de |dx|/|x| para un valor normal (ver el docstring). El
    # exponente llega a 2^128 - 1, fuera del eje: se dibuja hasta el borde con
    # una flecha y la cota real se anota encima.
    XMAX = 1e8
    bajos  = [2.0 ** -24, 2.0 ** -11, 2.0 ** -1, 2.0]
    altos  = [2.0 ** -11, 2.0 ** -1, None, 2.0]
    colores = [CLARO, ACENTO, MEDIO, CALIDO]
    for i, (b, a, c) in enumerate(zip(bajos, altos, colores)):
        fin = XMAX / 2.2 if a is None else a
        if fin > b:
            bx.plot([b, fin], [i, i], color=c, linewidth=7, solid_capstyle="butt",
                    zorder=2)
        bx.plot([b], [i], marker="|", color=c, markersize=11, markeredgewidth=2.0,
                zorder=3)
        if a is None:
            bx.plot([fin * 1.35], [i], marker=">", color=c, markersize=9,
                    zorder=3)
        else:
            bx.plot([a], [i], marker="|", color=c, markersize=11,
                    markeredgewidth=2.0, zorder=3)
    bx.set_xscale("log")
    bx.set_yticks(range(4)); bx.set_yticklabels(etiquetas, fontsize=8)
    bx.set_ylim(-0.7, 3.7)
    bx.set_xlim(1e-8, XMAX)
    bx.set_xlabel("Perturbación relativa del valor alterado,  $|\\Delta x| / |x|$",
                  fontsize=8.5)
    bx.grid(axis="x", alpha=0.3, linewidth=0.5)
    bx.tick_params(labelsize=8)
    bx.text(12.0, 2.33,
            "continúa fuera del eje hasta $2^{128}\\approx 3.4\\times10^{38}$; "
            "si el exponente\nresultante vale 255, el valor pasa a $\\pm\\infty$ o NaN",
            fontsize=7.4, color=GRIS, va="bottom", linespacing=1.25)
    for lado in ("top", "right"):
        bx.spines[lado].set_visible(False)
    _guardar(fig, out, "met_inyeccion_zonas")


# ---------------------------------------------------------------------------
def fig_descomposicion(out: Path) -> None:
    """Reparto de los operandos sobre una rejilla de dos por dos procesos."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.set_xlim(0, 15.4); ax.set_ylim(-1.35, 4.9); ax.axis("off")

    def matriz(x0, y0, w, h, filas, cols, etiqueta, sub, colores):
        for i in range(filas):
            for j in range(cols):
                ax.add_patch(Rectangle((x0 + j * w / cols, y0 + (filas - 1 - i) * h / filas),
                                       w / cols, h / filas,
                                       facecolor=colores[(i * cols + j) % len(colores)],
                                       edgecolor="black", linewidth=0.7))
        ax.text(x0 + w / 2, y0 + h + 0.28, etiqueta, ha="center", fontsize=10)
        ax.text(x0 + w / 2, y0 - 0.30, sub, ha="center", va="top",
                fontsize=7.5, color=GRIS)

    matriz(0.2, 0.6, 3.2, 3.4, 2, 1, "$A$", "franjas de filas\n($K$ replicada)",
           [CLARO, MEDIO])
    ax.text(4.1, 2.3, "$\\times$", fontsize=13, ha="center", va="center")
    matriz(4.8, 0.6, 3.2, 3.4, 1, 2, "$B$", "franjas de columnas\n($K$ replicada)",
           [CLARO, MEDIO])
    ax.text(8.7, 2.3, "$=$", fontsize=13, ha="center", va="center")
    matriz(9.4, 0.6, 3.6, 3.4, 2, 2, "$C$", "un bloque por proceso",
           [CLARO, MEDIO, MEDIO, CLARO])

    for i in range(2):
        for j in range(2):
            ax.text(9.4 + (j + 0.5) * 1.8, 0.6 + (1.5 - i) * 1.7,
                    f"$p_{{{i}{j}}}$", ha="center", va="center", fontsize=8.5)
    ax.text(13.35, 3.55, "rejilla $P_r \\times P_c$", fontsize=8.5, color=GRIS)
    ax.text(13.35, 3.05, "$K$ nunca se\nparticiona, así que", fontsize=7.5,
            color=GRIS, va="top")
    ax.text(13.35, 1.95, "cada bloque de $C$\nes definitivo y no\nrequiere reducción",
            fontsize=7.5, color=GRIS, va="top")
    _guardar(fig, out, "met_descomposicion")


# ---------------------------------------------------------------------------
def fig_flujo_checksums(out: Path) -> None:
    """Parte reutilizable de la codificación frente a la parte recurrente."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 6.9); ax.axis("off")

    def caja(x, y, w, h, texto, color, estilo="-"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                               linewidth=0.9, linestyle=estilo))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=8)

    # Bloque izquierdo: lo que se calcula una sola vez.
    ax.add_patch(Rectangle((0.15, 3.05), 5.4, 2.6, facecolor="none",
                           edgecolor=GRIS, linewidth=0.8, linestyle=":"))
    ax.text(2.85, 5.95, "Una vez por pareja de operandos", fontsize=8.5,
            ha="center", color=GRIS)
    caja(0.5, 4.35, 2.1, 0.8, "$s_A = \\mathrm{colsum}(A)$", CLARO)
    caja(3.0, 4.35, 2.1, 0.8, "$e_f = s_A^{\\top} B_f$", CLARO)
    ax.add_patch(FancyArrowPatch((2.6, 4.75), (3.0, 4.75), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.9, color="black"))
    ax.text(2.85, 3.85, "dependen solo de $A$ y $B$, que la\nmultiplicación nunca escribe",
            ha="center", va="top", fontsize=7.5, color=GRIS)

    # Bloque derecho: lo que se recalcula en cada multiplicación.
    ax.add_patch(Rectangle((6.35, 0.35), 5.35, 5.3, facecolor="none",
                           edgecolor=GRIS, linewidth=0.8, linestyle=":"))
    ax.text(9.0, 5.95, "Por fragmento, tras cada multiplicación", fontsize=8.5,
            ha="center", color=GRIS)
    caja(6.75, 4.35, 4.55, 0.8, "$a_f = \\mathrm{colsum}(C_f)$", MEDIO)
    caja(6.75, 3.15, 4.55, 0.8, "$|a_f - e_f| > \\tau$  ?", MEDIO)
    caja(6.75, 1.85, 4.55, 0.8, "localización de la fila", CLARO, estilo="--")
    caja(6.75, 0.65, 4.55, 0.8, "corrección del elemento", CLARO, estilo="--")
    for y0, y1 in ((4.35, 3.95), (3.15, 2.65), (1.85, 1.45)):
        ax.add_patch(FancyArrowPatch((9.02, y0), (9.02, y1), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=0.9,
                                     color="black"))
    ax.text(9.02, 2.83, "solo si se detecta", fontsize=7, color=GRIS,
            ha="center", va="center", style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    # El vínculo entre ambos: la parte reutilizable alimenta la detección.
    ax.add_patch(FancyArrowPatch((5.1, 4.32), (6.75, 3.68), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=1.0, color=ACENTO,
                                 connectionstyle="arc3,rad=-0.2"))
    ax.text(6.12, 2.92, "$e_f$ ya calculado\nalimenta la comparación", fontsize=7.5,
            color=ACENTO, ha="right", va="top")
    _guardar(fig, out, "met_flujo_checksums")


# ---------------------------------------------------------------------------
def fig_pipeline_streams(out: Path) -> None:
    """Línea de tiempo de los dos flujos con las compuertas por evento."""
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    F = 4
    dur_gemm, dur_ver = 2.4, 1.5
    ax.set_xlim(-3.1, F * dur_gemm + dur_ver + 0.3)
    ax.set_ylim(-1.05, 2.6); ax.axis("off")

    ax.text(-0.3, 1.85, "flujo de cómputo", fontsize=8, va="center", ha="right")
    ax.text(-0.3, 0.75, "flujo de verificación", fontsize=8, va="center", ha="right")
    for y in (1.55, 0.45):
        ax.plot([-0.1, F * dur_gemm + dur_ver + 0.15], [y, y], color=GRIS,
                linewidth=0.6)

    for f in range(F):
        x = f * dur_gemm
        ax.add_patch(Rectangle((x, 1.55), dur_gemm - 0.12, 0.6, facecolor=MEDIO,
                               edgecolor="black", linewidth=0.7))
        ax.text(x + (dur_gemm - 0.12) / 2, 1.85, f"GEMM $f_{f}$", ha="center",
                va="center", fontsize=7.5)
        xv = x + dur_gemm
        ax.add_patch(Rectangle((xv, 0.45), dur_ver, 0.6, facecolor=ACENTO,
                               edgecolor="black", linewidth=0.7))
        ax.text(xv + dur_ver / 2, 0.75, f"verif. $f_{f}$", ha="center",
                va="center", fontsize=7.5, color="white")
        ax.add_patch(FancyArrowPatch((xv - 0.06, 1.55), (xv + 0.04, 1.05),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.8, color=CALIDO))
    ax.text(F * dur_gemm + dur_ver + 0.15, 2.35, "evento por fragmento",
            fontsize=7, color=CALIDO, ha="right")
    ax.annotate("", xy=(dur_gemm, 0.15), xytext=(2 * dur_gemm, 0.15),
                arrowprops=dict(arrowstyle="<->", linewidth=0.7, color=GRIS))
    ax.text(1.5 * dur_gemm, -0.15, "la verificación de un fragmento transcurre\n"
            "mientras se multiplica el siguiente", ha="center", va="top",
            fontsize=7, color=GRIS)
    _guardar(fig, out, "met_pipeline_streams")


# ---------------------------------------------------------------------------
def fig_calibracion(out: Path) -> None:
    """Cadena que va del residuo medido al umbral declarado."""
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.set_xlim(0, 16.75); ax.set_ylim(0, 3.0); ax.axis("off")
    pasos = [("ejecución\nsin fallo", CLARO),
             ("residuos\n$|a_j - e_j|$", CLARO),
             ("máximo por\nmultiplicación", MEDIO),
             ("ajuste de\nvalores extremos", MEDIO),
             ("presupuesto de\nfalsas alarmas", ACENTO),
             ("umbral $\\tau$", CALIDO)]
    w, h, sep = 2.5, 1.15, 0.35
    for i, (texto, color) in enumerate(pasos):
        x = i * (w + sep)
        ax.add_patch(Rectangle((x, 1.0), w, h, facecolor=color, edgecolor="black",
                               linewidth=0.8))
        ax.text(x + w / 2, 1.0 + h / 2, texto, ha="center", va="center",
                fontsize=6.8, color="white" if color in (ACENTO, CALIDO) else "black")
        if i < len(pasos) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 1.0 + h / 2),
                                         (x + w + sep, 1.0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=9,
                                         linewidth=0.9, color="black"))
    ax.text(8.37, 0.45, "el criterio sustituye la elección de una constante por una "
            "probabilidad de falsa alarma declarada",
            ha="center", fontsize=7.3, color=GRIS)
    _guardar(fig, out, "met_calibracion")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/presentation/libro/figuras")
    out = Path(ap.parse_args().out)
    print(f"Generando figuras de metodología en {out}/")
    fig_inyeccion_zonas(out)
    fig_descomposicion(out)
    fig_flujo_checksums(out)
    fig_pipeline_streams(out)
    # fig_calibracion: retirada del libro, su contenido se expresa en el texto
    # de la seccion de calibracion. La funcion se conserva por si se recupera.


if __name__ == "__main__":
    main()
