#!/usr/bin/env python3
"""
Regenera en español las figuras de datos del libro.

Los graficadores del repositorio (plot_metrics, plot_comparison y
plot_zone_recall) rotulan en inglés, que es el idioma del artículo, y sus
salidas bajo docs/ son las figuras del artículo. Este guion no modifica ni esos
graficadores ni esas salidas: intercepta el texto de cada figura para
traducirlo, dirige la salida a un directorio temporal y copia el resultado a la
carpeta de figuras del libro, en PDF vectorial y con el nombre que cita el
documento.

Las trazas de perfilado se redibujan a partir de la exportación SQLite de
Nsight Systems en lugar de copiarse como capturas de pantalla, para que queden
en el idioma y el estilo del resto del libro. Si el informe .nsys-rep no tiene
su .sqlite al lado, se exporta con nsys a un directorio temporal.

    python3 scripts/figuras_libro.py [--out DIRECTORIO]      (por defecto, figuras_libro/)
"""

import argparse
import importlib.util
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.figure as mfig    # noqa: E402
import matplotlib.pyplot as plt     # noqa: E402
import matplotlib.text as mtext     # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent

# ---------------------------------------------------------------------------
# Traducción del texto de las figuras
# ---------------------------------------------------------------------------

ZONAS = {
    "any": "sin restricción",
    "sign": "signo",
    "exponent": "exponente",
    "sig_high": "signif. alto",
    "sig_low": "signif. bajo",
}

REGIMEN = {
    "Square, small (M=N=K, ~50-1024)": "cuadrado pequeño (M = N = K)",
    "Square, big (M=N=K, 256-10240)": "cuadrado grande (M = N = K)",
    "Non-square, small (M=N, K=256)": "no cuadrado pequeño (M = N, K = 256)",
    "Non-square, big (M=N, K=1024, 256-10240)": "no cuadrado grande (M = N, K = 1024)",
}

EXACTO = {
    **ZONAS,
    # plot_metrics: barridos
    "Matrix size  (M=N=K)": "Tamaño de la matriz (M = N = K)",
    "Matrix size  (M=N)": "Tamaño de la matriz (M = N)",
    "baseline cuBLAS": "sin protección (cuBLAS)",
    "ABFT (no fault)": "protegida, sin fallo",
    "ABFT (fault)": "protegida, con fallo",
    "GFLOPS (mean)": "GFLOPS (media)",
    "Throughput vs size — baseline vs ABFT (±fault)":
        "Rendimiento aritmético frente al tamaño",
    "Runtime (ms, mean)": "Tiempo de ejecución (ms, media)",
    "Runtime vs size — baseline vs ABFT (±fault)":
        "Tiempo de ejecución frente al tamaño",
    "Runtime overhead vs our cuBLAS": "Sobrecarga en tiempo",
    "ABFT runtime overhead vs our cuBLAS baseline":
        "Sobrecarga respecto de la ejecución sin protección",
    "Throughput overhead vs our cuBLAS": "Pérdida de rendimiento",
    "ABFT throughput overhead vs our cuBLAS baseline":
        "Degradación del rendimiento aritmético",
    # plot_metrics: tiempos por configuración, zonas y detección
    "Protected runtime (ms)": "Tiempo de ejecución (ms)",
    "base": "sin protección",
    "none": "sin fallo",
    "add": "con fallo",
    "calib": "calibración",
    "Runtime overhead": "Sobrecarga en tiempo",
    "ABFT overhead by SWIFI injection zone": "Sobrecarga por región de inyección",
    "size": "tamaño",
    "Recall": "Sensibilidad",
    "Detection precision": "Precisión de detección",
    "Correction precision": "Precisión de corrección",
    "Rate (1.0 = 100%)": "Proporción (1.0 = 100 %)",
    "Fragments observed": "Fragmentos verificados",
    "TP": "verdadero positivo (TP)",
    "FN": "falso negativo (FN)",
    "FP": "falso positivo (FP)",
    "TN": "verdadero negativo (TN)",
    # plot_metrics: distribución del residuo
    "|actualRow − expectedRow|": "|a[j] − e[j]|",
    "count (log)": "frecuencia (escala logarítmica)",
    "Cumulative distribution (positive samples)":
        "Distribución acumulada (lecturas positivas)",
    "CDF (fraction of samples ≤ x)": "fracción de lecturas ≤ x",
    # plot_comparison
    "M = N  (K fixed)": "M = N (K fija)",
    "GFLOPS/s": "GFLOPS",
    "Runtime (ms)": "Tiempo de ejecución (ms)",
    "ours / baseline cuBLAS": "propuesta: sin protección (cuBLAS)",
    "ours / online ABFT (no fault)": "propuesta: ABFT sin fallo",
    "ours / online ABFT (fault)": "propuesta: ABFT con fallo",
    "theirs / cuBLAS": "referencia: cuBLAS",
    "theirs / fused ABFT (no fault)": "referencia: ABFT fusionada sin fallo",
    "theirs / fused ABFT (fault)": "referencia: ABFT fusionada con fallo",
    "ours / ABFT (no fault)": "propuesta: ABFT sin fallo",
    "ours / ABFT (fault)": "propuesta: ABFT con fallo",
    "theirs / ABFT (no fault)": "referencia: ABFT fusionada sin fallo",
    "theirs / ABFT (fault)": "referencia: ABFT fusionada con fallo",
    "Runtime overhead vs our cuBLAS (%)":
        "Sobrecarga respecto de la línea base propia (%)",
    "Runtime overhead vs their cuBLAS (%)":
        "Sobrecarga respecto del cuBLAS de la referencia (%)",
    # plot_zone_recall
    "sign (bit 31)": "signo (bit 31)",
    "exponent (23-30)": "exponente (bits 23–30)",
    "significand high (13-22)": "significando alto (bits 13–22)",
    "significand low (0-12)": "significando bajo (bits 0–12)",
    "any (0-31)": "sin restricción (bits 0–31)",
    "Matrix size (M = N = K)": "Tamaño de la matriz (M = N = K)",
    "Recall = TP / (TP + FN)": "Sensibilidad = TP / (TP + FN)",
    "SWIFI recall per IEEE-754 bit zone":
        "Sensibilidad por región del formato IEEE 754",
    "bit zone": "región",
}


def _reg(s):
    return REGIMEN.get(s, s)


def _sin_miles(s):
    return s.replace(",", "")


REGLAS = [
    (r"Matrix size  \(M=N, K=(\d+) fixed\)",
     lambda m: f"Tamaño de la matriz (M = N, K = {m[1]} fija)"),
    (r"Protected runtime per config \((.+), mean, min/max bars\)",
     lambda m: f"Tiempo por configuración ({m[1]}; media y extremos)"),
    (r"Detection / correction quality \(SWIFI, (.+)\)",
     lambda m: f"Detección y corrección ({m[1]})"),
    (r"Confusion matrix \(SWIFI, (.+)\)",
     lambda m: f"Matriz de confusión ({m[1]})"),
    (r"Calibration noise distribution\n\(([\d,]+) distinct values from "
     r"([\d,]+) readings, ([\d,]+) exact zeros\)",
     lambda m: ("Distribución del residuo en ejecuciones limpias\n"
                f"({_sin_miles(m[1])} valores distintos en {_sin_miles(m[2])} "
                f"lecturas, {_sin_miles(m[3])} ceros exactos)")),
    (r"observed max = (.+)", lambda m: f"máximo observado = {m[1]}"),
    # El graficador rotula el percentil 99.9 como p99: se corrige aquí.
    (r"p50  =  (\S+)\np90  =  (\S+)\np99  =  (\S+)\np99  =  (\S+)\np100  =  (\S+)",
     lambda m: (f"p50    = {m[1]}\np90    = {m[2]}\np99    = {m[3]}\n"
                f"p99.9  = {m[4]}\nmáximo = {m[5]}")),
    (r"GFLOPS — (.+)", lambda m: f"Rendimiento aritmético: {_reg(m[1])}"),
    (r"Runtime — (.+)", lambda m: f"Tiempo de ejecución: {_reg(m[1])}"),
    (r"ABFT overhead vs our cuBLAS — (.+)",
     lambda m: f"Sobrecarga respecto de la línea base propia\n{_reg(m[1])}"),
    (r"ABFT overhead vs their cuBLAS — (.+)",
     lambda m: f"Sobrecarga respecto del cuBLAS de la referencia\n{_reg(m[1])}"),
    (r"([−-]?\d+(?:\.\d+)?)%", lambda m: f"{m[1]} %"),
]
REGLAS = [(re.compile(p, re.S), f) for p, f in REGLAS]


def traducir(s):
    if s in EXACTO:
        s = EXACTO[s]
    else:
        for rx, f in REGLAS:
            m = rx.fullmatch(s)
            if m:
                s = f(m)
                break
    # 4096x1024x4096 -> 4096×1024×4096
    return re.sub(r"(?<=\d)x(?=\d)", "×", s)


_set_text_original = mtext.Text.set_text


def _set_text_es(self, s):
    if isinstance(s, str) and s:
        s = traducir(s)
    return _set_text_original(self, s)


mtext.Text.set_text = _set_text_es

# Cada PNG que escriba un graficador se escribe también como PDF vectorial.
_savefig_original = mfig.Figure.savefig


def _savefig_con_pdf(self, fname, *args, **kwargs):
    _savefig_original(self, fname, *args, **kwargs)
    s = str(fname)
    if s.lower().endswith(".png"):
        k = dict(kwargs)
        k.pop("dpi", None)
        _savefig_original(self, s[:-4] + ".pdf", *args, **k)


mfig.Figure.savefig = _savefig_con_pdf


def ejecutar(modulo, argv):
    """Carga un graficador del repositorio y ejecuta su main() con argv."""
    ruta = SCRIPTS / f"{modulo}.py"
    spec = importlib.util.spec_from_file_location(modulo, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    anterior = sys.argv
    sys.argv = [str(ruta)] + [str(a) for a in argv]
    try:
        mod.main()
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"  ! {modulo} terminó con código {e.code}")
    finally:
        sys.argv = anterior
        plt.close("all")


# ---------------------------------------------------------------------------
# Trazas de perfilado
# ---------------------------------------------------------------------------

COLOR = {
    "multiplicación": "#2a6f9e",
    "codificación de A": "#9a9a9a",
    "checksum esperado": "#c2703d",
    "checksum observado": "#5a9367",
    "detección": "#8c6d9e",
    "localización y corrección": "#d9c26a",
}
ETIQUETA = {
    "multiplicación": "multiplicación (cuBLAS)",
    "codificación de A": "sumas de columnas de $A$",
    "checksum esperado": r"$\mathit{checksum}$ esperado",
    "checksum observado": r"$\mathit{checksum}$ observado",
    "detección": "detección",
    "localización y corrección": "localización y corrección",
}


def _es_gemm(nombre):
    n = nombre.lower()
    return not n.startswith("k_") and "gemm" in n


def _categoria(nombre, previa):
    n = nombre.lower()
    # Todo lo que no es un kernel propio pertenece a la biblioteca, incluida
    # la reducción con que cuBLAS combina las particiones de K en las formas
    # pequeñas (splitKreduce_kernel).
    if not n.startswith("k_"):
        return "multiplicación"
    if n.startswith("k_col_checksum_a"):
        return "codificación de A"
    if n.startswith("k_expected_row"):
        return "checksum esperado"
    if n.startswith("k_reduce_enc"):
        # Combina las sumas parciales del kernel que lo precede en el flujo.
        return previa or "checksum esperado"
    if n.startswith("k_actual_row"):
        return "checksum observado"
    if n.startswith("k_detect"):
        return "detección"
    return "localización y corrección"


def _sqlite(rep: Path, tmp: Path) -> Path:
    junto = rep.with_suffix(".sqlite")
    if junto.exists():
        return junto
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("nsys no está disponible para exportar " + rep.name)
    destino = tmp / (rep.stem + ".sqlite")
    subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite",
                    "true", "--output", str(destino), str(rep)],
                   check=True, capture_output=True)
    return destino


def _leer_kernels(db: Path):
    con = sqlite3.connect(db)
    filas = con.execute(
        "SELECT k.start, k.end, k.streamId, s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id "
        "ORDER BY k.start").fetchall()
    con.close()
    return [{"t0": a / 1e6, "t1": b / 1e6, "s": st, "n": nom}
            for a, b, st, nom in filas]


def _protegidas(ks, F):
    """Operaciones protegidas de la captura.

    Cada ensayo ejecuta la multiplicación sin protección y a continuación la
    protegida, casi sin pausa entre ambas, así que no basta con buscar los
    intervalos inactivos. La operación protegida empieza con las sumas de
    columnas de A, que abren el flujo de verificación, y termina con la
    detección de su último fragmento; se conservan las que contienen
    exactamente F multiplicaciones."""
    ops = []
    for i, k in enumerate(ks):
        if not k["n"].startswith("k_col_checksum_A"):
            continue
        detecciones = [d for d in ks[i:] if d["n"].startswith("k_detect")][:F]
        if len(detecciones) < F:
            continue
        t_ini, t_fin = k["t0"], detecciones[-1]["t1"]
        op = [x for x in ks if t_ini <= x["t0"] <= t_fin]
        if sum(1 for x in op if _es_gemm(x["n"])) == F:
            ops.append(op)
    if not ops:
        raise RuntimeError("no se pudieron separar las operaciones protegidas")
    return ops


def _forma(log: Path):
    m = re.search(r"M x K x N\s*:\s*(\d+) x (\d+) x (\d+)", log.read_text())
    F = re.search(r"Frags per rank\s*:\s*(\d+)", log.read_text())
    M, K, N = (int(x) for x in m.groups())
    forma = f"{M}$^3$" if M == K == N else f"{M} × {K} × {N}"
    return forma, int(F.group(1))


def dibujar_traza(prof: Path, regimen: str, destino: Path, tmp: Path,
                  ancho: float, alto: float, columnas: int):
    rep = prof / f"{regimen}_ours_online.nsys-rep"
    forma, F = _forma(prof / f"{regimen}_ours_online_nsys_run.log")
    ks = _leer_kernels(_sqlite(rep, tmp))
    prot = _protegidas(ks, F)
    # Operación representativa: la de duración mediana.
    dur = sorted(prot, key=lambda g: max(k["t1"] for k in g) - min(k["t0"] for k in g))
    op = dur[len(dur) // 2]

    computo = max({k["s"] for k in op},
                  key=lambda s: sum(k["t1"] - k["t0"] for k in op
                                    if k["s"] == s and _categoria(k["n"], None) == "multiplicación"))
    t0 = min(k["t0"] for k in op)
    t1 = max(k["t1"] for k in op)
    lapso = t1 - t0

    with plt.rc_context({"font.size": 8, "font.family": "serif",
                         "mathtext.fontset": "dejavuserif"}):
        fig, ax = plt.subplots(figsize=(ancho, alto))
        previa = {}
        presentes = []
        for k in op:
            cat = _categoria(k["n"], previa.get(k["s"]))
            if cat != "multiplicación":
                previa[k["s"]] = cat
            if cat not in presentes:
                presentes.append(cat)
            y = 1 if k["s"] == computo else 0
            # Los kernels muy breves se dibujan con un ancho mínimo y sin
            # borde, para que se vea su color y no solo el contorno.
            d = k["t1"] - k["t0"]
            estrecho = d < lapso * 0.006
            ax.barh(y, max(d, lapso * 0.004), left=k["t0"] - t0,
                    height=0.62, color=COLOR[cat],
                    edgecolor="none" if estrecho else "black",
                    linewidth=0 if estrecho else 0.25, zorder=3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["verificación", "cómputo"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(-lapso * 0.01, lapso * 1.01)
        ax.set_xlabel("Tiempo desde el inicio de la operación (ms)")
        ax.set_title(f"Operación protegida de {forma} sobre Ampere",
                     fontsize=8.5, pad=4)
        ax.grid(axis="x", alpha=0.25, linewidth=0.4, zorder=0)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(axis="y", length=0)
        orden = [c for c in COLOR if c in presentes]
        # Se reserva bajo el eje el espacio de la leyenda para que no se
        # monte sobre el rótulo del tiempo.
        filas = -(-len(orden) // columnas)
        fig.tight_layout(rect=(0, (0.08 + 0.17 * filas) / alto, 1, 1))
        fig.legend(handles=[Patch(facecolor=COLOR[c], edgecolor="black",
                                  linewidth=0.4, label=ETIQUETA[c]) for c in orden],
                   loc="lower center", ncol=columnas, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.savefig(destino, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    print(f"  {destino.name}: {len(prot)} operaciones protegidas, "
          f"se dibuja la de duración mediana ({lapso:.3f} ms)")


# ---------------------------------------------------------------------------
# Bondad del ajuste de valores extremos
# ---------------------------------------------------------------------------

def figura_qq(calib: Path, destino: Path):
    """Gráficos cuantil-cuantil de los máximos por pasada frente a la
    distribución de Gumbel ajustada por el método de los momentos, con las
    mismas funciones que emplea analyze_calibration.py para fijar el umbral."""
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "analyze_calibration", SCRIPTS / "analyze_calibration.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    diffs = ac.load_diffs(str(calib))
    pasadas = ac.load_repeats(str(calib / "abft_calibration.csv"))
    with plt.rc_context({"font.size": 8}):
        fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
        for ax, n in zip(axs.flat, sorted(diffs)):
            maximos = np.sort(ac.block_maxima(diffs[n], pasadas.get(n, 20)))
            mu, beta = ac.gumbel_fit(maximos)
            p = (np.arange(1, len(maximos) + 1) - 0.5) / len(maximos)
            teoricos = mu - beta * np.log(-np.log(p))
            r = np.corrcoef(teoricos, maximos)[0, 1]
            lo = min(teoricos.min(), maximos.min())
            hi = max(teoricos.max(), maximos.max())
            ax.plot([lo, hi], [lo, hi], "-", color="#c2703d", lw=1)
            ax.plot(teoricos, maximos, "o", ms=3.5, color="#2a6f9e")
            ax.set_title(f"{n}$^3$  (r = {r:.3f})", fontsize=8.5)
            ax.ticklabel_format(style="sci", scilimits=(0, 0))
            ax.grid(alpha=0.3, linewidth=0.4)
        fig.supxlabel("cuantil de la distribución de Gumbel ajustada")
        fig.supylabel("máximo observado por pasada")
        fig.tight_layout()
        fig.savefig(destino, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    print(f"  {destino.name}")


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figuras_libro"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = REPO / "docs"
    tmp = Path(tempfile.mkdtemp(prefix="figuras_libro_"))
    faltan = []

    def copiar(origen: Path, nombre: str):
        if not origen.exists():
            faltan.append(nombre)
            print(f"  FALTA {nombre} ({origen.name})")
            return
        shutil.copyfile(origen, out / f"{nombre}.pdf")
        viejo = out / f"{nombre}.png"
        if viejo.exists():
            viejo.unlink()
        print(f"  {nombre}.pdf")

    try:
        print("== Barridos ==")
        for sufijo in ("", "_ns"):
            d = docs / f"campaign{sufijo}"
            candidatos = [d / "abft_metrics.csv", d / "abft_calibration.csv",
                          docs / f"abft_metrics{sufijo}.csv"]
            metricas = next((c for c in candidatos if c.exists()), None)
            if metricas is None:
                print(f"  FALTA el CSV de métricas de campaign{sufijo}")
                continue
            t = tmp / f"campaign{sufijo}"
            ejecutar("plot_metrics", ["--metrics", metricas,
                                      "--diffs", tmp / "sin_diffs.csv", "--out", t])
            forma = "sq" if sufijo == "" else "ns"
            copiar(t / "gflops_vs_size.pdf", f"campana_gflops_{forma}")
            copiar(t / "runtime_overhead_vs_size.pdf", f"campana_overhead_{forma}")
            copiar(t / "time_vs_size.pdf", f"anx_time_{forma}")
            copiar(t / "gflops_overhead_vs_size.pdf", f"anx_gflops_ovh_{forma}")
            tamanos = ((256, 1024, 2048, 4096, 8192, 10240) if forma == "sq"
                       else (1024, 4096, 10240))
            for n in tamanos:
                copiar(t / f"protected_timing_{n}.pdf", f"anx_timing_{forma}_{n}")
            if forma == "sq":
                for n in (1024, 10240):
                    copiar(t / f"detection_metrics_{n}.pdf", f"deteccion_{n}")
                    copiar(t / f"confusion_matrix_{n}.pdf", f"confusion_{n}")
                for n in (512, 2048, 4096, 8192):
                    copiar(t / f"detection_metrics_{n}.pdf", f"anx_deteccion_{n}")
                    copiar(t / f"confusion_matrix_{n}.pdf", f"anx_confusion_{n}")
                copiar(t / "overhead_by_zone.pdf", "anx_ovh_zona")

        print("== Comparación ==")
        for plataforma, carpeta in (("maxwell", "comparison"), ("ampere", "comparison_pacca")):
            t = tmp / carpeta
            ejecutar("plot_comparison", ["--csv", docs / carpeta / "compare_all.csv",
                                         "--outdir", t])
            for r in ("big_sq", "big_ns", "small_sq", "small_ns"):
                copiar(t / f"gflops_{r}.pdf", f"cmp_{plataforma}_{r}")
                copiar(t / f"overhead_vs_ours_{r}.pdf", f"anx_ovh_{plataforma}_{r}")

        print("== Calibración ==")
        calib = docs / "calibration"
        for n in (512, 1024, 2048, 4096, 8192, 10240):
            t = tmp / f"calib_{n}"
            ejecutar("plot_metrics", ["--metrics", calib / "abft_calibration.csv",
                                      "--diffs", calib / f"abft_calib_diffs_{n}.csv",
                                      "--out", t])
            nombre = f"calib_dist_{n}" if n in (1024, 8192) else f"anx_calib_{n}"
            copiar(t / "calibration_distribution.pdf", nombre)
        ejecutar("plot_zone_recall", ["--csv", calib / "abft_zone_recall.csv",
                                      "--out", tmp / "zone_recall.png"])
        copiar(tmp / "zone_recall.pdf", "anx_zone_recall")
        figura_qq(calib, out / "anx_qq_gumbel.pdf")

        print("== Trazas ==")
        prof = docs / "profile_pacca"
        for regimen, nombre, ancho, alto, columnas in (
                ("big_sq", "traza_big_sq", 7.0, 2.3, 5),
                ("big_ns", "anx_traza_big_ns", 4.4, 2.6, 3),
                ("small_sq", "anx_traza_small_sq", 4.4, 2.6, 3),
                ("small_ns", "anx_traza_small_ns", 4.4, 2.6, 3)):
            try:
                dibujar_traza(prof, regimen, out / f"{nombre}.pdf", tmp,
                              ancho, alto, columnas)
                viejo = out / f"{nombre}.png"
                if viejo.exists():
                    viejo.unlink()
            except Exception as e:           # noqa: BLE001
                faltan.append(nombre)
                print(f"  FALLO {nombre}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if faltan:
        print("No se generaron: " + ", ".join(faltan))
        sys.exit(1)
    print(f"Listo: {out}")


if __name__ == "__main__":
    main()
