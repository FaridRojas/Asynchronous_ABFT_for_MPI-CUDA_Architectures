"""Cifras del libro para C1 (costo de un despliegue) y C13 (kernel de cuBLAS y
salto del piso de ruido), a partir de docs/c1_localize y docs/c13_kernels.

Uso, desde la raíz del repositorio:  python3 scripts/analisis_c1_c13.py
"""
import csv
import hashlib
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

# ---------------------------------------------------------------- C1
# Filas: baseline_only=1 (línea base independiente), scheme=online (localización
# solo con inyección), scheme=online_loc (--localize-always), inject=add (fallo).
by = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(DOCS / "c1_localize" / "c1_metrics.csv")):
    k = "base" if r["baseline_only"] == "1" else ("fallo" if r["inject"] == "add" else r["scheme"])
    by[int(r["M"])][k].append(r)


def pareada(r):
    """Sobrecarga frente a la línea base medida en el mismo proceso, ensayo a ensayo."""
    return 100.0 * (float(r["protected_mean_ms"]) / float(r["baseline_mean_ms"]) - 1.0)


def expuesto_us(r):
    return 1000.0 * (float(r["protected_mean_ms"]) - float(r["baseline_mean_ms"]))


print("C1: sobrecarga pareada (%), media de las dos ejecuciones de cada variante")
print(f"{'M':>6s} {'solo inyección':>15s} {'siempre':>9s} {'diferencia':>11s} {'dif. (us)':>10s}")
filas = []
for S in sorted(by):
    d = by[S]
    a = st.mean(pareada(r) for r in d["online"])
    b = st.mean(pareada(r) for r in d["online_loc"])
    us = st.mean(expuesto_us(r) for r in d["online_loc"]) - st.mean(expuesto_us(r) for r in d["online"])
    filas.append((S, a, b, b - a, us))
    print(f"{S:6d} {a:14.2f}% {b:8.2f}% {b - a:+10.2f} {us:+10.1f}")

g = [f for f in filas if f[0] >= 5120]
dif = [f[3] for f in g]
media, de = st.mean(dif), st.stdev(dif)
t975 = 2.086                                   # t de Student, 20 grados de libertad
mitad = t975 * de / math.sqrt(len(dif))
print(f"\nDesde 5120 ({len(g)} tamaños): {st.mean(f[1] for f in g):.2f} % frente a "
      f"{st.mean(f[2] for f in g):.2f} %; diferencia {media:+.3f} puntos, "
      f"IC 95 % [{media - mitad:+.2f}, {media + mitad:+.2f}]")
m = [f for f in filas if 1024 <= f[0] <= 2560]
print(f"De 1024 a 2560: {min(f[3] for f in m):.1f} a {max(f[3] for f in m):.1f} puntos, "
      f"{min(f[4] for f in m):.0f} a {max(f[4] for f in m):.0f} us por operación")
cru = [100.0 * (float(r["protected_mean_ms"]) / float(by[S]["base"][0]["baseline_mean_ms"]) - 1.0)
       for S in by if S >= 5120 for r in by[S]["online"]]
print(f"Detección desde 5120: {st.mean(f[1] for f in g):.2f} % pareada, "
      f"{st.mean(cru):.2f} % frente a la línea base independiente")

# ---------------------------------------------------------------- C13
print("\nC13: kernel de multiplicación de cada tamaño (nvprof, proceso 0)")
for p in sorted((DOCS / "c13_kernels").glob("nvprof_*_rank0.txt"),
                key=lambda p: int(p.stem.split("_")[1])):
    # Resumen de nvprof: «... Calls ... Name».  Las 3 llamadas de 32x32x32 que
    # aparecen desde 2816 son el calentamiento de 64x64 del programa, no los fragmentos.
    llamadas = re.findall(r"\s(\d+)\s+\S+\s+\S+\s+\S+\s+(sgemm_\w+)", p.read_text(errors="replace"))
    print(f"{int(p.stem.split('_')[1]):6d}: " + ", ".join(f"{n} x{c}" for c, n in llamadas))

print("\nC13: la calibración repetida reproduce la publicada (SHA-256)")
esperado = dict(l.split()[::-1] for l in (DOCS / "c13_kernels" / "reproduccion_sha256.txt")
                .read_text().splitlines() if l.strip() and not l.startswith("#"))
for nombre, h in sorted(esperado.items(), key=lambda x: int(re.findall(r"\d+", x[0])[0])):
    real = hashlib.sha256((DOCS / "calibration" / nombre).read_bytes()).hexdigest()
    print(f"  {nombre:30s} {'idéntico' if real == h else 'DISTINTO'}")

print("\nC13: calibración cada 256 entre 2048 y 4096")
prev = None
for p in sorted((DOCS / "c13_kernels" / "calibracion_fina").glob("abft_calib_diffs_*.csv"),
                key=lambda p: int(p.stem.split("_")[-1])):
    v = [float(x) for x in p.read_text().split()[1:]]
    mx, md = max(v), st.median(v)
    paso = "" if prev is None else f"  máximo x{mx / prev[0]:.2f}, mediana x{md / prev[1]:.2f}"
    print(f"{int(p.stem.split('_')[-1]):6d}: mediana {md:.3e}  máximo {mx:.3e}{paso}")
    prev = (mx, md)
