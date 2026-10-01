"""Frecuencia de reloj y desfase entre líneas base (docs/reloj, trabajo de
run_reloj_felix.sbatch).  Para cada fase compara la frecuencia, el consumo y
la temperatura medios con el tiempo de la línea base independiente y de la
pareada.  Solo cuentan las muestras con la GPU en cómputo (más de 100 W), para
excluir la generación de operandos en el host.

Uso, desde la raíz del repositorio:  python3 scripts/analisis_reloj.py
"""
import csv
import statistics as st
from datetime import datetime, timedelta, timezone
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "docs" / "reloj"
TZ = timezone(timedelta(hours=-5))          # hora local del clúster SC3

muestras = []
for r in csv.reader(open(D / "reloj.csv")):
    if r[0].startswith("timestamp"):
        continue
    t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=TZ).timestamp()
    muestras.append(dict(t=t, sm=float(r[2].split()[0]), w=float(r[4].split()[0]),
                         temp=float(r[5]), motivo=int(r[6], 16)))
fases = list(csv.DictReader(open(D / "fases.csv")))
metricas = list(csv.DictReader(open(D / "reloj_metrics.csv")))

print(f"{'fase':5s} {'forma':18s} {'MHz':>7s} {'W':>6s} {'°C':>5s} {'límite pot.':>11s}  línea base (ms)")
previa = {}
for f, m in zip(fases, metricas):
    a, b = float(f["inicio"]), float(f["fin"])
    s = [x for x in muestras if a <= x["t"] <= b and x["w"] > 100]
    forma = f"{f['M']}x{f['K']}x{f['N']}"
    mhz = st.mean(x["sm"] for x in s)
    tope = 100 * sum(1 for x in s if x["motivo"] & 0x4) / len(s)
    base = float(m["baseline_mean_ms"])
    extra = ""
    if f["fase"] == "prot" and forma in previa:
        mhz0, base0 = previa[forma]
        extra = f"  pareada {100 * (base / base0 - 1):+.2f} % frente a la independiente; frecuencia {100 * (mhz / mhz0 - 1):+.2f} %"
    print(f"{f['fase']:5s} {forma:18s} {mhz:7.1f} {st.mean(x['w'] for x in s):6.1f} "
          f"{st.mean(x['temp'] for x in s):5.1f} {tope:10.1f}%  {base:.3f}{extra}")
    if f["fase"] == "base":
        previa[forma] = (mhz, base)
