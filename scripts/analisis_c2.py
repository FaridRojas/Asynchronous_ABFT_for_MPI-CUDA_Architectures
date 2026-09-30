"""C2: descomposición del costo expuesto en el régimen grande a partir de trazas nvprof.

Cada ensayo ejecuta la pasada sin protección (F multiplicaciones solas) y luego la
protegida (F multiplicaciones junto al flujo de verificación) sobre los mismos
operandos.  Uso: python3 analisis_c2.py <dir_con_trazas> [F] [calentamientos]
"""
import csv
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "c2_traza"
F = int(sys.argv[2]) if len(sys.argv) > 2 else 8
WARM = int(sys.argv[3]) if len(sys.argv) > 3 else 2


def leer(p):
    lines = p.read_text(errors="replace").splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('"Start"'))
    rows = list(csv.DictReader(lines[i:]))
    units = rows[0]
    assert units["Start"] == "us" and units["Duration"] == "us", units
    out = []
    for r in rows[1:]:
        try:
            s, d = float(r["Start"]), float(r["Duration"])
        except (TypeError, ValueError):
            continue
        out.append(dict(s=s, e=s + d, d=d, name=r["Name"], stream=r["Stream"]))
    return sorted(out, key=lambda x: x["s"])


def analizar(p):
    ev = leer(p)
    gemm_name = "sgemm_128x128x8_NN_vec"
    gemms = [x for x in ev if x["name"].startswith(gemm_name)]
    n_tr = len(gemms) // (2 * F)
    assert len(gemms) == n_tr * 2 * F, (p.name, len(gemms))
    res = []
    for t in range(WARM, n_tr):
        base = gemms[2 * F * t: 2 * F * t + F]
        prot = gemms[2 * F * t + F: 2 * F * (t + 1)]
        nxt = gemms[2 * F * (t + 1)]["s"] if t + 1 < n_tr else float("inf")
        win = [x for x in ev if x["s"] >= base[-1]["e"] and x["s"] < nxt
               and not x["name"].startswith("[CUDA memcpy")]
        ver = [x for x in win if x["name"].startswith("k_")]
        t_base = base[-1]["e"] - base[0]["s"]
        w0, w1 = min(x["s"] for x in win), max(x["e"] for x in win)
        t_prot = w1 - w0
        head = prot[0]["s"] - w0
        tail = w1 - prot[-1]["e"]
        L = sum(x["d"] for x in prot) - sum(x["d"] for x in base)
        gp = (prot[-1]["e"] - prot[0]["s"]) - sum(x["d"] for x in prot)
        gb = t_base - sum(x["d"] for x in base)
        # verificación que corre mientras hay una multiplicación en curso
        def solape(v):
            return sum(max(0.0, min(v["e"], g["e"]) - max(v["s"], g["s"])) for g in prot)
        vtot = sum(v["d"] for v in ver)
        vov = sum(solape(v) for v in ver)
        res.append(dict(t_base=t_base, t_prot=t_prot, dT=t_prot - t_base, head=head,
                        tail=tail, L=L, dgap=gp - gb, gemm_base=st.mean(x["d"] for x in base),
                        gemm_prot=st.mean(x["d"] for x in prot), vtot=vtot, vov=vov))
    return res


print("F =", F, "| calentamientos descartados =", WARM)
print(f"{'traza':24s} {'T_base':>9s} {'dT':>8s} {'dT%':>6s} | {'alarg.':>8s} {'arranque':>9s} {'cola':>8s} {'huecos':>8s} | {'GEMM base':>10s} {'GEMM prot':>10s} {'alarg%':>7s} | {'verif':>7s} {'solapada':>8s}")
agg = {}
for p in sorted(D.glob("traza_*_rank*.csv"), key=lambda p: (int(p.stem.split("_")[1]), p.stem)):
    r = analizar(p)
    m = {k: st.mean(x[k] for x in r) for k in r[0]}
    S = int(p.stem.split("_")[1])
    agg.setdefault(S, []).append(m)
    print(f"{p.stem:24s} {m['t_base']/1000:8.2f}ms {m['dT']/1000:7.3f}ms {100*m['dT']/m['t_base']:5.2f}% | "
          f"{m['L']/1000:7.3f}ms {m['head']/1000:8.3f}ms {m['tail']/1000:7.3f}ms {m['dgap']/1000:7.3f}ms | "
          f"{m['gemm_base']/1000:9.3f}ms {m['gemm_prot']/1000:9.3f}ms {100*(m['gemm_prot']/m['gemm_base']-1):6.2f}% | "
          f"{m['vtot']/1000:6.2f}ms {100*m['vov']/m['vtot']:7.1f}%")

print("\nMedia de los dos procesos, fracción de la diferencia dT:")
for S, ms in sorted(agg.items()):
    m = {k: st.mean(x[k] for x in ms) for k in ms[0]}
    dT = m["dT"]
    print(f"  {S:5d}: dT={dT/1000:.3f} ms ({100*dT/m['t_base']:.2f} % de la pasada sin protección) = "
          f"alargamiento {100*m['L']/dT:.0f} % + arranque {100*m['head']/dT:.0f} % + cola {100*m['tail']/dT:.0f} % + huecos {100*m['dgap']/dT:.0f} %;"
          f" multiplicación {100*(m['gemm_prot']/m['gemm_base']-1):+.2f} % más larga con verificación concurrente")
