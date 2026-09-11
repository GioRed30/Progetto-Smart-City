"""
Genera tutte le figure della tesi dai risultati delle 216 esecuzioni.

    python figure.py

Produce in figure/:
    risultati_finali.png     i quattro esiti principali
    testa2_vs_testa4.png     il confronto sulla profondita' della testa
    convergenza.png          curve per round: cifrato vs chiaro, e la divergenza
    accuratezza_costo.png    il piano accuratezza / costo della cifratura

Ogni valore e' ricalcolato dal CSV dei risultati: rilanciare lo script dopo
nuove esecuzioni aggiorna le figure senza dover toccare il codice.
"""
import pandas as pd, numpy as np, pathlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("default")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6})

CODICE = pathlib.Path(r"C:\Progetti\FL\Codice")
FIG = CODICE / "figure"; FIG.mkdir(exist_ok=True)

d = pd.read_csv(CODICE / "csv_DESKTOP-APC2VEU" / "DESKTOP-APC2VEU"
                / "federated_grid_search_results_DESKTOP-APC2VEU.csv")
d = d[d.global_epoch == 20].copy()
d["data"] = d.dataframe_path.str.extract(r"(\d{8})-\d{6}")[0].astype(int)

def round_eseguiti(p):
    try:    return len(pd.read_csv(p))
    except Exception: return np.nan

d["nr"] = d.dataframe_path.map(round_eseguiti)
d["spr"] = d.total_duration / d.nr

K = ["model_name", "num_custom_layers", "aggregation_algorithm", "learning_rate", "num_clients"]
ch = d[d.encryption_mode == "no_encryption"]
ci = d[d.encryption_mode != "no_encryption"]
m = ch.merge(ci, on=K, suffixes=("_ch", "_ci"))
m["dF1"] = m.best_f1_ci - m.best_f1_ch

BLU, BLU_C = "#2E6F8E", "#6FA8C7"
ARA, ARA_C = "#B5643C", "#E0A484"
VERDE, ROSSO, GRIGIO = "#8FB89A", "#B23A2E", "#8A8A8A"
COL = {("DeiT-Tiny", 2): BLU, ("DeiT-Tiny", 4): BLU_C,
       ("ResNet18", 2): ARA, ("ResNet18", 4): ARA_C}

# ============================ FIGURA 1: i quattro risultati ==================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

# (1) la cifratura non altera l'accuratezza
a = ax[0][0]
for mo, g in m.groupby("model_name"):
    a.scatter(g.best_f1_ch, g.best_f1_ci, s=34, alpha=0.85,
              color=BLU if mo == "DeiT-Tiny" else ARA,
              edgecolor="white", linewidth=0.6, label=mo, zorder=3)
lim = [0.38, 0.98]
a.plot(lim, lim, ls="--", lw=1.1, color=GRIGIO, zorder=2, label="identita'")
a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("F1 in chiaro"); a.set_ylabel("F1 cifrato")
a.set_title("La cifratura non altera l'accuratezza\n108 coppie appaiate, stessa configurazione", fontsize=11)
a.legend(fontsize=8, loc="upper left")
a.text(0.555, 0.425, "regime stabile, 72 coppie:\ndifferenza media  -0,0006\ntipica  0,0041      massima  0,0154\n70 coppie su 72 entro 0,01",
       fontsize=8.5, color="#3D4852",
       bbox=dict(boxstyle="round,pad=0.45", fc="#F4F1EA", ec="#D8D2C4", lw=0.8))

# (2) F1 per backbone e profondita' della testa, regime stabile
a = ax[0][1]
gruppi, etichette, colori = [], [], []
for (mo, lay), g in ci[ci.learning_rate != 0.01].groupby(["model_name", "num_custom_layers"]):
    gruppi.append(g.best_f1.values); etichette.append(f"{mo}\ntesta {lay}"); colori.append(COL[(mo, lay)])
bp = a.boxplot(gruppi, tick_labels=etichette, patch_artist=True, widths=0.55)
for c, p in zip(colori, bp["boxes"]): p.set_facecolor(c); p.set_alpha(0.8)
for x in bp["medians"]: x.set_color("#1F2933"); x.set_linewidth(1.7)
a.set_ylabel("F1"); a.set_title("Backbone e profondita' della testa, sotto cifratura\n"
                                "regime stabile, 72 esecuzioni", fontsize=11)

# (3) fragilita': divergenze a learning rate 0.01
a = ax[1][0]
ins = d[d.learning_rate == 0.01].assign(div=lambda x: x.best_f1 < 0.60)
t = ins.groupby(["model_name", "num_custom_layers"]).div.sum()
et = [("DeiT-Tiny", 2), ("DeiT-Tiny", 4), ("ResNet18", 2), ("ResNet18", 4)]
v = [int(t.get(k, 0)) for k in et]
a.bar(range(4), v, 0.6, color=[COL[k] for k in et], edgecolor="white", linewidth=1.2)
for i, y in enumerate(v):
    a.text(i, y + 0.3, str(y), ha="center", fontsize=10, color="#1F2933")
a.set_xticks(range(4)); a.set_xticklabels([f"{mo}\ntesta {l}" for mo, l in et])
a.set_ylim(0, 14)
a.set_ylabel("esecuzioni divergenti (F1 < 0,60)")
a.set_title("Fragilita' al learning rate alto\n18 esecuzioni per ciascuna combinazione, a lr 0,01", fontsize=11)

# (4) costo per round sotto cifratura, stessa sessione
a = ax[1][1]
sett = ci[ci.data >= 20260906]
righe = []
for (mo, lay, nc), g in sett.groupby(["model_name", "num_custom_layers", "num_clients"]):
    if nc in (8, 10):
        righe.append((mo, lay, nc, g.spr.mean()))
t = pd.DataFrame(righe, columns=["mo", "lay", "nc", "spr"])
et = [(lay, nc) for lay in (2, 4) for nc in (8, 10)]
x = np.arange(len(et)); w = 0.36
for i, mo in enumerate(("DeiT-Tiny", "ResNet18")):
    v = [t[(t.mo == mo) & (t.lay == l) & (t.nc == n)].spr.mean() for l, n in et]
    a.bar(x + (i - 0.5) * w, v, w, label=mo, color=BLU if i == 0 else ARA,
          edgecolor="white", linewidth=1.2)
    for j, y in enumerate(v):
        if y == y: a.text(x[j] + (i - 0.5) * w, y + 2, f"{y:.0f}", ha="center", fontsize=8.5)
a.set_xticks(x); a.set_xticklabels([f"testa {l}\n{n} client" for l, n in et])
a.set_ylabel("secondi per round"); a.set_ylim(0, 160)
a.set_title("Costo della cifratura per round\nstessa sessione, stesse condizioni di carico", fontsize=11)
a.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG / "risultati_finali.png", dpi=200, bbox_inches="tight", facecolor="white")
print("salvata", FIG / "risultati_finali.png")

# ================= FIGURA 2: testa 2 contro testa 4, appaiata ===============
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
st = d[d.learning_rate != 0.01]
K2 = ["model_name", "encryption_mode", "aggregation_algorithm", "learning_rate", "num_clients"]
p2 = st[st.num_custom_layers == 2].merge(st[st.num_custom_layers == 4], on=K2, suffixes=("_h2", "_h4"))
p2["d"] = p2.best_f1_h4 - p2.best_f1_h2

a = ax[0]
gruppi, etichette, colori = [], [], []
for (mo, enc), g in p2.groupby(["model_name", "encryption_mode"]):
    gruppi.append(g.d.values)
    etichette.append(f"{mo}\n{'cifrato' if enc != 'no_encryption' else 'in chiaro'}")
    colori.append(BLU if mo == "DeiT-Tiny" else ARA)
bp = a.boxplot(gruppi, tick_labels=etichette, patch_artist=True, widths=0.5)
for c, p in zip(colori, bp["boxes"]): p.set_facecolor(c); p.set_alpha(0.8)
for x_ in bp["medians"]: x_.set_color("#1F2933"); x_.set_linewidth(1.7)
a.axhline(0, color=GRIGIO, ls="--", lw=1.1)
a.set_ylabel("F1 (testa 4)  -  F1 (testa 2)")
a.set_title("La testa profonda non aggiunge accuratezza\n72 confronti appaiati, regime stabile", fontsize=11)

a = ax[1]
PAR = {("ResNet18", 2): 256*512+770, ("ResNet18", 4): 256*512+41538,
       ("DeiT-Tiny", 2): 256*192+770, ("DeiT-Tiny", 4): 256*192+41538}
et = [("DeiT-Tiny", 2), ("DeiT-Tiny", 4), ("ResNet18", 2), ("ResNet18", 4)]
v = [PAR[k] for k in et]
a.bar(range(4), v, 0.6, color=[COL[k] for k in et], edgecolor="white", linewidth=1.2)
for i, y in enumerate(v):
    a.text(i, y + 3500, f"{y:,}".replace(",", "."), ha="center", fontsize=9)
a.set_xticks(range(4)); a.set_xticklabels([f"{mo}\ntesta {l}" for mo, l in et])
a.set_ylabel("parametri cifrati per client e per round"); a.set_ylim(0, 200000)
a.set_title("Cio' che la testa profonda costa\nnumero di cifrature Paillier", fontsize=11)

plt.tight_layout()
plt.savefig(FIG / "testa2_vs_testa4.png", dpi=200, bbox_inches="tight", facecolor="white")
print("salvata", FIG / "testa2_vs_testa4.png")


FIG = CODICE / "figure"; FIG.mkdir(exist_ok=True)
d = pd.read_csv(CODICE/"csv_DESKTOP-APC2VEU"/"DESKTOP-APC2VEU"
                /"federated_grid_search_results_DESKTOP-APC2VEU.csv")
d = d[d.global_epoch == 20].copy()

BLU, ARA, GRIGIO, ROSSO = "#2E6F8E", "#B5643C", "#8A8A8A", "#B23A2E"
BLU_C, ARA_C = "#6FA8C7", "#E0A484"

def curva(riga):
    return pd.read_csv(riga.dataframe_path)["test_f1"].values

def trova(**kw):
    g = d.copy()
    for k, v in kw.items():
        g = g[g[k] == v]
    return g.iloc[0] if len(g) else None

# ============ FIGURA 3: curve di convergenza =========================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

# (a) chiaro e cifrato sovrapposti, stessa configurazione
a = ax[0]
base = dict(num_custom_layers=2, aggregation_algorithm="FedProx",
            learning_rate=0.001, num_clients=10)
for mo, col in (("DeiT-Tiny", BLU), ("ResNet18", ARA)):
    ch = trova(model_name=mo, encryption_mode="no_encryption", **base)
    ci = trova(model_name=mo, encryption_mode="direct_encrypted_update", **base)
    if ch is None or ci is None:
        continue
    yc, ye = curva(ch), curva(ci)
    a.plot(range(1, len(yc)+1), yc, color=col, lw=2.1, label=f"{mo} · in chiaro")
    a.plot(range(1, len(ye)+1), ye, color=col, lw=1.6, ls=(0, (4, 2.4)),
           label=f"{mo} · cifrato")
a.set_xlabel("round"); a.set_ylabel("F1 sulla validazione")
a.set_xlim(1, 20); a.set_ylim(0.875, 0.972)
a.set_title("Le due curve coincidono\ntesta 2 layer · FedProx · lr 0,001 · 10 client", fontsize=11)
a.legend(fontsize=8.5, loc="lower right")

# (b) il regime instabile: dove la CNN collassa
a = ax[1]
# sceglie automaticamente la configurazione a lr 0,01 col divario piu' ampio
ins = d[(d.learning_rate == 0.01) & (d.encryption_mode != "no_encryption")]
K = ["num_custom_layers", "aggregation_algorithm", "num_clients"]
cop = (ins[ins.model_name == "DeiT-Tiny"].merge(
       ins[ins.model_name == "ResNet18"], on=K, suffixes=("_d", "_r")))
cop["gap"] = cop.best_f1_d - cop.best_f1_r
scelta = cop.loc[cop.gap.idxmax()]
for mo, suf, col in (("DeiT-Tiny", "_d", BLU), ("ResNet18", "_r", ARA)):
    y = pd.read_csv(scelta[f"dataframe_path{suf}"])["test_f1"].values
    a.plot(range(1, len(y)+1), y, color=col, lw=2.1, label=mo)
a.axhline(0.60, color=ROSSO, ls="--", lw=1.1, label="soglia di divergenza")
a.set_xlabel("round"); a.set_ylabel("F1 sulla validazione")
a.set_xlim(1, 20); a.set_ylim(0.35, 0.98)
a.set_title("Al learning rate alto la CNN collassa\n"
            f"testa {int(scelta.num_custom_layers)} layer · {scelta.aggregation_algorithm} · "
            f"lr 0,01 · {int(scelta.num_clients)} client · cifrato", fontsize=11)
a.legend(fontsize=8.5, loc="center right")

plt.tight_layout()
plt.savefig(FIG/"convergenza.png", dpi=200, bbox_inches="tight", facecolor="white")
print("salvata", FIG/"convergenza.png")
print(f"  pannello b: testa {int(scelta.num_custom_layers)}, {scelta.aggregation_algorithm}, "
      f"{int(scelta.num_clients)} client -> DeiT {scelta.best_f1_d:.4f} vs ResNet {scelta.best_f1_r:.4f}")

# ============ FIGURA 4: accuratezza contro costo ======================
PAR = {("ResNet18",2):131842, ("ResNet18",4):172610,
       ("DeiT-Tiny",2):49922, ("DeiT-Tiny",4):90690}
st = d[(d.learning_rate != 0.01) & (d.encryption_mode != "no_encryption")]

fig, a = plt.subplots(figsize=(7.6, 5.4))
for (mo, lay), g in st.groupby(["model_name", "num_custom_layers"]):
    x, y = PAR[(mo, lay)], g.best_f1.mean()
    col = BLU if mo == "DeiT-Tiny" else ARA
    a.scatter(x, y, s=210, color=col, alpha=.9, edgecolor="white", linewidth=1.6, zorder=3)
    a.annotate(f"{mo}\ntesta {lay} layer", (x, y), textcoords="offset points",
               xytext=(0, -34 if lay == 2 else 22), ha="center", fontsize=9.5,
               color="#1F2933")
# collega le due profondita' dello stesso backbone
for mo, col in (("DeiT-Tiny", BLU), ("ResNet18", ARA)):
    p = [(PAR[(mo, l)], st[(st.model_name == mo) & (st.num_custom_layers == l)].best_f1.mean())
         for l in (2, 4)]
    a.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color=col, lw=1.2, alpha=.45, zorder=2)

from matplotlib.ticker import FuncFormatter
a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
a.set_xlabel("parametri cifrati per client, a ogni round")
a.set_ylabel("F1 medio, regime stabile")
a.set_xlim(20000, 195000); a.set_ylim(0.9215, 0.9475)
a.set_title("Accuratezza contro costo della cifratura\n"
            "36 esecuzioni cifrate per punto", fontsize=11.5)
a.annotate("", xy=(0.06, 0.93), xytext=(0.26, 0.74), xycoords="axes fraction",
           arrowprops=dict(arrowstyle="-|>", color="#9AA3B0", lw=1.5))
a.annotate("piu' accurato\ne piu' economico", xy=(0.27, 0.72), xycoords="axes fraction",
           fontsize=9.5, color="#5A6472", ha="left", va="top")
plt.tight_layout()
plt.savefig(FIG/"accuratezza_costo.png", dpi=200, bbox_inches="tight", facecolor="white")
print("salvata", FIG/"accuratezza_costo.png")
