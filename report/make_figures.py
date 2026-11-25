# L-06: regenerates the report figures from committed results; needs H-02 (results/h02, store hw-20261125*), L-04 (results/e2e) and E-03 (results/e03) in the checkout
# usage: python report/make_figures.py   (q06.png and h01.png are copies of results/q06 and results/pilot)
import sys, json
sys.path.insert(0, '.')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
Q, C, B3, Y4 = '#00a88a', '#f47230', '#257de0', '#f2a600'
INK, MUTED, GRID, BG = '#101A2B', '#4A5566', '#E3DFD6', '#FDFDFB'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.titleweight': 'bold', 'axes.titlesize': 14, 'axes.titlecolor': INK, 'axes.titlelocation': 'left'})
out = 'report/fig/'

from data.curate import load_curated
df = load_curated()
vc = df.label.value_counts().sort_values()
fig, ax = plt.subplots(figsize=(9, 5.2))
cols = [C if k == 'Fast_Scattering' else '#7f8ea3' for k in vc.index]
ax.barh(vc.index.str.replace('_', ' '), vc.values, color=cols, height=0.62)
for i, v in enumerate(vc.values):
    ax.text(v + 1500, i, f'{v:,}'.replace(',', '.'), va='center', color=INK, fontsize=12)
ax.set(xlabel='glitch curati (confidenza ≥ 0,9)', xlim=(0, vc.max() * 1.18))
ax.grid(axis='y', visible=False)
ax.set_title('Famiglie del dataset; in arancio quella tenuta nascosta')
fig.tight_layout(); fig.savefig(out + 'dataset.png', dpi=200); plt.close(fig)


e = pd.read_csv('results/e01/e01_gravityspy.csv')
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
d = e[(e.L == 2) & (e.s == 1.0)] if (e.s == 1.0).any() else e[(e.L == 2) & (np.isclose(e.s, 1.0, atol=.01))]
for k, col, lab in (('fidelity', B3, 'fidelity kernel'), ('projected', Q, 'kernel proiettato')):
    dd = d[d.kernel == k].sort_values('n')
    ax[0].plot(dd.n, dd.var_offdiag, marker='o', color=col, lw=2, ms=8)
    ax[0].text(dd.n.iloc[-1] + .3, dd.var_offdiag.iloc[-1], lab, color=INK, va='center', fontsize=12)
ax[0].set(yscale='log', xlabel='numero di qubit', ylabel='varianza del kernel (log)', xlim=(3, 19.5))
ax[0].set_title('(a) Concentrazione al crescere dei qubit (s = 1)')
d = e[(e.L == 2) & (e.n == 8)]
for k, col, lab in (('fidelity', B3, 'fidelity'), ('projected', Q, 'proiettato')):
    dd = d[d.kernel == k].sort_values('s')
    ax[1].plot(dd.s, dd.kta, marker='o', color=col, lw=2, ms=8, label=lab)
ax[1].axvline(0.1778, color=MUTED, ls='--', lw=1)
ax[1].text(0.19, ax[1].get_ylim()[0] + 0.02, 's scelto = 0,18', color=MUTED, fontsize=11)
ax[1].set(xscale='log', xlabel='bandwidth s (log)', ylabel='allineamento con le classi (KTA)')
ax[1].legend(frameon=False)
ax[1].set_title('(b) Allineamento contro bandwidth (n = 8)')
fig.tight_layout(); fig.savefig(out + 'e01_slide.png', dpi=200); plt.close(fig)

f = pd.read_csv('results/e2e/fewshot_sweep.csv')
g = f.groupby(['model', 'k']).macro_f1.agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(8.5, 5))
for m, col, lab, dx in (('projected_svc', Q, 'SVM, kernel proiettato', -0.03), ('rbf_svm', C, 'SVM RBF classica', 0.03)):
    gg = g[g.model == m]
    x = gg.k * (1 + dx)
    ax.errorbar(x, gg['mean'], yerr=gg['std'], marker='o', color=col, lw=2, ms=8, capsize=4)
    ax.plot([], [], marker='o', color=col, lw=2, label=lab)
ax.legend(frameon=False, loc='lower right')
ax.set(xscale='log', xticks=[5, 10, 20, 50], xticklabels=['5', '10', '20', '50'], xlim=(4, 65), ylim=(0.6, 0.9),
       xlabel='esempi per famiglia (k)', ylabel='macro-F1 (7 famiglie, 700 test)')
ax.set_title('Few-shot: media e deviazione su 5 seed')
fig.tight_layout(); fig.savefig(out + 'fewshot.png', dpi=200); plt.close(fig)

m = pd.read_csv('results/e03/e03_metrics.csv')
t = m[(m.scope == 'target_family') & (m.metric == 'auroc')].groupby(['held_out', 'model']).value.mean().unstack()
order = t.mean(axis=1).sort_values().index
fig, ax = plt.subplots(figsize=(10, 5.6))
spec = (('projected_ocsvm', Q, 'OC-SVM proiettato', 'o'), ('rbf_ocsvm', C, 'OC-SVM RBF', 's'),
        ('isolation_forest', B3, 'Isolation Forest', 'D'), ('lof', Y4, 'LOF', '^'))
for i, fam in enumerate(order):
    ax.plot([t.loc[fam].min(), t.loc[fam].max()], [i, i], color=GRID, lw=6, solid_capstyle='round', zorder=1)
for mname, col, lab, mk in spec:
    ax.scatter(t.loc[order, mname], range(len(order)), color=col, marker=mk, s=80, label=lab, zorder=3, edgecolor=BG, linewidth=1.5)
ax.set_yticks(range(len(order)), [o.replace('_', ' ') for o in order])
for lbl in ax.get_yticklabels():
    if lbl.get_text() == 'Fast Scattering':
        lbl.set_fontweight('bold'); lbl.set_color(INK)
ax.set(xlabel='AUROC (famiglia nascosta contro famiglie note), media su 10 seed', xlim=(0.6, 1.0))
ax.axvline(0.5, color=MUTED, lw=1)
ax.legend(frameon=False, loc='lower right', fontsize=11)
ax.set_title('Ogni famiglia, a turno, fa da “sconosciuta”')
fig.tight_layout(); fig.savefig(out + 'e03_families.png', dpi=200); plt.close(fig)


from store.io import read_bloch
def L(r, e, tt): return np.asarray(read_bloch(f'{r}-{e}-{tt}')[1], float)
R = 'hw-20261125T124523Z'
s = json.load(open('results/h02/h02_summary.json'))
ex = pd.read_csv('results/h02/h02_experiments.csv').set_index('experiment')
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [1.3, 1, 1]})
bins = np.linspace(-.14, .14, 57)
for (e_, tt, col, lab) in (('main', 'L2', B3, 'emulatore del chip'), ('main', 'HW', Q, 'chip, con mitigazione'), ('control', 'HW', C, 'chip, senza mitigazione')):
    dlt = (L(R, e_, tt) - L(R, e_, 'L0')).ravel()
    ax[0].hist(dlt, bins, histtype='step', lw=2.2, color=col, density=True, label=f'{lab}: {np.sqrt((dlt**2).mean()):.3f}')
ax[0].legend(frameon=False, fontsize=11, title='errore tipico (rmse)', title_fontsize=11, loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.55)
ax[0].set(xlabel='valore misurato − valore esatto', ylabel='densità', yticks=[])
ax[0].set_title('(a) Errore sui vettori di Bloch')
vals = [ex.loc['main', 'kta_L0'], ex.loc['main', 'kta_L2'], ex.loc['main', 'kta_HW'], s['repeat_main_kta_HW']]
ax[1].bar(range(4), vals, color=['#7f8ea3', B3, Q, Q], width=0.6)
for i, v in enumerate(vals):
    ax[1].text(i, v + .008, f'{v:.3f}'.replace('.', ','), ha='center', color=INK, fontsize=12)
ax[1].set(xticks=range(4), ylim=(0, .62), ylabel='KTA, 120 glitch')
ax[1].set_xticklabels(['esatto', 'emulatore', 'chip', 'chip, 2ª corsa'], fontsize=11)
ax[1].grid(axis='x', visible=False)
ax[1].set_title('(b) Informazione sulle classi')
nv = s['novelty_preview_auroc']
vals = [nv['projected:L0'], nv['projected:HW'], nv['rbf_features:features']]
ax[2].bar(range(3), vals, color=['#7f8ea3', Q, C], width=0.6)
for i, v in enumerate(vals):
    ax[2].text(i, v + .008, f'{v:.3f}'.replace('.', ','), ha='center', color=INK, fontsize=12)
ax[2].set(xticks=range(3), ylim=(0.5, 1.02), ylabel='AUROC Fast Scattering')
ax[2].set_xticklabels(['proiettato,\nesatto', 'proiettato,\nchip', 'RBF classico,\nfeature'], fontsize=11)
ax[2].grid(axis='x', visible=False)
ax[2].set_title('(c) Trovare Fast Scattering (15 su 120)')
fig.tight_layout(); fig.savefig(out + 'h02_slide.png', dpi=200); plt.close(fig)

c = pd.read_csv('results/h02/h02_concentration.csv')
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for (k, tt), col, ls, lab in ((('fidelity', 'L0'), B3, '-', 'fidelity, esatto'), (('projected', 'L0'), Q, '-', 'proiettato, esatto'), (('projected', 'HW'), Q, '--', 'proiettato, chip')):
    dd = c[(c.kernel == k) & (c.tier == tt)].sort_values('n_qubits')
    ax.plot(dd.n_qubits, dd.variance, color=col, ls=ls, marker='o' if tt == 'L0' else 's', lw=2, ms=8)
    ax.text(12.4, dd.variance.iloc[-1], lab, color=INK, va='center', fontsize=12)
ax.set(yscale='log', xticks=[4, 8, 12], xlim=(3.5, 16.5), xlabel='qubit', ylabel='varianza del kernel (log)')
ax.set_title('Concentrazione misurata sul chip (s = 0,18, 30 glitch)')
fig.tight_layout(); fig.savefig(out + 'h02_conc.png', dpi=200); plt.close(fig)
print('ok')
