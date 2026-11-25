# Documentazione Modulo M-06: Aggregazione Statistica & Tabelle (Bootstrap & Wilcoxon)

Implementazione del modulo statistico di **Q-Spy** (`stats.py`) e della relativa suite di test (`tests/test_stats.py`), progettato per aggregare le metriche sperimentali sui random seed ($S \ge 10$), eseguire i test di significatività tra modelli quantistici e classici (Sezione 5.5 del proposal) ed esportare tabelle pronte per la pubblicazione in Markdown e LaTeX.

## 1. Esecuzione dei Test (`pytest`)

Assicurandosi di avere l'ambiente virtuale `.venv` attivo:

```bash
python -m pytest -vv ./tests/test_stats.py
```

## 2. Architettura e Indipendenza (`stats.py`)

* **Modulo standalone (zero accoppiamento):** `stats.py` non importa nulla da `models/`, `kernels/` o altri moduli interni del progetto, ma dipende esclusivamente da `numpy` e `scipy.stats`.
* **Strutture dati immutabili:** i risultati vengono restituiti tramite dataclass `@dataclass(frozen=True)`:
  * `BootstrapSummary(mean, std, ci_low, ci_high, n_seeds)`
  * `WilcoxonResult(statistic, p_value, mean_diff, significant)`

## 3. API Pubblica e Funzionalità

| Funzione | Descrizione e dettagli implementativi |
| :--- | :--- |
| `bootstrap_mean_ci(scores, n_resamples=10_000, confidence_level=0.95, seed=42)` | Calcola media campionaria, deviazione standard ($ddof=1$) e intervallo di confidenza percentile non parametrico ($[2.5\%, 97.5\%]$) tramite ricampionamento vettorizzato riproducibile (`np.random.default_rng(seed)`). Gestisce anche array a varianza nulla (`ci_low == ci_high == mean`). |
| `paired_wilcoxon(quantum_scores, classical_scores, alpha=0.05, alternative="two-sided")` | Esegue il test dei ranghi con segno di Wilcoxon appaiato (`scipy.stats.wilcoxon`) su array allineati seed-per-seed. Intercetta il caso di pareggio perfetto ($Q_i - C_i = 0$ per ogni seed) restituendo `statistic=0.0, p_value=1.0, significant=False`. |
| `compare_best_quantum_vs_classical(quantum_results, classical_results, ...)` | Seleziona automaticamente il miglior modello quantistico e la miglior baseline classica in base alla media sui seed ed esegue `paired_wilcoxon` tra i due vincitori. |
| `format_markdown_table(results, ...)` | Genera una tabella GitHub-Flavored Markdown con celle formattate come `mean [ci_low, ci_high]` ed evidenziazione in grassetto (`**...**`) del valore massimo di ogni colonna. |
| `format_latex_table(results, ...)` | Genera un ambiente `table` LaTeX in stile `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) con grassetto `\textbf{...}` sul migliore per colonna ed escape automatico dei caratteri speciali (`_`, `%`, `&`, `$`, `#`, `{`, `}`). |

## 4. Formato dei Dati in Ingresso

`stats.py` non riceve istanze di classificatori né matrici di feature, ma opera esclusivamente sugli **scalari delle metriche raccolti attraverso i seed**:

1. **Vettore 1D `[score_seed_0, ..., score_seed_9]`** $\rightarrow$ usato da `bootstrap_mean_ci` e `paired_wilcoxon`. In `paired_wilcoxon`, l'indice $i$ deve corrispondere allo stesso identico seed/split per entrambi i vettori.
2. **Dizionario piatto `{nome_modello: [score_seed_0, ...]}`** $\rightarrow$ usato da `compare_best_quantum_vs_classical` (un dizionario per i modelli quantistici e uno per i classici, riferiti alla stessa metrica).
3. **Dizionario nidificato `{nome_modello: {colonna: [score_seed_0, ...]}}`** $\rightarrow$ usato da `format_markdown_table` e `format_latex_table`, dove la chiave esterna definisce le righe (modelli) e quella interna definisce le colonne (metriche come `macro_f1`, `auroc`, oppure regimi come `k=5`, `k=10`, `k=20`, `k=50`).

## 5. Copertura dei Test (`tests/test_stats.py`)

* **Validazione contro `scipy.stats.bootstrap`:** verifica esattezza di `mean`, `std` e convergenza degli estremi `ci_low` e `ci_high` al 95% su input noti a 10 seed, oltre ai casi limite (input costante, array 2D, `NaN`, lunghezza $< 2$).
* **Validazione contro `scipy.stats.wilcoxon`:** verifica esattezza della statistica $W$ e del `p_value` bilaterale ($W=0, p=2/1024$ su 10 seed), gestione di differenze nulle e controllo su lunghezze disuguali.
* **Selezione vincitori e tabelle:** verifica la scelta automatica del miglior modello quantistico vs. miglior classico e la correttezza sintattica dell'output Markdown e LaTeX (`booktabs` ed escape degli underscore).

## 6. Esempio Rapido di Utilizzo

```python
from stats import (
    compare_best_quantum_vs_classical,
    format_latex_table,
    format_markdown_table,
)

# Dizionari popolati durante il loop sui 10 seed
results = {
    "fewshot_svc": {"macro_f1": q_f1_seeds, "balanced_acc": q_acc_seeds},
    "rbf_svm": {"macro_f1": rbf_f1_seeds, "balanced_acc": rbf_acc_seeds},
    "knn": {"macro_f1": knn_f1_seeds, "balanced_acc": knn_acc_seeds},
}

comparison = compare_best_quantum_vs_classical(
    quantum_results={"fewshot_svc": q_f1_seeds},
    classical_results={"rbf_svm": rbf_f1_seeds, "knn": knn_f1_seeds},
)
print(f"p-value: {comparison['wilcoxon'].p_value:.4f}")
print(format_markdown_table(results))
print(format_latex_table(results, caption="Protocol 1 results.", label="tab:p1"))
```