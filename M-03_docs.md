# Documentazione Modulo M-03: NoveltyOCSVM & Valutazione RQ4

Implementazione del rilevatore di novità principale di **Q-Spy** (`models/novelty.py`) e della relativa suite di test (`tests/test_novelty.py`), basato su `OneClassSVM(kernel="precomputed")` e allineato al **Protocollo 2 (Leave-One-Class-Out)** e alle metriche **RQ4**.

## 1. Esecuzione dei Test (`pytest`)

Assicurandosi di avere l'ambiente virtuale `.venv` attivo:

```bash
# Esecuzione standard della suite M-03
python -m pytest -v ./tests/test_novelty.py

# Esecuzione con stampa in console di iperparametri selezionati e metriche RQ4
python -m pytest -v -s ./tests/test_novelty.py
```

## 2. Architettura di `models/novelty.py`

* **Ereditarietà condivisa (`models/base.py`):** `NoveltyOCSVM` estende `BaseNoveltyModel` (estratta da `baselines.py` insieme a `BaseBudgetedModel` e `BaseFewShotModel`), ereditando lo stesso meccanismo di **equal-budget tuning** deterministico tramite `seed` e le convenzioni uniformi di output (`0` = famiglia nota/inlier, `1` = nuova famiglia/outlier; `score_samples(X)` con segno invertito affinché valori più alti indichino maggiore anomalia).
* **Costruzione del Kernel Proiettato ($k_P$):** la funzione `projected_quantum_kernel_matrix(B_a, B_b, gamma)` accetta vettori di Bloch sia in formato 2D $(N, 3n)$ sia 3D $(N, n, 3)$ nell'intervallo $[-1, 1]$ e calcola:
  $$k_P(x, x') = \exp\left(-\frac{\gamma}{n}\sum_{q=1}^{n} \|b_q(x) - b_q(x')\|_2^2\right)$$
  Permettendo di ottimizzare $\gamma$ interamente in post-processing classico senza ri-eseguire i circuiti quantistici.
* **Wrapper `_PrecomputedKernelOCSVM`:** incapsula `sklearn.svm.OneClassSVM(kernel="precomputed")`. Supporta nativamente:
  * `kernel="projected"`: calcolo esatto della matrice di Gram quantistica proiettata sui vettori di Bloch.
  * `kernel="rbf"`: kernel RBF precalcolato per test su fixture tabellari ($d=7$).
  * `kernel_fn`: funzione di kernel personalizzata iniettabile dall'esterno (es. per il kernel di fedeltà $k_F$ o per `kernels/projected.py`).

## 3. Iperparametri e Strategie di Tuning

Gli iperparametri si configurano direttamente nel costruttore di `NoveltyOCSVM(...)` (o nei default della classe):

| Parametro | Default | Ruolo e strategia di risoluzione |
| :--- | :--- | :--- |
| `nu_grid` | `[0.01, 0.05, 0.1, 0.15, 0.2]` | Limite superiore alla frazione di falsi positivi sugli inlier. Usato come target di calibrazione nei fold di validazione composti da sole famiglie note. |
| `gamma_grid` | `["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0]` | Larghezza del kernel. Per `kernel="projected"`, `"auto"` corrisponde a $\gamma = 1.0$ e `"scale"` a $\gamma = 1 / \mathrm{Var}(B_{\mathrm{train}})$ (poiché la normalizzazione per $1/n$ è già esplicita nella formula di $k_P$). |
| `budget` | `16` | Numero massimo di coppie $(\nu, \gamma)$ campionate in modo riproducibile tramite `seed`. |

**Garanzia Anti-Leakage:** `NoveltyOCSVM.fit(X_known)` esegue la K-Fold cross-validation esclusivamente sui campioni delle famiglie note. La classe held-out (es. `Fast_Scattering` in $O_3$) non viene mai utilizzata durante il tuning né nel fitting finale.

## 4. Metriche RQ4 (`evaluate_novelty`)

La funzione `evaluate_novelty(y_true, anomaly_scores, target_fpr=0.01)` e il metodo `model.evaluate(X, y)` restituiscono un dizionario con le tre metriche richieste dal proposal:

* `auroc`: Area Under the ROC Curve (`roc_auc_score`).
* `average_precision`: Area sotto la curva Precision-Recall (`average_precision_score`).
* `tpr_at_1pct_fpr`: True Positive Rate interpolato in corrispondenza di un False Positive Rate pari all'1% ($\mathrm{FPR} = 0.01$).

## 5. Copertura dei Test (`tests/test_novelty.py`)

* **Proprietà matematiche di $k_P$:** verifica simmetria della matrice di Gram, diagonale unitaria, valori in $(0, 1]$ ed equivalenza tra input 2D $(N, 3n)$ e 3D $(N, n, 3)$.
* **Acceptance Criterion 1 (`AUROC > 0.9`):** verifica su fixture LOCO sia nello spazio dei vettori di Bloch ($n=8$ qubit, 24 dimensioni) sia su feature tabellari ($d=7$).
* **Acceptance Criterion 2 (Zero Leakage):** utilizza un *spy kernel* e un set held-out contaminato (`999.0`) per dimostrare che nessun campione della classe nuova viene mai letto durante `fit()` e che i parametri selezionati e i support vector restano invariati.
* **Determinismo, Budget e Metriche:** verifica il rispetto del budget, la riproducibilità tramite `seed` e l'esattezza del calcolo di `auroc`, `average_precision` e `tpr_at_1pct_fpr`.

## 6. Esempio Rapido di Utilizzo

```python
from models.novelty import NoveltyOCSVM

model = NoveltyOCSVM(kernel="projected", n_qubits=8, budget=16, seed=42, cv_folds=5)
model.fit(X_train_known)

metrics = model.evaluate(X_test, y_test)
