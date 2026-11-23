# Documentazione Modulo M-04: Classical Baselines & Equal-Budget Interface

Implementazione delle baseline classiche per **Q-Spy** (`models/baselines.py`) e della relativa suite di test (`models/tests/test_baselines.py`), progettata per condividere la stessa interfaccia e lo stesso meccanismo di **equal-budget tuning** dei modelli quantistici (`M-02` e `M-03`).

---

## 1. Setup Ambiente (`venv`) ed Esecuzione Test

### Creazione e attivazione del Virtual Environment
```bash
# Creazione ambiente virtuale
python -m venv .venv

# Attivazione (Linux / macOS)
source .venv/bin/activate

# Attivazione (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Installazione dipendenze
pip install -r requirements.txt
```

### Esecuzione dei Test (`pytest`)
- Esecuzione standard degli unit tests
```bash
python -m pytest -v ./models/tests/test_baselines.py
```
- Esecuzione con stampa in console di metriche, best_params e inferenza
```bash
python -m pytest -v -s ./models/tests/test_baselines.py
```

---

## 2. Architettura Comune e Vincoli di Progetto

* **Input preprocessati:** tutti i modelli ricevono la matrice $X \in \mathbb{R}^{N \times 7}$ delle feature di Omicron già trasformate in $\log_{10}$ e scalate a monte (nessuno scaling interno, per evitare data leakage).
* **Contratto condiviso (`BaseBudgetedModel`):** classe base astratta riutilizzabile direttamente anche in `models/fewshot.py` (`M-02`) e `models/novelty.py` (`M-03`). Espone `fit()`, `predict()`, `score_samples()`, `get_budget_report()` e la factory `build_baseline(name, budget, seed, cv_folds)`.
* **Equal-Budget deterministico:** ogni modello valuta al massimo `budget` configurazioni di iperparametri estratte in modo riproducibile tramite `np.random.default_rng(seed)`.
* **Convenzioni dei due protocolli:**
  * **Protocollo 1 (`BaseFewShotBaseline`):** cross-validation stratificata adattiva (`min(cv_folds, min_class_count)` per gestire $k \in \{5, 10, 20, 50\}$) con ottimizzazione su `macro-F1`.
  * **Protocollo 2 (`BaseNoveltyBaseline`):** addestramento e calibrazione eseguiti **esclusivamente su famiglie note** (zero leakage degli outlier). `predict(X)` restituisce `0` (noto/inlier) o `1` (nuovo/outlier); `score_samples(X)` inverte il segno di `decision_function` affinché **score più alti indichino maggiore probabilità di anomalia** (allineato a AUROC, Average Precision e TPR@1%FPR).

---

## 3. Modelli Few-Shot

| Chiave Registry | Classe Baseline | Wrapper Interno | Dove modificare gli iperparametri |
| :--- | :--- | :--- | :--- |
| `rbf_svm` | `RBFSVMBaseline` | — (`SVC` nativo) | `RBFSVMBaseline._param_grid(self)` |
| `laplacian_svm` | `LaplacianSVMBaseline` | `_LaplacianSVC` | `LaplacianSVMBaseline._param_grid(self)` |
| `knn` | `KNNBaseline` | `_SafeKNeighborsClassifier` | `KNNBaseline._param_grid(self)` |
| `logistic_regression` | `LogisticRegressionBaseline` | — (`LogisticRegression` nativo) | `LogisticRegressionBaseline._param_grid(self)` |
| `hist_gradient_boosting` | `HistGradientBoostingBaseline` | — (`HistGradientBoostingClassifier` nativo) | `HistGradientBoostingBaseline._param_grid(self)` |

### Dettaglio iperparametri e scelte progettuali

1. **`RBFSVMBaseline`** (`models/baselines.py` $\rightarrow$ `RBFSVMBaseline._param_grid`)
   * **Griglia ($5 \times 6 = 30$ combinazioni):** `C` in `[1e-2, 1e-1, 1.0, 10.0, 100.0]`; `gamma` in `["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0]`.
   * **Perché:** copre 5 ordini di grandezza per regolarizzazione e larghezza del kernel. Impone `decision_function_shape="ovr"` (*One-vs-Rest*, Sez. 5.3 del proposal) e `random_state=self.seed`.

2. **`LaplacianSVMBaseline`** (`models/baselines.py` $\rightarrow$ `LaplacianSVMBaseline._param_grid`)
   * **Griglia ($5 \times 6 = 30$ combinazioni):** identica a `RBFSVMBaseline` (`C` e `gamma`).
   * **Perché:** garantisce simmetria totale dello spazio di ricerca rispetto al kernel RBF. Il wrapper `_LaplacianSVC` è necessario perché `SVC` non accetta `kernel="laplacian"` come stringa nativa né `"scale"`/`"auto"` in `laplacian_kernel`: risolve $\gamma$ numericamente e passa la matrice precomputata a `SVC(kernel="precomputed", decision_function_shape="ovr")`.

3. **`KNNBaseline`** (`models/baselines.py` $\rightarrow$ `KNNBaseline._param_grid`)
   * **Griglia ($5 \times 2 \times 3 = 30$ combinazioni):** `n_neighbors` in `[1, 3, 5, 7, 9]`; `weights` in `["uniform", "distance"]`; `metric` in `["euclidean", "manhattan", "chebyshev"]`.
   * **Perché:** esplora vicinati dispari e diverse metriche $L_p$ nello spazio $d=7$. Il wrapper `_SafeKNeighborsClassifier` limita dinamicamente `n_neighbors` a `min(n_neighbors, len(X_fold))` per evitare `ValueError` nei fold interni ridotti (es. $k=5$).

4. **`LogisticRegressionBaseline`** (`models/baselines.py` $\rightarrow$ `LogisticRegressionBaseline._param_grid`)
   * **Griglia ($7 \times 2 \times 2 = 28$ combinazioni):** `C` in `[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]`; `class_weight` in `[None, "balanced"]`; `fit_intercept` in `[True, False]`.
   * **Perché:** fornisce un benchmark lineare ben regolarizzato; `max_iter=1000` e `random_state=self.seed` assicurano convergenza stabile e riproducibilità.

5. **`HistGradientBoostingBaseline`** (`models/baselines.py` $\rightarrow$ `HistGradientBoostingBaseline._param_grid`)
   * **Griglia ($3 \times 3 \times 3 \times 4 \times 2 = 216$ combinazioni):** `learning_rate` in `[0.03, 0.1, 0.2]`; `max_iter` in `[50, 100, 200]`; `max_leaf_nodes` in `[7, 15, 31]`; `min_samples_leaf` in `[1, 2, 4, 8]`; `l2_regularization` in `[0.0, 1.0]`.
   * **Perché:** il default di scikit-learn (`min_samples_leaf=20`) impedirebbe qualsiasi split nel regime $k=5$ con 6 classi ($N_{\text{train}} = 30$). Abbassarlo a $[1, 8]$ e disattivare `early_stopping=False` rende l'albero effettivamente addestrabile su pochi shot.

---

## 4. Modelli Novelty Detection

| Chiave Registry | Classe Baseline | Wrapper Interno | Dove modificare gli iperparametri |
| :--- | :--- | :--- | :--- |
| `rbf_ocsvm` | `RBFOCSVMBaseline` | — (`OneClassSVM` nativo) | `RBFOCSVMBaseline._param_grid(self)` |
| `isolation_forest` | `IsolationForestBaseline` | — (`IsolationForest` nativo) | `IsolationForestBaseline._param_grid(self)` |
| `lof` | `LOFBaseline` | `_SafeLocalOutlierFactor` | `LOFBaseline._param_grid(self)` |

### Dettaglio iperparametri e scelte progettuali

1. **`RBFOCSVMBaseline`** (`models/baselines.py` $\rightarrow$ `RBFOCSVMBaseline._param_grid`)
   * **Griglia ($5 \times 6 = 30$ combinazioni):** `nu` in `[0.01, 0.05, 0.1, 0.15, 0.2]`; `gamma` in `["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0]`.
   * **Perché:** `nu` controlla l'upper bound sui falsi positivi sugli inlier ed è usato da `BaseNoveltyBaseline` come target di calibrazione nei fold di validazione composti da sole classi note.

2. **`IsolationForestBaseline`** (`models/baselines.py` $\rightarrow$ `IsolationForestBaseline._param_grid`)
   * **Griglia ($3 \times 4 \times 4 \times 3 \times 2 = 288$ combinazioni):** `n_estimators` in `[100, 200, 300]`; `max_samples` in `["auto", 0.5, 0.75, 1.0]`; `contamination` in `[0.01, 0.05, 0.1, 0.15]`; `max_features` in `[0.5, 0.75, 1.0]`; `bootstrap` in `[False, True]`.
   * **Perché:** `contamination` è allineato al range di `nu`; `max_samples` e `max_features` espressi come frazioni funzionano su qualsiasi dimensione di campione/fixture sulle $d=7$ feature.

3. **`LOFBaseline`** (`models/baselines.py` $\rightarrow$ `LOFBaseline._param_grid`)
   * **Griglia ($5 \times 3 \times 4 = 60$ combinazioni):** `n_neighbors` in `[5, 10, 15, 20, 30]`; `metric` in `["euclidean", "manhattan", "chebyshev"]`; `contamination` in `[0.01, 0.05, 0.1, 0.15]`.
   * **Perché:** impone `novelty=True` (obbligatorio in scikit-learn per valutare nuovi punti di test) e usa `_SafeLocalOutlierFactor` per limitare `n_neighbors` a `len(X_fold) - 1` su training set piccoli.

---

## 5. Esempio Rapido di Utilizzo nel Runner (`L-02`)

```python
from models.baselines import build_baseline

# Few-shot (Protocollo 1)
clf = build_baseline("laplacian_svm", budget=16, seed=42, cv_folds=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
audit = clf.get_budget_report()

# Novelty Detection (Protocollo 2 - solo classi note in fit)
det = build_baseline("isolation_forest", budget=16, seed=42, cv_folds=5)
det.fit(X_train_known)
is_novel = det.predict(X_test)           # 0 = noto, 1 = anomalia
anomaly_scores = det.score_samples(X_test)  # più alto = più anomalo
```