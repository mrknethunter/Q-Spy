# Documentazione Modulo M-02: FewShotSVC, Equal-Budget & Valutazione RQ3

## 1. Esecuzione dei Test (`pytest`)

Assicurandosi di avere l'ambiente virtuale `.venv` attivo:

```bash
# Esecuzione della suite M-02
python -m pytest -vv -s ./tests/test_fewshot.py

# Esecuzione dell'intera suite modelli (M-02, M-03, M-04)
python -m pytest -vv -s ./tests/
```

## 2. Architettura di `models/fewshot.py`

* **Ereditarietà condivisa (`models/base.py`):** `FewShotSVC` estende `BaseFewShotModel`, ereditando la nested cross-validation stratificata adattiva (`min(cv_folds, min_class_count)` per operare senza errori anche nel regime $k=5$) e il campionamento deterministico delle configurazioni tramite `seed`.
* **Wrapper `_PrecomputedKernelSVC`:** incapsula `sklearn.svm.SVC(kernel="precomputed", decision_function_shape="ovr", random_state=seed)` (Sezione 5.3 del proposal) e gestisce tre modalità di input:
  * `kernel="projected"` (default): riceve vettori di Bloch $(N, 3n)$ o $(N, n, 3)$ e calcola la matrice di Gram tramite $k_P(x, x') = \exp\left(-\frac{\gamma}{n}\sum_{q=1}^{n}\|b_q(x)-b_q(x')\|_2^2\right)$, consentendo di ottimizzare $\gamma$ in post-processing classico senza ri-eseguire circuiti quantistici.
  * `kernel="precomputed"`: accetta direttamente matrici di Gram quadrate $(N_{\text{tr}}, N_{\text{tr}})$ in `fit()` e rettangolari $(N_{\text{te}}, N_{\text{tr}})$ in `predict()` / `score_samples()` (usato per il kernel di fedeltà $k_F$ di riferimento o kernel esterni), affettando correttamente i sotto-blocchi nei fold interni della nested CV.
  * `kernel="rbf"` o `kernel_fn`: consente test diretti su feature tabellari ($d=7$) o l'iniezione di kernel custom (come l'ablazione product-map senza porte $W_{CZ}$).

## 3. Iperparametri e Garanzia Equal-Budget (Bowles et al.)

Gli iperparametri sono configurabili nel costruttore di `FewShotSVC(...)`:

| Modalità `kernel` | Parametri nel tuning | Griglia Default | Motivazione progettuale |
| :--- | :--- | :--- | :--- |
| **`projected`** / **`rbf`** | `C` e `gamma` | `c_grid`: `[1e-2, 1e-1, 1.0, 10.0, 100.0]`<br>`gamma_grid`: `["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0]` | Spazio candidato da $5 \times 6 = 30$ configurazioni, identico a `RBFSVMBaseline` e `LaplacianSVMBaseline`. |
| **`precomputed`** | Solo `C` | `np.logspace(-3, 3, num=max(budget, 30))` | Quando la matrice di Gram è già fissata a monte ($k_F$), la griglia monodimensionale su `C` viene generata con almeno `budget` valori così da valutare esattamente lo stesso numero di configurazioni dei modelli a 2 iperparametri. |

* **Verifica formale (`assert_equal_search_budget`):** la funzione `assert_equal_search_budget(models, expected_budget=None)` ispeziona `get_budget_report()["n_configs_evaluated"]` su una sequenza mista di modelli quantistici e baseline classiche, sollevando un `AssertionError` in caso di qualsiasi discrepanza nel budget di ricerca.

## 4. Metriche RQ3 (`evaluate_fewshot`)

La funzione `evaluate_fewshot(y_true, y_pred, target_names=None)` e il metodo `model.evaluate(X, y, target_names=None)` restituiscono un dizionario con:

* `macro_f1`: F1-score macro-mediato sulle famiglie di glitch (`f1_score(..., average="macro")`).
* `balanced_accuracy`: accuratezza bilanciata (`balanced_accuracy_score`).
* `per_class`: dizionario indicizzato per classe/nome famiglia contenente `precision`, `recall`, `f1-score` e `support`.

## 5. Copertura dei Test (`tests/test_fewshot.py`)

* **Acceptance Criterion 1 (`k in {5, 10, 20, 50}`):** test parametrizzato su tutti e quattro i regimi $k$-shot del Protocollo 1 su vettori di Bloch ($n=8$ qubit, 6 classi), verificando `macro_f1 > 0.85`, `balanced_accuracy > 0.85` e la completezza del report `per_class`.
* **Modalità `precomputed` ($k_F$):** verifica del training e dell'inferenza su matrici di Gram precalcolate quadrate e rettangolari con tuning del solo parametro `C`.
* **Acceptance Criterion 2 (Equal-Budget Cross-Modello):** addestra sullo stesso split `FewShotSVC` (sia `projected` sia `precomputed`) insieme a tutte e 5 le baseline classiche (`rbf_svm`, `laplacian_svm`, `knn`, `logistic_regression`, `hist_gradient_boosting`) e verifica tramite `assert_equal_search_budget()` che tutti abbiano valutato lo stesso identico numero di configurazioni.
* **Accuratezza Metriche:** test unitario sull'esattezza numerica di `macro_f1`, `balanced_accuracy` e metriche `per_class`.

## 6. Esempio Rapido di Utilizzo

```python
from models import FewShotSVC, assert_equal_search_budget, build_baseline

q_model = FewShotSVC(kernel="projected", n_qubits=8, budget=16, seed=42, cv_folds=5)
q_model.fit(B_train, y_train)
metrics = q_model.evaluate(B_test, y_test, target_names=class_names)

c_model = build_baseline("rbf_svm", budget=16, seed=42, cv_folds=5)
c_model.fit(X_train, y_train)

assert_equal_search_budget([q_model, c_model], expected_budget=16)
```