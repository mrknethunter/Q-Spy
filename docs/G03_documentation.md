# G-03 — Confronto tra tier e controllo del drift

## Obiettivo e confini

`diagnostics/compare.py` confronta Bloch vector e matrici kernel già disponibili,
tra tier (per esempio L0, L1, L2 e HW) e tra reference point misurati all'inizio
e alla fine di una campagna. Il modulo opera in memoria e dipende soltanto da
NumPy. Non legge lo store, non importa `store.io`, non esegue circuiti e non
costruisce il projected kernel.

L'interfaccia esterna dei risultati Bloch segue la forma L-02
`read_bloch(run_id) -> (ids, B, meta)`: `B` è `float32` o `float64`, con forma
`(N, n_qubits, 3)` e valori in `[-1, 1]`. I kernel vengono passati già calcolati
insieme agli ID che ne definiscono righe e colonne.

## Fondamento e convenzioni

La proposta Q-Spy separa il livello di esecuzione quantum dalle diagnostiche
NumPy e specifica il confronto L0/L2/HW per RQ1. Nomina l'errore di Frobenius
dei Bloch vector e l'alignment dei kernel hardware/ideale; descrive inoltre il
controllo del drift tramite reference point misurati a inizio e fine campagna.
La proposta non definisce una formula specifica di alignment dei kernel.

G-03 sceglie quindi esplicitamente il **centered normalized Frobenius
alignment** richiesto nella task. Questa scelta è una convenzione implementativa
di G-03, non una formula attribuita alla proposta. La funzione non calcola il
projected kernel: confronta solo matrici fornite dal chiamante.

La consultazione delle fonti scientifiche disponibili nel progetto informa il
contesto, senza attribuire loro formule che non specificano:

- Huang et al., *Power of data in quantum machine learning* (2021), tratta
  modelli quantistici proiettati e le loro feature misurabili.
- Thanasilp et al., *Exponential concentration in quantum kernel methods*
  (2024), analizza concentrazione dei kernel e fonti tra cui il rumore.
- Pranjic et al., *Unsupervised Quantum Anomaly Detection on Noisy Quantum
  Processors* (2024), studia kernel proiettati con tomografia parziale su
  hardware e riporta che il drift di calibrazione può limitare la precisione
  degli esperimenti più lunghi.
- Shaydulin e Wild, *Importance of Kernel Bandwidth in Quantum Machine
  Learning* (2022), e Canatar et al., *Bandwidth Enables Generalization in
  Quantum Kernel Models* (2023), trattano l'effetto della bandwidth sulle
  proprietà dei kernel quantistici.
- Bowles et al., *Better than classical? The subtle art of benchmarking
  quantum machine learning models* (2024), discute cautele nel benchmarking
  QML.

Questi lavori motivano il confronto quantitativo e il monitoraggio del drift;
non definiscono le metriche specifiche implementate qui.

## Metriche

### Errore dei Bloch vector

Per ogni qubit `q`, sui campioni abbinati per ID:

\[
E_q = \sqrt{\frac{1}{3N}\sum_{i=1}^{N}\sum_{\ell=1}^{3}
  (B^{(a)}_{i,q,\ell}-B^{(b)}_{i,q,\ell})^2}.
\]

L'errore RMSE globale normalizza anche per il numero di qubit:

\[
E_{\mathrm{global}} = \sqrt{\frac{1}{3Nn}\sum_{i=1}^{N}
  \sum_{q=1}^{n}\sum_{\ell=1}^{3}
  (B^{(a)}_{i,q,\ell}-B^{(b)}_{i,q,\ell})^2}.
\]

La norma non normalizzata è riportata separatamente:

\[
F_{\mathrm{global}}=\lVert B_a-B_b\rVert_F.
\]

La RMSE per qubit e globale sono normalizzate rispetto al numero di elementi;
la norma di Frobenius cresce anche con la dimensione dell'input e non va
interpretata come metrica normalizzata.

### Kernel alignment

Per matrici Gram simmetriche `K_a` e `K_b` sugli stessi campioni, G-03 centra
entrambe con `H = I - 11ᵀ/N` e calcola:

\[
A(K_a,K_b)=\frac{\langle HK_aH,HK_bH\rangle_F}
 {\lVert HK_aH\rVert_F\,\lVert HK_bH\rVert_F}.
\]

Il risultato è compreso tra `-1` e `1`; piccoli superamenti dovuti
all'arrotondamento vengono limitati a questo intervallo. Se una matrice
centrata ha norma nulla, l'alignment non è definito e viene sollevato
`ValueError`.

### Drift

Il drift riusa le tre metriche Bloch sui reference ID comuni a inizio e fine
campagna. Se vengono fornite entrambe le matrici kernel, riporta anche il loro
centered normalized Frobenius alignment. Non applica soglie e non trasforma i
valori in giudizi di accettabilità.

## API pubbliche

| Funzione | Firma | Risultato |
|---|---|---|
| `bloch_frobenius_error` | `(ids_a, B_a, ids_b, B_b) -> dict` | ID confrontati, `rmse_per_qubit`, `rmse_global`, `frobenius_global` |
| `kernel_alignment` | `(K_a, K_b, ids_a, ids_b) -> float` | Alignment centrato normalizzato |
| `compare_runs` | `(run_a, run_b, *, tier_a, tier_b, kernel_a=None, kernel_b=None) -> dict` | Report tra tier con errori Bloch, kernel opzionali e metadati |
| `drift_report` | `(run_start, run_end, *, run_start_id=None, run_end_id=None, kernel_start=None, kernel_end=None) -> dict` | Report start/end con ID dei run, reference ID, metriche e metadati |

`run_a`, `run_b`, `run_start` e `run_end` seguono la forma
`(ids, B, meta)` di L-02. `kernel_a`, `kernel_b`, `kernel_start` e
`kernel_end`, se usati, hanno forma `(sample_ids, K)`. Le due matrici kernel
vanno fornite insieme; i loro ID devono corrispondere agli ID dei rispettivi
risultati Bloch.

I due insiemi di ID devono coincidere. Se l'ordinamento è diverso, le righe di
`B_b` vengono riallineate esplicitamente e per le matrici vengono riordinate
righe e colonne secondo `ids_a`. Il report elenca gli ID nell'ordine di
riferimento A (o start). Gli ID devono essere univoci e serializzabili in JSON.

`drift_report` usa gli ID espliciti se forniti; altrimenti legge
`meta['run_id']`. Se nessuna delle due fonti identifica un run, solleva
`ValueError` invece di inventarne uno. Metadati vuoti o parziali sono mantenuti
così come disponibili. I metadati devono essere convertibili in JSON; array e
scalari NumPy vengono convertiti nei corrispondenti valori Python.

## Struttura dei report

`compare_runs` restituisce un dizionario con:

- `comparison_type: "tier"` e le etichette `tier_a`, `tier_b`;
- `bloch_vector_error`, con `sample_ids`, RMSE per qubit, RMSE globale e norma
  di Frobenius;
- `kernel_comparison`, `None` se non sono forniti kernel, altrimenti ID e
  alignment centrato;
- `meta_a` e `meta_b`, normalizzati per la serializzazione JSON.

`drift_report` restituisce `comparison_type: "drift"`, gli ID dei run iniziale
e finale, `reference_ids`, `bloch_drift`, `kernel_alignment` (oppure `None`),
`meta_start` e `meta_end`.

## Esempio sintetico

```python
import numpy as np

from diagnostics.compare import compare_runs, drift_report

ids = ["ref-0", "ref-1", "ref-2"]
ideal = np.zeros((3, 2, 3), dtype=np.float32)
hardware = ideal.copy()
hardware[:, 1, :] += np.float32(0.3)

report = compare_runs(
    (ids, ideal, {"run_id": "ideal-run"}),
    (ids, hardware, {"run_id": "hardware-run"}),
    tier_a="L0",
    tier_b="HW",
)
assert np.allclose(report["bloch_vector_error"]["rmse_per_qubit"], [0.0, 0.3])

drift = drift_report(
    (ids, ideal, {"run_id": "campaign-start"}),
    (ids, hardware, {"run_id": "campaign-end"}),
)
```

## Validazioni, dipendenze e limiti

- Dipendenza effettiva: NumPy; pytest serve solo per i test.
- `B` deve rispettare forma, dtype `float32` o `float64`, finitezza e intervallo
  L-02.
- Dopo la validazione, i Bloch vector sono rappresentati come `float64` per il
  calcolo delle metriche; gli array ricevuti dal chiamante restano invariati.
- ID duplicati, mancanti, incompatibili o con cardinalità diversa sono errori.
- Le matrici kernel devono essere reali, quadrate, finite e simmetriche entro
  tolleranza numerica (`rtol=1e-7`, `atol=1e-10`). Non viene verificata la
  semidefinitezza positiva perché non è necessaria per l'alignment definito.
- Non sono hardcodati dimensioni di dataset o conteggi di reference point.
- Il modulo non persiste risultati, non carica filesystem o Parquet e non
  dipende da Qiskit, backend o `store.io`.

Il contratto L-02 effettivo è documentato in `docs/contracts.md`; specifica
`float32` o `float64` per `B`, in accordo con la validazione di questo modulo.

## Test

`tests/test_compare.py` copre metriche analitiche, riallineamento degli ID,
errori di input, kernel centrati, report start/end e un test di accettazione
sintetico con perturbazioni note e report validato tramite `json.dumps`.
Comando previsto dalla root del repository:

```bash
.venv/bin/python -m pytest tests/test_compare.py
```

Il risultato dell'esecuzione viene riportato nel resoconto della task; la
presenza dei test da sola non è dichiarata come verifica di accettazione.
