# G-01 — Diagnostica della concentrazione dei kernel

## Scopo

La task G-01 implementa alcune metriche per analizzare una matrice kernel `K` e capire come sono distribuite le similarità tra le osservazioni e quanto il kernel rappresenta le etichette del problema.

I deliverable sono:

- `diagnostics/concentration.py`
- `diagnostics/alignment.py`

Questi moduli **analizzano** matrici kernel già calcolate: non costruiscono il kernel e non addestrano modelli.

## Concetti teorici

### 1. Valori fuori diagonale

Una matrice kernel contiene la similarità tra ogni coppia di osservazioni:

$$
K_{ij}=k(x_i,x_j)
$$

La diagonale contiene le similarità di ogni osservazione con sé stessa. Per questo, per studiare quanto le osservazioni diverse si assomigliano, si considerano gli elementi fuori diagonale.

Per una matrice simmetrica, il codice estrae solo il triangolo superiore: ogni coppia viene contata una volta, senza duplicati.

### 2. Varianza fuori diagonale

La varianza misura quanto i valori fuori diagonale sono dispersi attorno alla loro media.

- Varianza bassa: le similarità tra coppie distinte sono simili tra loro; può essere un segnale di concentrazione.
- Varianza più alta: le similarità presentano una maggiore variabilità.

Il codice usa la varianza della popolazione (`ddof=0`).

### 3. Spettro degli autovalori

Gli autovalori descrivono come l'informazione del kernel è distribuita nelle sue direzioni principali.

- Uno spettro dominato da pochi autovalori indica che gran parte della struttura è concentrata in poche direzioni.
- Uno spettro più distribuito indica che contribuiscono più direzioni.

`eigen_spectrum()` restituisce tutti gli autovalori in ordine decrescente. Non elimina i piccoli autovalori negativi, così eventuali problemi numerici o di positività restano visibili.

### 4. Effective rank

L'effective rank riassume quanto lo spettro è distribuito. Si basa sull'entropia degli autovalori normalizzati:

$$
p_i=\frac{\lambda_i}{\sum_j \lambda_j},
\qquad
r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right)
$$

Un valore vicino a 1 indica che poche direzioni dominano; un valore più alto indica una distribuzione più uniforme degli autovalori.

Nel codice gli autovalori negativi compatibili con piccoli errori numerici vengono trattati come zero. Se ci sono autovalori negativi significativi, la funzione segnala che la matrice non è PSD. Per una matrice nulla restituisce 0.

### 5. Kernel-target alignment

L'alignment misura quanto le similarità rappresentate dal kernel sono coerenti con le etichette di classe.

Il codice costruisce una matrice target `T`:

$$
T_{ij} =
\begin{cases}
1 & \text{se } y_i=y_j\\
0 & \text{se } y_i\ne y_j
\end{cases}
$$

Poi centra `K` e `T` e calcola la centered kernel-target alignment:

$$
\operatorname{CKTA}(K,T)=
\frac{\langle K_c,T_c\rangle_F}
{\|K_c\|_F\,\|T_c\|_F}
$$

Valori più alti indicano una maggiore coerenza tra il kernel e le classi. La funzione è pensata per classificazione, anche multiclass; non per target continui di regressione.

## API e utilizzo

### `diagnostics/concentration.py`

| Funzione                       | Input           | Output   | Descrizione                                      |
| ------------------------------ | --------------- | -------- | ------------------------------------------------ |
| `off_diagonal_values(K)`       | Matrice `N × N` | Array 1D | Estrae i valori fuori diagonale senza duplicati  |
| `off_diagonal_variance(K)`     | Matrice `N × N` | `float`  | Calcola la varianza dei valori fuori diagonale   |
| `eigen_spectrum(K)`            | Matrice `N × N` | Array 1D | Restituisce gli autovalori in ordine decrescente |
| `effective_rank(K, tol=1e-10)` | Matrice `N × N` | `float`  | Calcola l'effective rank spettrale               |

### `diagnostics/alignment.py`

| Funzione                        | Input                                      | Output  | Descrizione                                 |
| ------------------------------- | ------------------------------------------ | ------- | ------------------------------------------- |
| `kernel_target_alignment(K, y)` | Matrice `N × N` e vettore di `N` etichette | `float` | Calcola la centered kernel-target alignment |

### Esempio

Eseguire dalla cartella principale del repository, dove è presente il package `diagnostics`:

```python
import numpy as np

from diagnostics.concentration import (
    off_diagonal_values,
    off_diagonal_variance,
    eigen_spectrum,
    effective_rank,
)
from diagnostics.alignment import kernel_target_alignment

# Kernel sintetico: due gruppi con similarità interna alta
# e similarità tra gruppi bassa.
K = np.array([
    [1.0, 0.9, 0.1, 0.1],
    [0.9, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 0.9],
    [0.1, 0.1, 0.9, 1.0],
])

# Etichette delle due classi
y = np.array([0, 0, 1, 1])

print("Valori fuori diagonale:", off_diagonal_values(K))
print("Varianza:", off_diagonal_variance(K))
print("Autovalori:", eigen_spectrum(K))
print("Effective rank:", effective_rank(K))
print("Kernel-target alignment:", kernel_target_alignment(K, y))
```

## Requisiti e controlli

- Dipendenza esterna: `numpy`.
- `K` deve essere una matrice quadrata, finita e simmetrica.
- Per `effective_rank()`, `K` deve essere positiva semidefinita (PSD), a meno di piccoli errori numerici.
- Per `kernel_target_alignment()`, `y` deve essere un vettore monodimensionale con una label per ogni riga/colonna di `K` e almeno due classi.
- Le funzioni sollevano `ValueError` quando gli input non rispettano i controlli previsti.

Installazione della dipendenza:

```bash
python -m pip install numpy
```

## Test

I test sintetici e i test sugli input non validi possono essere eseguiti dalla root del repository:

```bash
python test.py
```

Il test sintetico controlla i risultati principali e l'invarianza rispetto alla scala positiva del kernel. I test aggiuntivi controllano, tra gli altri casi, matrici non quadrate, non simmetriche, valori mancanti, kernel non PSD ed etichette con dimensioni errate.

## Limiti e relazione con l'accettazione G-01

Questi moduli forniscono le metriche diagnostiche richieste dalla task. Non generano da soli i kernel fidelity e projected, non scelgono i dati sperimentali e non producono automaticamente il confronto della Figura 3.

La verifica del criterio di accettazione — riprodurre il trend sintetico della Figura 3 della proposta, con varianza fidelity molto inferiore a quella projected a `n=12` — richiede che il codice venga eseguito sulle matrici kernel e sulle condizioni sperimentali appropriate. Questa documentazione non dichiara tale criterio già verificato.
