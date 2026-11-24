# G-02 — Geometric difference

## Obiettivo

`diagnostics/geometric_difference.py` confronta due matrici kernel Gram già
calcolate: `K_C`, classica, e `K_Q`, quantistica. Il modulo non costruisce i
kernel, non dipende da M-01 e non accede a dati o circuiti quantistici.

## Riferimento e definizione

H.-Y. Huang, R. Kueng e J. Preskill, “Power of data in quantum machine
learning”, *Nature Communications* 12, 2631 (2021), equazione (5),
[doi:10.1038/s41467-021-22539-9](https://doi.org/10.1038/s41467-021-22539-9).
Nella notazione di questa API:
$$

g(K_C\Vert K_Q)=\sqrt{\left\|K_Q^{1/2}K_C^{-1}K_Q^{1/2}\right\|_\infty}.
$$

L'ordine è asimmetrico: `K_C` è il kernel candidato classico e `K_Q` quello
quantistico. Il risultato misura la separazione geometrica tra le similarità
dei due modelli e non è una misura di accuratezza o di vantaggio empirico.

Il riferimento assume `Tr(K_C) = Tr(K_Q) = N`. L'implementazione controlla
questa convenzione e emette `RuntimeWarning` se una traccia differisce da `N`
(tolleranze: `rtol=1e-7`, `atol=1e-10`). Non scala né modifica i kernel.
Questa scelta è rilevante: una scala diversa cambia gli autovalori e quindi il
significato di un ridge assoluto `lambda`.

## Regolarizzazione ridge

Si regolarizza soltanto il kernel classico:
$$
K_{C,\lambda}=K_C+\lambda I,\qquad
g_\lambda=\sqrt{\left\|K_Q^{1/2}K_{C,\lambda}^{-1}
K_Q^{1/2}\right\|_\infty}.
$$

`lambda` è assoluto e non negativo, nelle unità degli elementi kernel.
`lambda=0` richiede che `K_C` sia numericamente definita positiva;
`lambda>0` consente di usare `K_C` PSD singolare. Anche per kernel identici,
un ridge positivo in generale dà un valore diverso da 1. Se gli autovalori
del kernel cambiano molto di scala, scegliere una griglia coerente con tali
autovalori o costruire una griglia relativa esplicitamente; l'API non seleziona
automaticamente un lambda ottimale.

## Algoritmo numerico e tolleranze

1. Converte gli input reali in `float64` e controlla forma quadrata, dimensioni
   uguali, finitezza e simmetria.
2. Accetta la simmetria entro `rtol=1e-7`, `atol=1e-10`; solo dopo questo
   controllo rimuove l'asimmetria di arrotondamento con `(A + A.T) / 2`.
3. Controlla la PSD con gli autovalori simmetrici; la tolleranza è
   `1e-12 + 1e-10 * max(abs(eigenvalues))`. Autovalori negativi più grandi
   causano `ValueError`; quelli compatibili con arrotondamenti vengono
   troncati a zero solo nella costruzione della radice PSD di `K_Q`.
4. Calcola `K_Q^(1/2)` mediante `numpy.linalg.eigh`.
5. Aggiunge `lambda` alla diagonale di una copia di `K_C`, quindi usa
   `numpy.linalg.solve` per risolvere il sistema. Non usa l'inversa esplicita.
6. Valuta `K_Q^(1/2) @ solve(K_C + lambda I, K_Q^(1/2))`, simmetrizza il
   prodotto per il solo rumore del solve, e calcola la norma spettrale con
   `eigvalsh`. La radice quadrata finale produce `g`.

Fallimenti di un solve e risultati non finiti non vengono sostituiti con
valori inventati. Gli input strutturalmente non validi e lambda negativo
generano un errore.

## API pubblica

```python
geometric_difference(K_C, K_Q, regularization=0.0) -> float
geometric_difference_sweep(K_C, K_Q, regularizations=None) -> dict
```

Esempio:

```python
import numpy as np
from diagnostics.geometric_difference import (
    geometric_difference,
    geometric_difference_sweep,
)

K_C = np.array([[1.0, 0.2], [0.2, 1.0]])
K_Q = np.array([[1.0, 0.1], [0.1, 1.0]])
g = geometric_difference(K_C, K_Q, regularization=1e-6)
sweep = geometric_difference_sweep(K_C, K_Q)
```

## Regularisation sweep

La griglia iniziale è assoluta e log-spaziata: `1e-10, 1e-9, ..., 1e-2`.
È la proposta implementativa della task, non un valore scientificamente
ottimale universale. È stata scelta per esaminare più ordini di grandezza; va
interpretata insieme alla scala dello spettro dei kernel. Il ridge positivo
sposta gli autovalori di `K_C` verso l'alto e riduce il condizionamento.

Per matrici PSD, `g_lambda` è matematicamente non crescente al crescere di
lambda, anche quando `K_C` e `K_Q` non commutano: l'inversa
`(K_C + lambda I)^(-1)` è monotona decrescente nell'ordine di Loewner e il
sandwich con `K_Q^(1/2)` conserva tale ordine. La norma spettrale di queste
matrici PSD è il loro massimo autovalore, anch'esso monotono.

La funzione restituisce liste allineate, preservando ordine e duplicati. I
lambda convertibili in float finiti sono memorizzati come float; i valori
finiti negativi restano nel risultato ma hanno `valid=False`. Input non
convertibili, `NaN` e `+/-Infinity` diventano `None` in `regularizations`.
Le righe non valide hanno `values=None`, un messaggio leggibile in `errors` e
`condition_numbers=None` quando la condizione non è calcolabile. Il risultato
non contiene valori non finiti ed è serializzabile con JSON rigoroso, per
esempio `json.dumps(result, allow_nan=False)`.

All'inizio dello sweep i kernel sono validati una volta, viene emesso al più un
warning sulla convenzione di traccia e viene calcolata una sola volta la radice
PSD di `K_Q`. Ogni lambda riutilizza quella radice attraverso un helper privato,
senza ripetere validazione o warning.

| Chiave | Contenuto |
|---|---|
| `regularizations` | lambda finiti richiesti come float; input non convertibili o non finiti diventano `None` |
| `values` | `g_lambda`, oppure `None` se il calcolo fallisce |
| `valid` | booleano per ogni calcolo |
| `errors` | `None` o messaggio del problema per riga |
| `condition_numbers` | condizionamento spettrale di `K_C + lambda I`, oppure `None` |

Una griglia vuota è rifiutata. I valori di regolarizzazione negativi, se finiti e convertibili in float, vengono mantenuti nel risultato con stato non valido; i valori non finiti o non convertibili vengono rappresentati come None.
Kernel condivisi non validi causano invece `ValueError` prima del ciclo. Il dizionario usa liste e valori JSON-friendly, così E-02 può associare ogni misura al proprio `k` e serializzare i risultati.

```python
result = geometric_difference_sweep(K_C, K_Q, [0.0, 1e-6, 1e-3])
for lam, value, valid in zip(
    result["regularizations"], result["values"], result["valid"]
):
    print(lam, value, valid)
```

## Test

`tests/test_geometric_difference.py` include identità, soluzione analitica
diagonale, non negatività, asimmetria, formula regolarizzata, matrice singolare
e quasi singolare, sweep e condizionamento, avviso sulla traccia, input non
validi e riproducibilità. Non essendo presenti nella checkout `docs/contracts.md`
né `tests/fixtures/synthetic.py` di L-02, i test G-02 usano matrici PSD
sintetiche costruite indipendentemente. I test effettivamente eseguiti e i
relativi esiti sono riportati nel report di completamento.

## Limitazioni e compatibilità

- Il confronto richiede matrici Gram complete, quadrate e con lo stesso numero
  di esempi; input train/test rettangolari non sono supportati.
- Non viene effettuata una normalizzazione automatica di traccia o scala.
- La griglia predefinita è assoluta, non relativa agli autovalori.
- La radice PSD e il solve hanno costo cubico nella dimensione del kernel e
  richiedono memoria quadratica.
- Non è disponibile una fixture condivisa L-02 in questa checkout.

La firma principale accetta direttamente le matrici prodotte in futuro da
`projected_kernel(B1, B2, gamma)` e `rbf_kernel(X1, X2, gamma)` di M-01.
E-02 può consumare il float singolo o le righe allineate dello sweep senza
dipendenze inverse. Il modulo usa solo NumPy, già adottato dalle diagnostiche
presenti.
