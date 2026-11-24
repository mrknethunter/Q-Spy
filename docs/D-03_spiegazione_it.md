# D-03 — Trasformazioni delle feature senza leakage

## Feature e ordine

La proposta ufficiale, sezione 4.2, indica sette quantità Omicron, tutte
trasformate con `log10`: SNR, frequenza di picco, frequenza centrale,
bandwidth, durata, Q-value e ampiezza. L’ordine è mantenuto in
`FEATURE_COLUMNS` (`snr`, `peak_frequency`, `central_frequency`, `bandwidth`,
`duration`, `q_value`, `amplitude`). La proposta non definisce i nomi delle
colonne del DataFrame: questi nomi `snake_case` sono provvisori e vanno
confrontati con lo schema prodotto da D-02 quando disponibile.

`make_features(df, scaler=None)` seleziona esclusivamente le sette colonne,
ignorando metadati e colonne aggiuntive, e mantiene l’ordine delle righe. La
funzione restituisce una matrice NumPy `float64` di forma `(N, 7)` e lo scaler
usato. Non modifica identificativi e non elimina righe.

## Pipeline matematica

Per la feature `j`, dopo la trasformazione logaritmica si calcolano mediana e
intervallo interquartile sui dati di fit:

```text
u_j = log10(x_j)
z_j = (u_j - median_j) / IQR_j
X_tilde_j = clip(z_j, -1, 1)
```

Il componente usa `sklearn.preprocessing.RobustScaler` con quantili 25 e 75,
poi clipping netto a `[-1, 1]`. Non applica il bandwidth quantistico `s` né
`theta = s * pi * X_tilde`: questi appartengono al componente quantistico.
Non applica `tanh` e non apprende parametri di compressione adattivi.

Per un feature costante o con IQR nullo, `RobustScaler` usa una scala pari a 1
per evitare divisioni per zero; la feature costante del training diventa zero
dopo il centraggio. Eventuali valori diversi in validation/test vengono
trasformati con la stessa scala e poi limitati dal clipping.

## Fit senza leakage

Senza scaler, la funzione adatta mediana e IQR sulle righe ricevute. Il
chiamante deve passare solo il training set. Per validation e test deve
riutilizzare l’istanza restituita:

```python
X_train, scaler = make_features(train_df)
X_validation, _ = make_features(validation_df, scaler=scaler)
X_test, _ = make_features(test_df, scaler=scaler)
```

Nel ramo di trasformazione non viene eseguito alcun fit; l’istanza non viene
modificata. Il test anti-leakage confronta le statistiche di training con
validation differenti, ne controlla l’invarianza dopo transform e verifica che
un fit deliberatamente contaminato sul train più validation dia statistiche
diverse.

## Errori e limiti

La funzione rifiuta DataFrame vuoti, colonne richieste mancanti, valori non
convertibili a `float64`, NaN, infiniti e valori non positivi. Non aggiunge
offset o pseudocount. Rifiuta scaler non adattati, con numero di feature
diverso da sette o, quando lo scaler conserva i nomi, con nomi/ordine diversi.
Lo scaler fornito deve inoltre avere `quantile_range=(25.0, 75.0)`,
`with_centering=True`, `with_scaling=True` e `unit_variance=False`; configurazioni
diverse vengono rifiutate. Un risultato numerico non finito dopo il fit o
transform produce un errore.

La proposta e il repository non includono ancora il contratto completo L-02
né lo schema effettivo D-02 in questo checkout: nomi delle colonne e
compatibilità contrattuale richiedono conferma all’integrazione. Non è presente
un manifest delle dipendenze. Il Python di sistema non include scikit-learn e
pytest; per verificare i test sono stati installati temporaneamente sotto
`/tmp`, senza modificare le dipendenze del progetto.

La compressione `tanh` adattiva è solo un’alternativa sperimentale non
implementata; potrà essere valutata separatamente con un’ablation, senza essere
considerata una funzionalità disponibile.

## Test

`tests/test_transforms.py` copre shape, dtype, intervallo, ordine, formula di
log e scaling, clipping, riuso dello scaler e anti-leakage, costanza/IQR nullo,
input vuoto, valori non validi, colonne mancanti/addizionali e scaler
incompatibile o non adattato. Esecuzione completa: `python3 -m pytest -q`, 19
test superati. Non esiste un test suite contrattuale L-02 nel checkout e
`docs/contracts.md` non è presente, pertanto la compatibilità L-02 non è stata
verificata. I casi dello scaler coprono separatamente quantili, centratura,
scaling e `unit_variance`.
