# D03 — Spiegazione dell’implementazione

## 1. A cosa serve D03

D03 prepara i dati numerici dei glitch prima che vengano utilizzati dagli esperimenti di Q-Spy. Ogni riga rappresenta un trigger e contiene sette misure provenienti dai parametri Omicron. Queste misure possono avere scale molto diverse: per esempio, una frequenza e un’ampiezza non hanno necessariamente valori confrontabili. Il compito trasforma quindi i dati in una matrice ordinata e numericamente più adatta ai modelli del progetto.

## 2. Quali dati vengono trasformati

La proposta del progetto elenca sette feature: SNR, frequenza di picco, frequenza centrale, bandwidth, durata, Q-value e ampiezza. Nel codice ho usato nomi provvisori in formato `snake_case`: `snr`, `peak_frequency`, `central_frequency`, `bandwidth`, `duration`, `q_value` e `amplitude`.

Questi nomi sono raccolti nella costante `FEATURE_COLUMNS` all’inizio di `features/transforms.py`. Tenerli in un solo punto rende semplice adattare il codice quando D02 stabilirà i nomi definitivi delle colonne. La funzione ignora le altre colonne del DataFrame, come ID, label, detector e tempo GPS, perché non sono le sette misure da passare come input al modello.

## 3. La funzione principale

La funzione pubblica è `make_features(df, scaler=None)`. Riceve un DataFrame e, facoltativamente, uno scaler già addestrato. Restituisce una coppia: la matrice trasformata `X_tilde` e lo scaler utilizzato.

La matrice restituita ha forma `(N, 7)`, dove `N` è il numero di righe ricevute, ed è di tipo `float64`. La funzione controlla che l’input sia un DataFrame, che contenga tutte le colonne richieste e che abbia almeno una riga. Controlla inoltre che i valori siano finiti e strettamente positivi, perché il logaritmo in base 10 non è definito per zero o numeri negativi.

## 4. Perché si applica il logaritmo

Le sette misure possono avere distribuzioni molto sbilanciate e code lunghe: pochi trigger possono avere valori molto più grandi della maggior parte degli altri. Applicare `log10` comprime le differenze di scala e riduce l’influenza dei valori estremi. Questa trasformazione è indicata nella proposta del progetto per tutte le sette feature.

Il logaritmo viene applicato prima dello scaler. Per questo, i valori originali devono essere positivi. Se D02 produce valori pari a zero o negativi per una delle feature, sarà necessario confrontarsi con la specifica e decidere come gestirli; il codice attuale segnala l’errore invece di correggere i dati silenziosamente.

## 5. Come funziona lo scaler robusto

Dopo il logaritmo, il codice usa `RobustScaler` di scikit-learn con intervallo interquartile dal 25° al 75° percentile. In pratica, ogni feature viene centrata usando la mediana e ridimensionata usando l’IQR. Queste statistiche sono meno sensibili ai valori estremi rispetto a media e deviazione standard.

Se `scaler` non viene passato, la funzione crea un nuovo `RobustScaler` e lo adatta ai dati ricevuti. Lo scaler conserva le statistiche apprese, che vengono restituite al chiamante insieme alla matrice. Se invece viene passato uno scaler già adattato, la funzione lo usa per trasformare i dati senza ricalcolare le statistiche.

## 6. Come si evita il data leakage

Il data leakage si verifica quando informazioni provenienti da validation o test influenzano il processo di preparazione dei dati usato per il training. Per evitarlo, lo scaler deve essere adattato solo sulle righe di training. Il codice non può sapere autonomamente quale partizione contiene: è responsabilità di chi chiama la funzione passare i dati corretti.

Il flusso previsto è: chiamare `make_features` sul training set senza scaler, conservare lo scaler restituito e riutilizzarlo per trasformare validation e test. In questo modo le statistiche restano quelle del training. Se si chiama la funzione senza scaler su validation o test, la funzione adatterà un nuovo scaler su quei dati e il flusso non rispetterà il requisito di assenza di leakage.

Esempio d’uso:

```python
X_train, scaler = make_features(train_df)
X_validation, _ = make_features(validation_df, scaler=scaler)
X_test, _ = make_features(test_df, scaler=scaler)
```

## 7. Perché si usa `tanh`

La robust scaling da sola non garantisce che i valori siano compresi tra `-1` e `1`. La task richiede quell’intervallo, ma non specifica la formula per ottenerlo. In questa implementazione ho scelto `tanh`, una funzione che porta i valori trasformati dentro `[-1, 1]` e comprime gradualmente quelli molto grandi in valore assoluto.

Questa scelta è quindi una decisione implementativa provvisoria, non un requisito esplicito del progetto. È isolata alla fine della funzione, quindi si potrà sostituire con un’altra formula qualora il team concordi una convenzione diversa.

## 8. Controlli sugli input e messaggi d’errore

La funzione interrompe l’elaborazione con un messaggio chiaro se mancano colonne, se il DataFrame è vuoto, se ci sono valori non finiti o non positivi, oppure se lo scaler passato non è uno `RobustScaler` già adattato. Questi controlli aiutano a individuare problemi nello schema o nei dati vicino al punto in cui si presentano, invece di lasciare che producano risultati numerici non validi più avanti nella pipeline.

## 9. File creati

- `features/transforms.py` contiene la costante delle colonne e la funzione `make_features`.
- `features/__init__.py` rende la funzione e la costante importabili dal pacchetto `features`.
- `tests/test_transforms.py` contiene test per forma, tipo e intervallo dell’output, riuso dello scaler, dati sintetici e input non validi.
- `README.md` descrive il contratto e mostra come riutilizzare lo scaler tra le partizioni.
- Questo documento spiega in italiano le scelte e il funzionamento del codice.

## 10. Stato e punti da confermare

L’implementazione segue la firma e le caratteristiche dell’output indicate da L02 e i criteri riportati nella task D03. Non dipende dal codice D02: usa dati sintetici per i test e nomi provvisori per le colonne.

Restano da verificare quando saranno disponibili i contratti completi e D02: i nomi reali delle colonne, l’eventuale gestione di valori non positivi e la scelta condivisa della funzione di compressione in `[-1, 1]`. I test sono stati eseguiti nell’ambiente virtuale D03: tutti e 7 passano.
