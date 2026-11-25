## Problema e risultato della modifica

Questa PR rende nuovamente eseguibile il codice E-03 e completa i controlli necessari per valutare il replay di **Fast_Scattering** senza contaminare training, tuning o calibrazione con la famiglia sconosciuta.

Nella copia ricevuta i due runner terminavano a metà istruzione, il preprocessing richiedeva una gestione specifica dei fold interni e la verifica dei risultati non ricalcolava la soglia dai punteggi di validation salvati. Inoltre il manifest storico dichiarava completo un insieme di risultati privo di nove artefatti necessari alla verifica indipendente.

La PR include sia il lavoro E-03 già presente nella copia iniziale sia le correzioni della revisione. **Stato: IMPLEMENTED — NOT VALIDATED.** Non contiene una nuova esecuzione completa sui dati Gravity Spy e non propone i risultati storici come validazione del codice corretto.

Riferimento: #29. La task resta aperta finché non saranno completate esecuzione e validazione scientifica.

## Cosa cambia, e perché

### 1. Runner completo e scritture limitate a E-03

- Completa le parti troncate di `e03_novelty.py` e `e03_parallel.py`, ripristinando serializzazione degli output, ritorno dei risultati e gestione degli errori.
- Scrive i manifest in modo atomico: un'interruzione non deve lasciare un JSON scritto solo in parte.
- Usa l'argomento supportato `root` di `write_bloch` per salvare le cache sotto la directory di output E-03, evitando scritture nella componente condivisa `store/`.
- Conserva le cache dei worker anche nell'aggregato finale e registra gli hash degli output.

### 2. Preprocessing indipendente per ogni fold interno

- Aggiunge l'adapter locale `e03_preprocessing.py`, mantenendo modelli, generazione dei candidati e funzione di scoring di M-04.
- Ogni scaler viene stimato esclusivamente sul training del proprio fold interno; validation e relativo embedding quantistico usano quello stesso scaler.
- Dopo la selezione, il modello finale usa lo scaler stimato sul training esterno selezionato.
- Registra appartenenza agli ID di training, hash delle feature grezze, centro e scala di ogni fit. Un fit con righe non registrate viene respinto prima di chiamare la trasformazione condivisa.

### 3. Metriche e due fonti di incertezza tenute separate

- Raccoglie in `e03_metrics.py` AUROC, **Average Precision** e TPR massimo fra i punti ROC effettivamente raggiunti con FPR <= 1%, senza interpolazione della metrica.
- Mantiene i CI appaiati fra seed, condizionati al test fissato.
- Aggiunge intervalli da bootstrap stratificato degli eventi, con gli stessi indici ricampionati per tutti i modelli e per le differenze projected-minus-classical.
- Corregge l'uso del limite FPR nel valore puntuale del bootstrap e rende esplicito il modello proiettato di riferimento: cambiare l'ordine del dizionario non cambia il segno del confronto.
- Un solo seed non viene presentato come una stima dell'intervallo fra seed. Le rotazioni sovrapposte non vengono trattate come dataset indipendenti.

### 4. Soglie e verifica indipendente più rigorose

- Calibra la soglia sul percentile 99 dei soli score di validation nota, usando la convenzione conservativa `higher` e la decisione stretta `score > threshold`.
- Salva gli score di validation oltre alle predizioni di test. Il verificatore ricalcola soglia e FPR dagli score salvati, invece di fidarsi soltanto dei valori riportati nell'audit.
- Introduce `e03_integrity.py` per richiedere tutti gli artefatti obbligatori, controllare hash e percorsi, e respingere casi duplicati o mancanti anche quando un dizionario ne nasconderebbe la duplicazione.
- Controlla copertura completa delle impronte dei sorgenti, appartenenza degli ID, classi sconosciute dichiarate nel manifest D-04, scaler dei fold, ID esterni comuni ai quattro modelli, numero di candidati e conteggi noti/sconosciuti.
- Conserva la ricalcolazione di metriche e intervalli dalle predizioni; un'esecuzione parziale o incompatibile non viene accettata come risultato completo.

### 5. Parallelismo e ripresa controllata

- Mantiene worker separati per famiglia e limita i thread delle librerie numeriche nei processi.
- Aggiunge `resume=True`, legato alla stessa identità di dataset, split, sorgenti, software, seed, budget, circuito e impostazioni.
- Riutilizza soltanto worker completi, compatibili e verificati tramite hash.
- Conserva i tentativi incompleti in `failed_attempts/` prima di rieseguirli; un worker completo ma corrotto viene respinto.
- Verifica tutti i worker prima di aggregare e conserva i dati di staging fino al superamento dei controlli di integrità sull'aggregato.

### 6. Diagnostica leggibile e documentazione coerente con le prove disponibili

- Aggiunge `missed_fast_events.csv`: ID dei Fast_Scattering mancati per seed/modello, detector, run, score, soglia, vicino noto e distanza nelle sette feature trasformate.
- Mantiene i conteggi aggregati degli errori normalizzati per supporto e rende dinamiche le tabelle per detector e observing run.
- Esplicita i conteggi di vicinato per Tomte, Scattered_Light, Low_Frequency_Burst e Low_Frequency_Lines, senza assumerli come conclusioni fisiche predeterminate.
- Elimina dal template conteggi Gravity Spy codificati a mano e un riferimento scientifico locale non disponibile. Distingue misure, interpretazioni supportate dalla fonte locale e ipotesi causali da validare.
- Aggiorna audit, metodo, rapporto e ricevuta di validazione. La vecchia ricevuta viene conservata; le tabelle precedenti restano qualificate come risultati storici non rivalidati.

## Protocollo e configurazione

Il confronto resta fra projected OC-SVM, RBF OC-SVM, Isolation Forest e LOF in modalità novelty, con score più alto = più anomalo.

- P2 D-04 esistente: **Fast_Scattering sempre esclusa da training e validation**, anche quando ruota un'altra famiglia. Le classi sconosciute vengono lette dal manifest.
- Otto rotazioni, seed 0–9, otto candidati per modello e tre fold interni. Stessi ID esterni per tutti i modelli; `target_family` e `protocol_unknown` restano separati.
- L0 esatto: 8 qubit, 2 layer, CZ e `s=0.1778`, coerenti con il tag locale `hw-freeze`. Bandwidth applicata una sola volta nell'embedding.
- Kernel proiettato contrattuale implementato tramite RBF su `B_flat / sqrt(n)`.
- Cap espliciti: 2048 training, 2048 validation e 8192 test. Non si presenta questa configurazione come esperimento senza limiti di campionamento.
- Il YAML include il budget di bootstrap sugli eventi. Uguale numero di configurazioni non viene descritto come uguale costo computazionale.

## File principali

| Area | File |
| --- | --- |
| Esecuzione e ripresa | `experiments/e03_novelty.py`, `experiments/e03_parallel.py` |
| Preprocessing, metriche e integrità | `experiments/e03_preprocessing.py`, `experiments/e03_metrics.py`, `experiments/e03_integrity.py` |
| Verifica e report | `experiments/e03_report.py` |
| Configurazione | `experiments/rq4/e03_gravityspy.yaml` |
| Test | `tests/test_e03.py`, `tests/test_e03_preprocessing.py`, `tests/test_e03_metrics.py`, `tests/test_e03_integrity.py`, `tests/test_e03_review.py` |
| Documentazione | `docs/E03.md`, `docs/E03_IMPLEMENTATION_AUDIT.md`, `docs/E03_REVIEW_REPORT.md`, `docs/E03_results.md`, `docs/E03_validation.json` |
| Ricevute della revisione | `results/e03/review_20261125/` |

## Verifiche realmente eseguite

- **18 test stdlib passati**, zero fallimenti e zero skip: sintassi dei sorgenti/test E-03, output mancanti o alterati, casi duplicati, manifest parziali, identità di ripresa, scrittura atomica e soglie con campioni piccoli, ties, score costanti e valori non finiti.
- **80 casi P2 completi controllati in sola lettura**, sul catalogo di 318.801 righe: ID unici e partizioni disgiunte; esclusione delle classi sconosciute da train/validation; tutte le classi sconosciute nel test; test fisso fra seed della stessa rotazione.
- Il controllo delle partizioni usa il runtime bundled NumPy 2.3.5/pandas 3.0.1: è una verifica di appartenenza, non una validazione dei fit nell'ambiente pinned.
- `git diff --check` superato nella preparazione della PR.

La suite scientifica è stata tentata ma **si ferma prima della raccolta dei test** perché manca pytest nel runtime disponibile; mancano anche scikit-learn e Qiskit/Aer. I test scientifici presenti e i cinque nuovi test di regressione sono quindi **scritti ma non eseguiti**. Nessun pacchetto è stato installato e nessuna dipendenza è stata cambiata.

## Cosa resta da validare

Servono il runtime pinned e i dati grezzi/curati Gravity Spy per eseguire le suite scientifiche, l'integrazione sintetica completa, il controllo effettivo di batching/kernel e i nuovi test di resume. Successivamente va eseguito il confronto completo di 80 casi in una nuova directory, seguito dalla ricalcolazione indipendente di metriche, soglie e CI.

Il manifest storico elenca 18 artefatti: nove presenti conservano gli hash, ma mancano otto archivi di predizioni e `selected_ids.json.gz`. I risultati storici non permettono quindi la rivalidazione numerica. Non vengono generati dati sostitutivi né avanzate affermazioni di vantaggio quantistico.

Le modifiche sono limitate a file E-03. Componenti condivise, split, contratti, dichiarazioni delle dipendenze e configurazione CI restano invariati. I rapporti di audit descrivono la fase locale precedente alla pubblicazione; il successivo commit e l'apertura della PR sono stati richiesti esplicitamente dall'utente.
