## Problema risolto e nuova evidenza

La review ha identificato un difetto scientifico: il vecchio tuning OC-SVM premiava soltanto la vicinanza del FPR a nu, senza valutare la separazione delle anomalie. La nuova selezione condivisa massimizza l’AUROC con classi note usate come anomalie surrogate, seguendo la procedura gamma/nu di `docs/decisions.md §4` con una deviazione esplicita sul budget.

**Il replay reale è ora completato e verificato: 80 casi, quattro modelli, 3.200 valori metrici ricalcolati.** La validazione tecnica non viene equiparata all’accettazione formale preregistrata di H3.

Riferimento: #29, mantenuta aperta per l’accettazione scientifica del progetto.

## Correzione condivisa degli iperparametri

- `models/base.py` inoltra `fit(X, family_labels)` a un solo selettore comune in `models/baselines.py`, riusato dai quattro detector E-03. Il percorso senza etichette resta compatibile ma riporta esplicitamente `unlabelled_calibration_only_not_H3`.
- Ogni famiglia nota viene esclusa a turno dal fit e dalla calibrazione di nu e usata come anomalia surrogata; tre fold interni per famiglia. Fast_Scattering e la famiglia sconosciuta della rotazione non partecipano alla selezione.
- Gamma usa tutti i nove fattori `c=logspace(-2,2,9)`, divisi per la mediana delle distanze al quadrato del rispettivo training trasformato. Per la rappresentazione proiettata resta valida la normalizzazione del kernel `B_flat/sqrt(n)`.
- Nu è scelta separatamente sui soli noti interni verso il target del 5% di falsi allarmi, prima di leggere gli score delle surrogate. Trial, FPR effettivo, gamma risolta e mediana sono registrati; non si nasconde l’errore della calibrazione discreta.
- Il candidato gamma vincente massimizza l’AUROC media delle surrogate; le parità preferiscono gamma più piccola. La soglia operativa all’1% resta calibrata sulla validation esterna indipendente.
- `e03_tuning.py` verifica candidati, fold, supporti, medie, calibrazione e selezione del vincitore dal relativo audit.

**Deviazione dichiarata:** nove candidati per ciascuno dei quattro modelli, contro il budget generico di 60 nella decisione. Si copre l’intera griglia prescritta di nove gamma; la calibrazione interna di nu comporta fit aggiuntivi esplicitamente conteggiati. Uguale numero di candidati non significa uguale costo. Non si dichiara H3 conforme alla preregistrazione senza la decisione del lead su questa deviazione.

## Altre modifiche E-03 incluse

- Completati i runner originariamente troncati; aggiunte scritture atomiche dei manifest, gestione degli errori e cache Bloch limitate alla directory E-03 tramite l’interfaccia condivisa esistente.
- Scaler fit separatamente per ogni training interno e rifit finale sul solo training esterno selezionato; stato e appartenenza degli ID verificabili, anche per l’embedding quantistico.
- Controlli completi di partizioni D-04, ID comuni ai modelli, source/output hash, budget e casi duplicati o mancanti.
- Salvati score di validation e test; soglia percentile 99 con `higher` e decisione stretta `score > threshold`, ricalcolata indipendentemente.
- AUROC, Average Precision e massimo TPR realmente raggiunto con FPR <=1%; distinti dai TPR/FPR ottenuti con soglia di validation.
- Bootstrap appaiato fra seed ed eventi tenuti separati, con riferimento proiettato esplicito e senza aggregare rotazioni sovrapposte come dataset indipendenti.
- Resume solo per worker completi con identità coerente di dati, split, codice, software e parametri; tentativi incompleti conservati. Thread numerici limitati.
- Corretta la concatenazione di diagnostiche vuote e la rappresentazione dei CI non stimabili con un solo seed; nessuna soppressione dei FutureWarning.
- Diagnostica per famiglia/detector/run con denominatori; singoli Fast_Scattering mancati e relativi vicini noti; interpretazioni fisiche distinte dalle misure e supportate dalla fonte locale disponibile.
- Documentata la differenza con E-04: l’hardware usa lo scaler del training P2 seed 0 completo; un confronto L0/hardware deve riallineare lo scaler e i suoi metadati.
- La gestione di ulteriori issue/follow-up è lasciata all’utente come da ultima indicazione; questa PR non implementa i tre detector alternativi M-03.

## Verifiche realmente eseguite

- **135 test + 29 subtest passati**, usando lo stack di `requirements.txt` e `-W error::FutureWarning`, incluse regressioni M-03/M-04, contratti, integrazione L0 e nuovo tuning reale.
- **Un ulteriore test del verificatore parallelo passato**: verifica numerica dei worker, sopravvivenza degli snapshot e rigetto delle predizioni alterate. Uno smoke test di provenienza ripete un test già contato nella suite principale.
- Tutti gli otto CSV Gravity Spy hanno dimensioni e MD5 del manifest; la curation produce 318.801 righe con il fingerprint D-04 atteso. Pacchetti e input sono confinati alle cache locali E-03 escluse da Git, dopo autorizzazione esplicita.
- **80/80 casi e 3.200 metriche verificati indipendentemente** dai worker originali; poi uguaglianza di tutte le tabelle, audit, ID e hash dell’aggregato. Ricalcolati soglie, scaler, vincitori del tuning, CI fra seed ed eventi.
- Gli hash degli ID selezionati corrispondono allo storico in tutti gli 80 casi. I vecchi score mancanti non vengono dichiarati rivalidati.
- Il replay completo ha richiesto 3.632,344 secondi (circa 60,5 minuti): il costo del nuovo tuning annidato è diverso dal precedente.

## Risultati principali Fast_Scattering

Medie sui dieci seed; CI 95% e tutte le rotazioni nel rapporto. Il test è fisso fra seed e comune ai modelli. Questi sono i sottoinsiemi dichiarati di massimo 2048 train / 2048 validation / 8192 test, non un fit senza cap.

| Modello | AUROC | AP | TPR descrittiva @1% FPR |
| --- | --- | --- | --- |
| projected_ocsvm | 0.9312 | 0.9007 | 0.0876 |
| rbf_ocsvm | 0.9583 | 0.9251 | 0.0921 |
| isolation_forest | 0.9336 | 0.8797 | 0.0383 |
| lof | 0.9473 | 0.9300 | 0.2441 |

AUROC 0,958 non significa rilevare il 95,8% delle anomalie alla soglia operativa. RBF ordina meglio gli score sull’intera ROC; LOF rileva di più nella regione all’1% FPR. Non viene affermato un vantaggio quantistico né una superiorità statistica generale.

## Rapporti, dati e riproduzione

- Rapporto corrente e risultati completi: `docs/E03_results_loco.md`.
- Metodo e risposta puntuale alla review: `docs/E03.md`, `docs/E03_REVIEW_RESPONSE.md`.
- Ricevuta corrente: `docs/E03_validation.json`; log e ricevute dei worker sotto `results/e03/review_followup_20261125/`.
- Risultati completi in nove ZIP lossless sotto `results/e03/loco_1125_archives/`, con SHA-256 in `archives_manifest.json`. Contengono tutti i 270 file originali, incluse predizioni, validation score, ID, audit e Bloch, senza conversione dei fine riga. I dati grezzi e l’ambiente non vengono caricati.
- Anteprime ROC pubblicate separatamente in `results/e03/loco_1125_figures/`; i risultati storici e le vecchie ricevute sono conservati e chiaramente separati.

Le sole componenti condivise modificate sono `models/base.py` e `models/baselines.py`, come richiesto dalla review. Split, feature transforms, quantum backend, contratti, dipendenze dichiarate, CI ed esterna documentazione di progetto restano invariati.
