# E-03 — risposta alla review sulla selezione degli iperparametri

## Fonte e correzione dell'interpretazione precedente

Questo documento risponde alla review fornita direttamente dall'utente il
25 novembre 2026, usando la proposta, i contratti, le specifiche M-03/M-04 e
`docs/decisions.md §4` come contesto. Non attribuisce questa review a commenti
GitHub: al momento della lettura la PR #74 non ne conteneva.

La precedente scelta di conservare in E-03 l'obiettivo M-04 basato solo sulla
vicinanza del FPR a nu era insufficiente per H3. Registrare meglio quel criterio
non ne corregge il difetto scientifico. La review ne richiede la sostituzione
per il confronto primario. Tale richiesta autorizza esplicitamente la modifica
condivisa, supersedendo il precedente divieto limitatamente a questa correzione.

I dati storici confermano i numeri citati: **156/160** selezioni OC-SVM hanno
margine tra i primi due punteggi di calibrazione inferiore a 0,01; AUROC media
storica RBF/Fast_Scattering **0,3642583**, proiettata/Extremely_Loud **0,1918007**.
Sono controlli sulle tabelle precedenti, non nuova validazione delle predizioni.
Un piccolo margine non è da solo una dimostrazione matematica di casualità,
ma conferma che quel criterio non misura la separazione richiesta da H3.

## Risposta puntuale

| Rilievo | Correzione |
| --- | --- |
| FPR vicino a nu usato per selezionare gamma | `fit(X, family_labels)` passa al selettore condiviso `fit_surrogate_loco`; gamma massimizza l'AUROC media di classi note lasciate fuori come anomalie surrogate. |
| Gamma assoluta non coerente con la scala della rappresentazione | Nove fattori `c=logspace(-2,2,9)`; ogni estimatore calcola `gamma=c/median(pdist(X_fit, squared))` sul proprio training già trasformato. |
| Nu deve usare un target noto del 5% | Per ciascun candidato gamma, nu viene calibrata sui soli noti interni prima di leggere gli score delle anomalie surrogate. Sono salvati trial, FPR misurato e scelta. La griglia discreta conserva i cinque valori M-04; l'errore di calibrazione finito resta visibile. |
| Fast_Scattering mai nel tuning | Esclusione completa D-04 prima del campionamento; il selettore condiviso rifiuta anche direttamente questa etichetta. Il rotated outer unknown è escluso dal chiamante E-03 insieme a Fast_Scattering. |
| Correzione una volta sola nel condiviso | Il metodo originale risiede effettivamente in `models/base.py`, non in `models/baselines.py`. Il primo effettua il dispatch; il secondo contiene un solo selettore comune e la risoluzione median-gamma OC-SVM. E-03 non duplica l'obiettivo. |
| Warning da concatenazione di diagnostiche vuote | Il parent concatena solo frame non vuoti, conservando lo schema se tutti sono vuoti. I CI non stimabili con un solo seed usano NaN coerenti, evitando anche il warning di confronto None/NaN. I test trattano `FutureWarning` come errore. |
| Detector M-03 aggiuntivi | Aperta [issue #75](https://github.com/mrknethunter/Q-Spy/issues/75) per KDE, kernel kNN e ricostruzione kernel PCA sulla medesima matrice kernel. Non si sostiene che siano già implementati. |
| Scaler E-04/hardware diverso | Documentato in E03.md: gli embedding per hardware e L0 devono usare lo stesso scaler P2 seed 0, gli stessi ID e gli stessi metadati; un embedding E-03 fold-specifico non è direttamente confrontabile con quello hardware. |

## Procedura operativa

1. Il training esterno contiene esclusivamente famiglie note.
2. Ogni famiglia nota diventa a turno la famiglia surrogata. Tutte le sue righe
   sono escluse dal fit e dalla calibrazione di nu di quel fold.
3. Le altre famiglie vengono suddivise in fit e validation nota tramite tre
   fold deterministici. Ogni scaler usa solo il fit corrispondente; per il
   proiettato l'embedding segue quella trasformazione specifica.
4. Per ogni c, si sceglie nu avvicinando il FPR noto interno al 5%, con nu
   più piccola in caso di parità. La scelta non vede score delle surrogate.
5. Gamma viene scelta dall'AUROC media noto/surrogata sui fold; a parità viene
   preferito c più piccolo. Non si inverte il segno usando il test reale.
6. Una partizione interna del training esterno calibra la nu finale; poi
   estimatore e scaler vengono rifittati sull'intero training selezionato.
7. La validation esterna indipendente calibra la soglia operativa all'1% con
   decisione stretta `score > threshold`. Il 5% per nu e l'1% operativo sono
   due passaggi distinti, non due descrizioni della stessa misura.

Isolation Forest e LOF mantengono le proprie griglie, ma usano il medesimo
obiettivo discriminativo con classi surrogate. Il contratto `fit(X)` senza
etichette resta compatibile e riporta esplicitamente
`unlabelled_calibration_only_not_H3`: non è il percorso usato da E-03 e non
costituisce accettazione di H3.

## Deviazione esplicita sul budget

`docs/decisions.md §4` indica un budget generico di 60 configurazioni. Il nuovo
E-03 valuta **nove configurazioni per ciascun modello**, coprendo tutta la griglia
prescritta di nove fattori gamma. Nu è una calibrazione interna deterministica,
con fit aggiuntivi registrati separatamente. Non si contano ripetizioni dello
stesso c per costruire artificialmente 60 candidati. Questo aggiorna anche il
precedente budget E-03 di otto candidati.

Questa è una deviazione dichiarata, non conformità completa alla
preregistrazione: l'accettazione formale dell'ipotesi primaria H3 richiede la
decisione del lead sul budget. I risultati sono confronti descrittivi con
uguale numero di candidati, non uguale numero di fit o costo computazionale.

## Evidenza e tracciabilità

Sono stati eseguiti **135 test e 29 subtest** nell'ambiente isolato con le
versioni di `requirements.txt`, usando `-W error::FutureWarning`: nessun
fallimento e nessun warning. Sono inclusi modelli condivisi M-03/M-04,
integrazione L0, scalers, confronto di candidati con FPR identico ma AUROC
opposta, gamma mediana, esclusione di Fast_Scattering e ripresa dopo errore.

Le prime esecuzioni hanno rilevato e permesso di correggere una fixture NumPy
che troncava l'etichetta, percorsi temporanei Windows troppo lunghi, duplicati
legittimi della stessa selezione di training nell'audit e il confronto None/NaN
dei CI con un solo seed. I log iniziali sono conservati, insieme a quello
passante, sotto `results/e03/review_followup_20261125/`.

L'utente ha autorizzato l'installazione isolata dopo la review. Gli otto CSV
Gravity Spy sono stati scaricati solo nella cache E-03, verificati con dimensioni
e MD5 del manifest e curati con il codice condiviso invariato. Le 318.801 righe
ottenute corrispondono al fingerprint D-04. Il nuovo replay completo è salvato
separatamente in `results/e03/loco_1125/`; i risultati precedenti restano intatti.
La sua verifica indipendente e lo stato effettivo sono riportati nella ricevuta
aggiornata, senza equiparare l'avvio del run al suo completamento.

La verifica del percorso hardware in `scripts/qpu_campaign.py:90–97` conferma che `gravityspy_features` fitta lo scaler su tutti gli ID di training del protocollo/seed congelato e riusa quello scaler sugli ID hardware. Non è lo scaler del sottoinsieme E-03 né quello di un suo fold interno.

## Istruzione successiva sulla gestione delle issue

L’utente ha successivamente richiesto di non aprire altre issue e di occuparsi personalmente del follow-up. La #75 era già stata creata prima di questa indicazione; non viene modificata né vengono aperte ulteriori issue. Il lavoro corrente resta limitato alla correzione, alla validazione e all’aggiornamento della PR #74.

## Esito conclusivo del replay

Tutti gli 80 casi sono completati (durata registrata 3.632,344 s) e la verifica indipendente ha ricalcolato 3.200 valori metrici, soglie, scaler, selezione degli iperparametri e intervalli. Il controllo dell’aggregato rispetto agli otto worker verificati è passato. Il test del verificatore parallelo è passato in aggiunta ai 135 test/29 subtest della suite principale.

Su Fast_Scattering, AUROC media 0,9583 per RBF contro 0,9312 per proiettata; LOF è migliore per TPR descrittiva all’1% FPR (0,2441). Gli ID selezionati coincidono con quelli storici in tutti gli 80 casi: il salto non dipende da un diverso test o prevalenza. Non è una crescita della quantità di anomalie. L’accettazione preregistrata H3 non viene dichiarata: il budget 9/60 resta una deviazione esplicita.
