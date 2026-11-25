# Guida locale E-03

Questa cartella contiene la revisione locale E-03 sul branch già esistente
`E-03-task`. Il lavoro non esegue operazioni Git: niente creazione o cambio di
branch, staging, commit, push o apertura di PR. Le vecchie istruzioni di upload
GitHub non si applicano a questa consegna locale.

## Metodo e limiti

Leggere [E03.md](E03.md) per protocollo P2, preprocessing fold-local, budget,
soglie, metriche, intervalli bootstrap e comandi. Il report
[E03_results.md](E03_results.md) conserva i numeri della precedente esecuzione
solo come storico: quella esecuzione riutilizzava lo scaler esterno nei fold
CV e gli artefatti necessari alla verifica non sono presenti in questo
checkout. Non usare quei valori come evidenza di accettazione.

La nuova implementazione produce intervalli bootstrap per trigger separati
dall'incertezza tra seed. Il dataset curato D-02 e lo stack Python/Qiskit
necessario non sono inclusi in questo checkout, quindi qui non è stata eseguita
una nuova analisi Gravity Spy. I test helper sintetici effettivamente eseguiti
e il blocco di raccolta della suite d'integrazione sono registrati in
[E03_validation.json](E03_validation.json).

## Artefatti attesi dopo un'esecuzione reale

Una nuova esecuzione completa, in una directory risultati vuota e con input
D-02 già autorizzato, produce predizioni, ID selezionati, audit, tabelle
metriche e grafici in `results/e03/`. Gli intervalli per trigger sono in
`e03_event_bootstrap.csv` e `e03_event_paired_differences.csv`. Il report
indipendente ricalcola le metriche e gli intervalli dagli score salvati.
