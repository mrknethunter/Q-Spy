> **Aggiornamento dopo review:** la consegna corrente è la PR #74 su `E-03-task-review`. Metodo: E03.md; risultati corretti e verificati: E03_results_loco.md; ricevuta: E03_validation.json. Gli archivi lossless sono in `results/e03/loco_1125_archives/`; estrarre tutti i nove ZIP nella stessa directory nuova. Le istruzioni di caricamento sotto conservano il contesto della prima consegna e non richiedono altre PR o issue.

# E-03 — Consegna su GitHub e riproduzione

## Caricamento dal sito

1. Apri `mrknethunter/Q-Spy`, seleziona il main aggiornato e crea un ramo
   `E-03-novelty-L0` con sorgente `main`.
2. Estrai `Q-Spy-E03-upload.zip`. La cartella estratta contiene direttamente
   quattro cartelle: `experiments`, `tests`, `docs`, `results`.
3. Nel nuovo ramo scegli **Add file → Upload files**. Trascina le quattro
   cartelle dal normale Esplora file. Non trascinare lo ZIP o la cartella esterna.
4. I percorsi devono iniziare da `experiments/e03_novelty.py`,
   `docs/E03.md`, `tests/test_e03.py` e `results/e03/…`.
5. Messaggio commit: `Add E-03 L0 novelty experiment and audited results`.
   Conferma sul ramo E-03, poi apri la PR verso `main`.
6. Titolo: `[E-03] Novelty experiment at L0: Fast_Scattering replay`.
   Descrizione proposta:

```text
Closes #29

Confronta OC-SVM con kernel proiettato L0, OC-SVM RBF, Isolation Forest e LOF
su tutti gli 80 split P2: otto famiglie held-out per dieci seed. Usa le
baseline condivise, budget uguali e controlli espliciti sull'esclusione delle
famiglie held-out da training, validation e scaling.

L'esecuzione è resource-bounded: massimo 2048 righe train, 2048 validation e
8192 test per split, con gli stessi ID per i quattro modelli. Include ROC,
AUROC/AP/TPR@1%FPR, intervalli bootstrap appaiati fra seed e una soglia operativa
calibrata solo su validation nota. Gli intervalli sono condizionati al test
fisso; non costituiscono una prova generale di vantaggio quantistico.

Include report fisico, predizioni, audit delle selezioni e verifica indipendente
che ricalcola metriche e intervalli. Esecuzione esatta L0, senza job hardware.

Metodo: docs/E03.md
Risultati e nota fisica: docs/E03_results.md
Verifiche e provenienza: docs/E03_validation.json
```

I nuovi file non sostituiscono le implementazioni degli altri: la consegna
aggiunge runner/config/test, documentazione e risultati. Dipendenze e split
versionati del main restano invariati. Prima del merge, il team lead deve
valutare anche le scelte sperimentali e i limiti dichiarati nel report.

## Leggere i risultati

- `docs/E03_results.md`: tabella principale Fast_Scattering, rotazioni e nota fisica.
- `results/e03/roc_fast_scattering.png`: ROC principale e dettaglio a basso FPR.
- `results/e03/roc_rotations.png`: otto rotazioni.
- `results/e03/e03_summary.csv`: tutte le medie e i CI, compresi i due scope.
- `results/e03/manifest.json`: stato `complete`, configurazione e hash.

Un alto AUROC non garantisce un'alta TPR al basso FPR richiesto. La TPR sulla
ROC del test è distinta dalla TPR ottenuta con una soglia scelta in anticipo
sulla validation. Una famiglia vicina nello spazio delle feature non equivale
a una classificazione fisica o all'identificazione di un accoppiamento causale.

## Eseguire sul proprio PC

Per caricare o leggere i risultati non serve eseguire codice. Per riprodurli
serve un clone del main con questi file, Python 3.12 e le versioni fissate
dal suo `requirements.txt`. Il comando e le opzioni sono in `docs/E03.md`.
Su un portatile con poca RAM riduci `workers` a 2 o 1 nella copia del YAML;
mantieni gli altri parametri e usa una nuova directory di output.

Non caricare `data/raw`, il CSV curato completo, ambienti Python o cache Bloch
`store/runs/e03-*`. Lo ZIP contiene già soltanto i file da aggiungere alla PR.
