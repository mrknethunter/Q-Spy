# E-03 — Consegna su GitHub e riproduzione

## Aggiornare la PR #67 già aperta

1. Estrai lo ZIP della correzione. Apri **il ramo esistente `E-03-novelty-L0`**.
2. Per ogni cartella numerata dello ZIP, entra nella cartella e trascina il
   suo contenuto in **Add file → Upload files**, dalla radice del repository.
   Non trascinare lo ZIP o la cartella numerata esterna.
3. Controlla percorsi come `models/baselines.py`, `experiments/e03_novelty.py`,
   `docs/E03_review.md` e `results/e03/…`. Conferma direttamente sul ramo E-03.
4. Messaggio commit: `Fix E-03 nested novelty tuning and rerun results`.
   La PR #67 si aggiorna automaticamente. **Non serve un nuovo ramo o una nuova PR.**
5. I vecchi risultati negli stessi percorsi vengono sostituiti con il rilancio
   corretto. La correzione include anche `models/baselines.py` condiviso e i test.
6. Crea l'issue M-03 usando `docs/M03_followup.md` e riportane il numero nella PR.
   La creazione automatica ha ricevuto 404: il testo pronto non è un'issue aperta.

Descrizione breve per la PR:

```text
Closes #29

Corregge il tuning condiviso della novelty: gamma dalla griglia a mediana delle
distanze e LOCO annidato sulle sole classi note; nu calibrato al 5% su validation
nota. Gli scaler interni escludono classi surrogate e righe di calibrazione.

Sostituisce i risultati precedenti con un nuovo rilancio E-03 e audit verificato.
Rimuove il FutureWarning, documenta budget effettivi e compatibilità dello scaler
hardware per E-04. Le deviazioni e i limiti sono espliciti in docs/E03_review.md.
M-03: aggiungere qui il numero dell'issue di follow-up dopo averla creata.
```

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
