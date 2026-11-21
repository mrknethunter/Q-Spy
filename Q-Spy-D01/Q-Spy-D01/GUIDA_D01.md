# D-01 — Consegna e integrazione

Questa cartella parte dallo ZIP Q-Spy fornito e aggiunge il lavoro della D-01.
La struttura completa della L-01 e lo stato dei branch remoti non sono verificabili
da questo snapshot: contieneva soltanto README, licenza e `.gitignore`.

## File della task

| File | Scopo |
| --- | --- |
| `data/download.py` | Scarica gli otto CSV originali, controlla dimensioni e MD5, riutilizza quelli validi |
| `data/manifest.json` | Fissa versione Zenodo, URL, dimensioni e checksum pubblicati |
| `data/profile.py` | Ricalcola la scheda leggendo tutti gli originali verificati |
| `docs/data_card.md` | Fonte, colonne e unità, conteggi per classe/run/detector, distribuzione della confidence, range delle feature, licenza e citazione |
| `docs/data_summary.json` | Statistiche esatte in formato leggibile da altri programmi |
| `tests/test_data_download.py` | Verifica integrità, riparazione, interruzioni, errori e idempotenza |
| `tests/test_data_profile.py` | Verifica statistiche, confini degli intervalli e gestione di dati anomali |

La modifica al `.gitignore` esclude `/data/raw/` e i download temporanei.
Gli originali pesano complessivamente circa 742 MB e non sono inclusi nello ZIP.

## Prova sul tuo computer

Servono Python 3.10 o successivo e una connessione Internet per il primo download.
Apri PowerShell nella cartella del progetto ed esegui:

```powershell
python data/download.py
```

Su Windows, se Python è installato tramite il launcher, puoi usare `py` al posto
di `python` in tutti i comandi. Non servono `pip install` o pacchetti aggiuntivi.

Il primo comando scarica tutti i file in `data/raw/`. Ripetendolo verifica quelli
esistenti e li mantiene se corretti. Questi comandi verificano offline gli
originali, rigenerano la scheda e avviano i test:

```powershell
python data/download.py --verify-only
python data/profile.py
python -m unittest discover -s tests -v
```

## Decisioni ancora da confermare

La task nomina sette feature senza elencarle. Nel materiale disponibile manca
la proposta del progetto che dovrebbe fissarle. Ho quindi documentato questa
selezione **provvisoria**, senza presentarla come una decisione del gruppo:

1. `duration`
2. `peak_frequency`
3. `central_freq`
4. `bandwidth`
5. `amplitude`
6. `snr`
7. `q_value`

Sono sette parametri numerici Omicron disponibili nei CSV. Quando il gruppo
conferma l'elenco, si aggiorna `FEATURES` in `data/profile.py` se necessario e si
rigenera la scheda; non occorre riscaricare i dati validi. Va aggiornato anche
il testo sullo stato provvisorio. L'unità fisica di `amplitude` non è dichiarata
esplicitamente dalla release: la scheda la segnala senza inventare una conversione.

La verifica dei download e i conteggi sui dati sono distinti da queste decisioni.
Non segnare concluso il criterio relativo alle sette feature finché l'elenco
non è stato confermato con la proposta o dal responsabile del progetto.

## Integrare sul repository aggiornato

Lo ZIP non contiene la cronologia Git. Non sostituire il repository del gruppo
con questa cartella. Parti dalla tua copia aggiornata, segui le istruzioni
`CONTRIBUTING.md` della L-01 e crea il branch previsto, per esempio
`feat/D-01-gravity-spy-data`, a partire da `develop`.

Copia i nuovi file della task nelle cartelle corrispondenti del repository.
Aggiungi le due regole dati al `.gitignore` esistente, mantenendo le altre regole.
Se i file esistono già nella versione aggiornata del gruppo, confrontali e integra
le modifiche prima di sovrascriverli. Puoi tenere questa guida fuori dalla PR.

```powershell
git status --short
git check-ignore data/raw/H1_O1.csv
```

Controlla che i CSV originali non siano tra i file da aggiungere. Aggiungi soltanto
gli script, il manifest, la documentazione, i test e la modifica al `.gitignore`.
Apri la pull request verso `develop`, come richiesto dalla L-01; il team lead
gestisce il passaggio a `main`. Il lavoro non è stato pubblicato su GitHub.

## Testo iniziale per la pull request

**Titolo:** `[D-01] Add verified Gravity Spy download and raw data card`

Add a standard-library downloader for the pinned Gravity Spy O1–O3b release,
using the publisher's file sizes and MD5 checksums. Valid files are reused;
failed transfers never install unverified data. Raw CSVs are excluded from Git.

Add a reproducible data card and machine-readable statistics for all eight
detector/run files, plus tests for download failures, idempotence and profiling.

The seven-feature vector is explicitly provisional pending comparison with the
project proposal. The upstream release does not explicitly document the physical
unit/normalization of the amplitude field; this remains flagged in the card.
Keep the PR in draft until these documentation decisions are resolved.
