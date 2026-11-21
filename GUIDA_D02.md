# D-02 — Preparazione del dataset Gravity Spy

Questa consegna contiene il codice D-01 ricevuto, senza modifiche agli script
di download e profilazione, più la curation D-02. Le cartelle del progetto sono
riportate a un solo livello: la radice è quella che contiene `data`, `docs` e
`tests`. Gli originali e il CSV curato si rigenerano con i comandi sotto e non
sono inclusi nello ZIP.

## Prova sul tuo computer

Estrai lo ZIP e apri PowerShell **dentro `Q-Spy-D02`**.
Servono Python 3.10 o successivo e circa 2 GB liberi per originali, file
temporanei e CSV curato. Installa le due dipendenze di D-02:

```powershell
python -m pip install -r requirements-d02.txt
```

Se su Windows usi `py` per avviare Python, sostituisci `python` con `py` in tutti
i comandi. I test usano `unittest`: non serve installare pytest.

Se hai già scaricato gli otto CSV di D-01, copia la cartella `raw` dentro `data`
oppure indica il percorso con `--raw-dir`. Altrimenti esegui:

```powershell
python data/download.py
```

Il downloader controlla i checksum e riutilizza gli originali già validi.
Genera quindi il dataset curato e avvia i test:

```powershell
python -m data.curate
python -m unittest discover -s tests -v
```

Puoi eseguire soltanto i test di contratto locali con:

```powershell
python -m unittest discover -s tests/contracts -v
```

I test sono offline e non richiedono i 742 MB di dati originali.

Il comando di curation produce:

| File | Contenuto |
| --- | --- |
| `data/curated/curated.csv` | Dataset curato, con feature ancora grezze |
| `docs/curation_summary.json` | Configurazione, checksum delle sorgenti, esclusioni e conteggi |
| `docs/data_card.md` | Scheda D-01 preservata e sezione D-02 aggiornata |

La consegna include già i due report ottenuti dalla verifica sui dati originali;
rigenerandoli puoi confrontare i tuoi risultati. Se esegui `data/profile.py`,
questo script D-01 riscrive la propria scheda: esegui successivamente
`python -m data.curate` per riaggiungere la sezione D-02.

## Regole adottate

- Soglia inclusiva `0.9 <= confidence <= 1`; le etichette sono predizioni ML.
- Sette classi note: `Blip`, `Extremely_Loud`, `Koi_Fish`,
  `Low_Frequency_Burst`, `Low_Frequency_Lines`, `Scattered_Light`, `Tomte`.
  La motivazione e i conteggi reali sono in `docs/data_card.md`.
- `Fast_Scattering` resta nel dataset come classe held-out, solo da O3a/O3b.
  La colonna `is_held_out` la identifica. Le task successive devono escluderla
  dall'addestramento di novelty detection: la curation non effettua gli split.
- ID pubblicati ripetuti vengono deduplicati all'interno del rilevatore;
  poi si raggruppano trigger con gap temporali consecutivi <= 0,1 s,
  separatamente per H1 e L1 e senza usare le etichette. Le catene transitive
  appartengono allo stesso gruppo. Si tiene la confidenza maggiore, poi SNR,
  con spareggio deterministico. Gli stessi istanti H1/L1 non vengono fusi.
- La finestra di 0,1 s è una scelta tecnica configurabile, non un parametro
  scientifico imposto dalla release. `--coincidence-window 0` effettua soltanto
  deduplica su coincidenze temporali esatte.
- Si rimuovono feature non finite/nonpositive e tempi/ID non validi.
  `q_value=1e20`, segnalato da D-01, è escluso per prudenza. Non si afferma che
  la release ne dimostri il significato di sentinel; `--keep-extreme-q` lo mantiene.
- Ogni classe selezionata deve sopravvivere in entrambi i rilevatori, altrimenti
  viene restituito un errore esplicito invece di un dataset incompleto.
- Nessun bilanciamento o sottocampionamento è imposto per default.
  Un cap opzionale campiona all'interno delle coppie `(label, ifo)`, mantenendo
  entrambi i rilevatori e approssimativamente le loro proporzioni. Il seed è fisso.

Esempio di configurazione:

```powershell
python -m data.curate --confidence 0.95 --coincidence-window 0.05 --max-per-class 1000 --seed 42
```

Questo esempio **cambia il dataset e i report**: i conteggi distribuiti nello
ZIP corrispondono invece ai parametri predefiniti.

## Uso da Python

```python
from data.curate import CurateConfig, load_curated

df = load_curated()
# Oppure:
df = load_curated(config=CurateConfig(confidence_threshold=0.95))
known = df.loc[~df["is_held_out"]]
held_out = df.loc[df["is_held_out"]]
```

`load_curated()` non scarica, non salva, non scala e non crea split.
I report sono accessibili anche come `df.attrs['curation']`.
Lo schema contiene `id`, `gps`, `ifo`, `label`, `confidence`, le sette feature
grezze di D-01, `run` e `is_held_out`. Le feature, GPS e confidence sono float64;
GPS è in secondi GPS, non Unix/UTC. Il clustering usa i secondi/nanosecondi esatti
del picco. Gli ID sono `ifo:gravityspy_id`, stabili e unici nel dataset curato.

## Integrare nel repository del gruppo

Lo ZIP ricevuto aveva il codice in `Q-Spy-D01/Q-Spy-D01/` rispetto alla radice
GitHub. Evita di aggiungere un'altra cartella contenitore dentro quella struttura.
La task richiede i percorsi `data/curate.py` e `docs/data_card.md` **dalla radice
effettiva del progetto**. Se il gruppo ha nel frattempo sistemato la struttura,
integra lì i file D-02. Lo ZIP non contiene la cronologia Git.

Non sostituire l'intero repository con lo ZIP. Lavora su un branch D-02 dalla
base prevista dal gruppo, che deve includere D-01 e L-02. Copia i file seguenti
nelle cartelle corrispondenti e conserva le altre modifiche del gruppo:

| File | Azione |
| --- | --- |
| `data/curate.py` | Nuova implementazione D-02 |
| `data/__init__.py` | Aggiungi solo se il package non ha già un proprio file |
| `requirements-d02.txt` | Aggiungi o integra le dipendenze nella configurazione esistente |
| `tests/test_data_curate.py` | Nuovi test di curation |
| `tests/contracts/test_curated_contract.py` | Contratto locale; integra senza sostituire la suite L-02 |
| `tests/fixtures/curation.py` | Fixture offline |
| `tests/__init__.py`, `tests/contracts/__init__.py`, `tests/fixtures/__init__.py` | Aggiungi solo se assenti |
| `docs/data_card.md` | Integra la sezione D-02; conserva eventuali aggiornamenti D-01 del gruppo |
| `docs/curation_summary.json`, `docs/validation_d02.md` | Report e prove sui dati originali |

Aggiungi alle regole ignore esistenti `/data/raw/`, `/data/curated/` e `*.part`.
Se i dati restano temporaneamente in cartelle annidate, escludi anche quei
percorsi. Non caricare i CSV su GitHub. README, licenza, manifest, script D-01
e relativi test sono inclusi per poter provare la consegna, ma non sono nuove
modifiche D-02.

## Punti da verificare prima del merge

Il branch fornito non contiene L-02: la suite locale verifica la firma e lo
schema dell'issue condivisa, ma la suite comune effettiva resta da eseguire
dopo l'integrazione. La selezione delle sette feature è ancora provvisoria
in D-01 e deve essere confermata dal gruppo. La finestra di coincidenza,
la scelta delle classi e il trattamento di Q estremo sono espliciti e
configurabili, così possono essere valutati nella revisione della PR.

## Testo per la pull request

**Titolo:** `[D-02] Curate confident Gravity Spy classes and held-out Fast_Scattering`

Implement `data.curate.load_curated()` over the eight checksum-verified D-01
CSVs. Apply configurable confidence filtering, deterministic detector-local
ID/time deduplication, seven documented known classes and O3-only
Fast_Scattering marked as held-out. Optional class caps sample within detector
strata without replacement; no scaling or train/test split is performed.

Add offline curation and local interface tests, real per-detector/run counts
and reproducible curation provenance. Document the configurable 0.1 s
coincidence window, conservative extreme-Q exclusion and provisional feature
list. The actual L-02 contract suite was absent from the supplied branch and
must be run after integration.
