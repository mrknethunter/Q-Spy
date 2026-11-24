# D-04 — Consegna e caricamento

Questa consegna aggiunge **solo i nuovi file D-04** al progetto D-02 già
caricato: codice, split reali già generati, test e documentazione. Non contiene
un'altra copia di D-01/D-02 e non modifica la loro curation.

## Prima verifica sul tuo computer

Integra i file dello ZIP nelle cartelle corrispondenti della copia locale del
progetto che contiene già D-02. La radice deve contenere `data/curate.py`,
`data/splits.py`, `docs` e `tests`.

Le dipendenze sono le stesse di D-02: Python 3.10+, NumPy e pandas.
Da PowerShell nella radice del progetto:

```powershell
python -m pip install -r requirements-d02.txt
python -m unittest discover -s tests -v
python -c "from data.splits import load_split; s=load_split('P1',0,5,None); print(len(s.train_ids),len(s.val_ids),len(s.test_ids))"
```

L'ultimo comando deve stampare **35 140 700**. Non serve riscaricare i CSV:
`load_split` legge i file già presenti in `data/splits/v1/`.
Usa `py` al posto di `python` se sul tuo Windows avvii Python con quel comando.

## Cosa è stato preparato

| Protocollo | Numero split | Definizione |
| --- | --- | --- |
| P1 | 40 | k=5,10,20,50 per classe, dieci seed, stesso test bilanciato per tutti |
| P2 | 80 | Fast_Scattering sempre riservata e rotazione di ciascuna delle sette classi note, dieci seed |
| P3 | 10 | Train/validation su classi note O3a, test su O3b |

`data/splits/v1/manifest.json` registra i conteggi di ogni partizione per classe,
rilevatore e run, l'impronta del dataset e i checksum dei file. Le regole precise
sono in `docs/splits.md`, compresi i default non specificati nell'issue:
validation P1 di 20 esempi per classe, pool test di 100 per classe, frazioni
P2 60/20/20 sui noti e validation P3 del 20% di O3a.

Per P2 la rotazione di una classe nota lascia **anche Fast_Scattering riservata**:
quella prova ha quindi due classi sconosciute nel test. Per P3 Fast_Scattering
O3a è inutilizzata e quella O3b rimane nel test. Queste interpretazioni sono
documentate per la revisione del gruppo; la suite comune effettiva L-02 non
era presente nel materiale ricevuto.

## Caricare su GitHub

1. Parti dal ramo che contiene D-02. **Se la PR D-02 non è ancora unita a
   `main`, usa `feat/D-02-curation` come sorgente**; se è già stata unita,
   usa `main` aggiornato. Così D-04 dispone del caricamento dati e della
   struttura dei test già introdotti da D-02.
2. Crea il nuovo ramo **`feat/D-04-protocol-splits`**.
3. Premi **Add file → Upload files**.
4. Estrai lo ZIP sul computer. Apri le cartelle contenitore fino a vedere
   direttamente **`data`**, **`docs`**, **`tests`** e **`GUIDA_D04.md`**.
5. Apri una normale finestra di **Esplora file con Windows+E**, seleziona
   questi quattro elementi e trascinali nel riquadro GitHub. La finestra
   “choose your files / Apri” non importa le cartelle.
6. Controlla che l'elenco mostri **`data/splits.py`** e
   **`data/splits/v1/manifest.json`**, senza `Q-Spy-D04/` davanti.
7. Come messaggio del commit scrivi:
   `Add D-04 deterministic protocol splits and tests`.
8. Conferma sul ramo D-04 e apri **Compare & pull request**.

Se D-02 non è ancora unita, confronta inizialmente D-04 con il ramo D-02:
questa PR conterrà solo le aggiunte D-04. Quando D-02 viene unita, il gruppo
può portare la base della PR a `main`. Se D-02 è già in main, usa main
direttamente come base.

I file `.npz` sono **gli indici degli split richiesti dalla task**: devono
essere caricati, insieme al manifest. Sono compressi e non contengono i CSV
originali. Non aggiungere regole che escludano tutta la cartella `data/splits`.

## File della modifica

| File | Scopo |
| --- | --- |
| `data/splits.py` | API, generazione deterministica, verifica del dataset e integrità |
| `data/splits/v1/catalog.npz` | ID e metadati allineati |
| `data/splits/v1/P1.npz`, `P2.npz`, `P3.npz` | Assegnazioni dei 130 split |
| `data/splits/v1/manifest.json` | Versione, configurazione, impronte e conteggi |
| `tests/test_protocol_splits.py` | Test dei protocolli, riproducibilità ed errori |
| `tests/contracts/test_split_contract.py` | Contratto locale della firma condivisa |
| `tests/fixtures/protocol_splits.py` | Piccolo dataset di prova deterministico |
| `docs/splits.md` | Definizione completa dei protocolli |
| `docs/validation_d04.md` | Risultati sui dati originali |

## Testo per la pull request

**Titolo:** `[D-04] Deterministic P1/P2/P3 protocol splits`

```text
Implementata load_split(protocol, seed, k, held_out), con split versionati già generati sul dataset D-02 di 318.801 righe.

P1: 40 split, k=5/10/20/50 per classe e dieci seed, pool di test bilanciato fisso.
P2: 80 split, Fast_Scattering riservata e rotazione delle altre classi; validation con sole famiglie note e test con esempi noti e sconosciuti.
P3: dieci split, train/validation O3a e test O3b, preservando la riserva Fast_Scattering.

Inclusi controlli di disgiunzione, riproducibilità byte per byte, impronta del dataset e conteggi per classe/rilevatore/run. Le scelte su budget di validation e gestione delle classi held-out sono documentate per la revisione del gruppo.

La suite comune L-02 non era disponibile: eseguiti i test di contratto locali sulla firma condivisa. Da verificare il tipo Split e le ulteriori convenzioni del gruppo prima del merge.
```

## Rigenerare gli split

Non è necessario per usarli, ma puoi riprodurre i file con:

```powershell
python -m data.splits --curated-csv data/curated/curated.csv
```

Oppure `python -m data.splits`, che richiama la curation D-02 sugli otto
originali verificati. Un output già identico viene mantenuto; se il dataset
o i parametri sono cambiati il comando rifiuta di sovrascrivere v1.
Per una nuova scelta sperimentale usa una nuova directory, per esempio
`--output-dir data/splits/v2`. L'elenco delle sette feature resta quello
provvisorio D-02: una sua modifica richiede un'impronta e una versione nuove.
