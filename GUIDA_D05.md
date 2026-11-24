# D-05 — Bozza in attesa di L-05

Hai confermato che **L-05 è ancora da fare**. Questa consegna prepara comunque
gli ID veri richiesti da D-05, i test e il collegamento alla futura configurazione.
Il file `data/splits/hardware_subsets.json` è dichiarato `draft_pending_l05`:
non presenta la configurazione come approvata e non avvia lavori hardware.

## Cosa contiene

| Gruppo | Punti | Note |
| --- | --- | --- |
| Main | 120 | 15 per famiglia, inclusa Fast_Scattering; 60 H1 e 60 L1 |
| Controllo | 40 | 5 per famiglia; disgiunto dal main; 20 H1 e 20 L1 |
| Concentrazione | 3 × 30 | Tre sottoinsiemi del main, disgiunti tra loro |
| Riferimenti drift | 5 | Punti del main da rimisurare; cinque famiglie, inclusa Fast_Scattering |

Si usano **160 eventi distinti**: i gruppi di concentrazione e i riferimenti
riutilizzano punti del main. Tutti provengono da O3b nel test P2 congelato
e sono esclusi da **tutte le partizioni train/validation dei 130 split D-04**.
Queste scelte di composizione e riuso sono esplicite, perché l'issue non ne
specifica ogni dettaglio. Il gruppo deve confermarle con L-05.

## Caricare la bozza su GitHub

Questo ZIP contiene **solo le aggiunte D-05**. Il ramo deve includere già
D-02 e D-04. Se D-04 non è stata ancora caricata, completa prima quel caricamento.

1. Se D-04 non è unita a main, parti da **`feat/D-04-protocol-splits`**;
   se è già unita, parti da `main` aggiornato.
2. Crea **`feat/D-05-hardware-subsets`**.
3. Apri **Add file → Upload files**.
4. Estrai questo ZIP e apri le cartelle contenitore fino a vedere direttamente
   `data`, `docs`, `tests` e `GUIDA_D05.md`.
5. Da una normale finestra di **Esplora file, Windows+E**, trascina questi
   quattro elementi nel riquadro GitHub. La finestra “Apri / choose your files”
   non importa le cartelle.
6. Verifica che compaia **`data/splits/hardware_subsets.json`**, senza
   `Q-Spy-D05/` davanti.
7. Messaggio commit: `Add D-05 hardware subset proposal pending L-05`.
8. Apri una **Draft pull request**. Se D-04 è ancora in revisione, la base
   iniziale può essere il ramo D-04 per mostrare solo queste aggiunte;
   altrimenti usa main.

Mantieni esplicita la dipendenza L-05 nella PR. Gli ID e i test si possono già
revisionare; l'issue D-05 resta da completare con il vero freeze e la conferma
delle regole da parte del gruppo.

## Verificare sul computer

Integra lo ZIP nella radice del progetto che contiene già D-04. Le dipendenze
restano NumPy e pandas di D-02, con Python 3.10+. Da PowerShell:

```powershell
python -m unittest discover -s tests -v
python -c "from data.hardware_subsets import load_hardware_subsets; x=load_hardware_subsets(allow_draft=True); print(x['status'],x['subsets']['main']['size'],x['subsets']['control']['size'])"
```

Il secondo comando deve stampare **`draft_pending_l05 120 40`**.
Sostituisci `python` con `py` se usi il launcher Windows. Per leggere il JSON
e fare i test non devi riscaricare i CSV originali: bastano gli split D-04.

Il comando senza `allow_draft=True` restituisce intenzionalmente un errore:
L-05 non è completata. Non è un problema del caricamento dei file.

## Quando L-05 sarà davvero pronta

Il team lead deve completare la propria task: configurazione scientifica,
motivazione, approvazione del gruppo e tag `hw-freeze`. **Questa consegna non
crea quel tag e non inventa i valori n, L, s, gamma o i modelli.**

In una vera copia locale Git del repository, con il tag disponibile, esegui:

```powershell
git fetch origin --tags
python -m data.hardware_subsets --curated-csv data/curated/curated.csv --bind-freeze-from .
```

Se non hai il CSV curato, usa il caricamento D-02 degli originali:

```powershell
python -m data.hardware_subsets --bind-freeze-from .
```

La sola cartella estratta dallo ZIP non ha la cronologia/tag Git: il collegamento
va effettuato nel clone del repository. Vengono letti la configurazione,
`docs/decisions.md` e il manifest D-04 dal commit marcato dal tag. Gli ID
restano identici, mentre il JSON registra commit e checksum. Un tag mancante,
file assente/vuoto o manifest diverso causa un errore senza sostituire la bozza.

Il collegamento verifica i byte Git e non le approvazioni su GitHub né il
significato scientifico del YAML: queste verifiche restano alla L-05 del gruppo.
Assicuratevi che i modelli scelti siano compatibili con il pool di valutazione
P2/O3b proposto. Poi aggiornate il JSON nella PR e completate la revisione.

## File da revisionare

| File | Scopo |
| --- | --- |
| `data/splits/hardware_subsets.json` | ID veri, conteggi, sovrapposizioni e stato pending |
| `data/hardware_subsets.py` | Generazione riproducibile, validazione e collegamento al tag reale |
| `tests/test_hardware_subsets.py` | Test offline, compresi tag Git fittizi usati solo nei test temporanei |
| `docs/hardware_subsets.md` | Regole e schema completi |
| `docs/validation_d05.md` | Prove sui dati originali e limiti attuali |

## Testo per la Draft PR

**Titolo:** `[D-05] Draft hardware subsets pending L-05 freeze`

```text
Preparata la proposta D-05 con ID fissi reali: main 120 bilanciato e comprensivo di Fast_Scattering, controllo 40 disgiunto, tre gruppi da 30 disgiunti tra loro all'interno del main e cinque riferimenti drift riutilizzati dal main.

I 160 eventi unici sono O3b, provengono dal test P2 congelato e non sono mai usati in train/validation in nessuno dei 130 split D-04. Inclusi conteggi per classe/rilevatore/run, selezione deterministica e test.

L-05 è ancora da completare: hardware_subsets.json è draft_pending_l05 e il loader predefinito rifiuta l'uso della bozza. Nessuna configurazione scientifica, approvazione o tag hw-freeze è stato creato da questa modifica.

Da confermare con il gruppo il pool di valutazione e la politica di riuso. Dopo L-05, il comando documentato collega il vero tag/commit e i checksum della configurazione agli stessi ID senza riselezionarli.
```
