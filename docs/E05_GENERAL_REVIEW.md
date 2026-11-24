# E-05 — Revisione generale

## Osservazioni

### ER-01 — Exit status non segnala i prerequisiti bloccanti

- **File/funzione:** `experiments/e05/cli.py`, `main()`.
- **Tipo:** limite operativo noto.
- **Descrizione:** il comando `analyze` restituisce `0` anche quando `build_report()` produce `status = blocked_by_prerequisites`.
- **Conseguenza:** uno script che controlla soltanto il codice di uscita può trattare come riuscita un'analisi che non ha soddisfatto i prerequisiti.
- **Gravità proposta:** Medium.
- **Dipendenza/decisione:** serve una decisione sul contratto operativo della CLI.
- **Soluzione proposta:** in una modifica futura, restituire un codice non zero per gli stati bloccati oppure documentare e standardizzare il controllo del campo `status` nel JSON.

### ER-02 — La rigenerazione può lasciare figure obsolete

- **File/funzione:** `experiments/e05/figures.py`, `render_figures()`; `experiments/e05/cli.py`, ramo `figures`.
- **Tipo:** bug dimostrabile dal flusso di scrittura.
- **Descrizione:** `render_figures()` scrive i file disponibili ma non rimuove gli SVG generati da una precedente analisi nella stessa directory. Il comando `figures` aggiorna l'elenco nel JSON senza eliminare eventuali file non più generati.
- **Conseguenza:** la cartella può contenere figure obsolete che non compaiono in `figures_generated` e contraddicono il report corrente.
- **Gravità proposta:** Medium.
- **Dipendenza/decisione:** nessuna dipendenza esterna; va definito il comportamento di rigenerazione della directory.
- **Soluzione proposta:** rendere la rigenerazione idempotente rimuovendo in modo circoscritto gli SVG E-05 precedenti oppure scrivendo ogni rigenerazione in una directory nuova.

### ER-03 — Il controllo della calibrazione verifica soprattutto la forma

- **File/funzione:** `experiments/e05/analysis.py`, `_hardware_metadata_error()`.
- **Tipo:** rischio potenziale confermato dalle interfacce del writer.
- **Descrizione:** per T1, T2 e readout error viene controllata la presenza delle chiavi, non che i valori siano numerici, finiti e nel dominio atteso. Anche `two_qubit_errors` è accettato come qualsiasi mapping, incluso vuoto. Il writer H-02 può registrare `None` quando una proprietà del backend non è disponibile.
- **Conseguenza:** una campagna può superare il controllo di presenza pur avendo snapshot di calibrazione senza misure utilizzabili o senza errori dei coupler.
- **Gravità proposta:** Medium.
- **Dipendenza/decisione:** il responsabile H-02 deve definire quali valori e coupler sono obbligatori per una calibrazione valida.
- **Soluzione proposta:** concordare il dominio e la copertura minimi, poi validarli senza alterare il formato condiviso H-02.

### ER-04 — La provenienza della selezione gamma non è verificata dal codice

- **File/funzione:** `experiments/e05/analysis.py`, `_validate_gamma()` e `build_report()`.
- **Tipo:** decisione architetturale necessaria.
- **Descrizione:** se `gamma` compare nel file congelato, il codice controlla che sia finito e positivo e lo etichetta `frozen_config`; non legge un artefatto della selezione training-only né ne verifica metodo, parametro `c` o dati di training.
- **Conseguenza:** il valore numerico può essere consumato senza che il report dimostri come è stato selezionato, mentre la specifica richiede una provenienza approvata e non leakage dai dati di valutazione.
- **Gravità proposta:** Medium.
- **Dipendenza/decisione:** il Project Owner e i responsabili della selezione gamma devono definire la sorgente autorevole e il relativo contratto.
- **Soluzione proposta:** definire un artefatto/API versionato con valore, metodo, `c` e provenienza degli ID training, poi collegarlo a E-05.

### ER-05 — I report salvati ricevono una validazione strutturale minima

- **File/funzione:** `experiments/e05/cli.py`, `_read_report()`; `experiments/e05/figures.py`, `render_figures()`.
- **Tipo:** rischio potenziale nell'input di rigenerazione.
- **Descrizione:** `_read_report()` verifica soltanto il campo `schema`. Le chiavi dei gruppi contenute nel JSON vengono interpolate direttamente nei nomi degli SVG degli spettri.
- **Conseguenza:** un report modificato o malformato può causare errori di rendering; nomi di esperimento appositamente costruiti possono influenzare il percorso di output della figura.
- **Gravità proposta:** Medium.
- **Dipendenza/decisione:** nessuna dipendenza esterna; serve definire quali forme di report salvato sono supportate come input CLI.
- **Soluzione proposta:** validare struttura e nomi dei gruppi contro lo schema E-05 ammesso e verificare che il percorso risolto resti dentro la directory `figures`.

### ER-06 — Il parsing degli stati di approvazione dipende dal testo dei documenti

- **File/funzione:** `experiments/e05/analysis.py`, `inspect_l05_approval()` e `inspect_d05_approval()`.
- **Tipo:** rischio di fragilità documentale.
- **Descrizione:** L-05 e D-05 sono interpretati tramite pattern e frasi specifiche in Markdown. Frasi storiche o cambi di formulazione possono influenzare lo stato rilevato anche quando il significato editoriale è diverso.
- **Conseguenza:** il report può classificare un'approvazione come pendente o non verificata per effetto della formulazione del documento, non di uno stato machine-readable stabile.
- **Gravità proposta:** Medium.
- **Dipendenza/decisione:** i responsabili D-05/L-05 devono stabilire una fonte strutturata dell'approvazione compatibile con il processo del progetto.
- **Soluzione proposta:** introdurre, tramite decisione condivisa, un campo o artefatto versionato con stato, approvatore e riferimento alla configurazione; mantenere i documenti descrittivi come contesto.

### ER-07 — Due test assumono che le approvazioni restino pendenti

- **File/funzione:** `tests/e05/test_e05.py`, `test_current_l05_decision_is_not_approved_by_tag_alone()` e `test_current_d05_binding_does_not_imply_final_acceptance()`.
- **Tipo:** bug di fragilità dei test rispetto all'evoluzione del progetto.
- **Descrizione:** i test verificano uno stato documentale corrente specifico (`pending`) e testi descrittivi specifici, anziché verificare soltanto che gli stati siano interpretati coerentemente con i documenti presenti.
- **Conseguenza:** dopo un'approvazione legittima, la suite E-05 può fallire pur in presenza di codice corretto e documentazione aggiornata.
- **Gravità proposta:** Low.
- **Dipendenza/decisione:** aggiornamento necessario quando il processo registra l'approvazione effettiva.
- **Soluzione proposta:** separare i test unitari del parser, basati su documenti temporanei, dai controlli read-only dello stato corrente; rendere questi ultimi esplicitamente dipendenti dallo stato atteso del checkout.

### ER-08 — Integrazione G-03 (risolta nel working tree aggiornato)

- **File/funzione:** `experiments/e05/analysis.py`, import e `_load_g03()`; `tests/e05/test_e05.py`, `test_actual_g03_integration_when_dependency_is_available()`.
- **Tipo:** osservazione storica, risolta dall'integrazione di `origin/main`.
- **Descrizione:** al momento della revisione iniziale `diagnostics.compare` mancava dal checkout E-05-task e i test condizionali venivano saltati. Il merge di `origin/main` (commit `6b237e3`) ha portato G-03 nel branch; i test d'integrazione E-05 per float32 e float64 sono ora eseguiti e passano.
- **Conseguenza:** la precedente limitazione di dipendenza non è più attuale nel branch aggiornato. I test usano ancora fixture sintetiche e non dimostrano una validazione hardware.
- **Gravità proposta:** Informational, risolta per l'integrazione del modulo.
- **Dipendenza/decisione:** nessuna per l'import e le API; resta la normale necessità di dati H-02 approvati per una prova hardware.
- **Soluzione proposta:** nessuna ulteriore integrazione G-03; mantenere aggiornati i riferimenti di stato nei report E-05.

### ER-09 — Il lettore della configurazione supporta solo un sottoinsieme YAML

- **File/funzione:** `experiments/e05/analysis.py`, `_parse_scalar()` e `read_frozen_config()`.
- **Tipo:** limite noto dell'implementazione.
- **Descrizione:** il parser gestisce la configurazione piatta attuale, ma non implementa YAML completo; per esempio, la rimozione dei commenti avviene prima di interpretare stringhe quotate e le liste inline sono divise sulle virgole.
- **Conseguenza:** una futura modifica al file congelato con sintassi YAML valida ma più articolata può essere letta in modo errato o rifiutata.
- **Gravità proposta:** Low.
- **Dipendenza/decisione:** eventuale modifica del formato o delle dipendenze di configurazione deve rispettare l'ambiente ufficiale del progetto.
- **Soluzione proposta:** mantenere e documentare il sottoinsieme ammesso oppure adottare un parser YAML già approvato quando il contratto di configurazione lo consente.

### ER-10 — I metadati non costituiscono un'attestazione autenticata dell'origine

- **File/funzione:** `experiments/e05/analysis.py`, `_campaign_source()` e `_hardware_metadata_error()`.
- **Tipo:** rischio di provenienza e limite architetturale.
- **Descrizione:** quando `source_kind` manca, la classificazione `hardware` deriva dalla presenza e coerenza dei metadati H-02. Il formato attuale non include una firma o un riferimento autenticato al ledger che dimostri che quei metadati provengono da un job QPU realmente inviato.
- **Conseguenza:** una directory costruita o alterata manualmente con metadati conformi può apparire hardware secondo i controlli E-05, pur non fornendo una prova indipendente dell'origine.
- **Gravità proposta:** Medium.
- **Dipendenza/decisione:** il responsabile H-02 e il Project Owner devono decidere se l'attuale provenance-by-metadata è sufficiente o se il progetto richiede una verifica del ledger.
- **Soluzione proposta:** se richiesto dal contratto, collegare il report a un artefatto ledger verificabile o a una attestazione di provenienza definita da H-02; non introdurre un nuovo campo localmente in E-05.
