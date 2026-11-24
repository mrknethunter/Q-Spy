# E-05 — Audit e report di implementazione

Data verifica: 24 novembre 2026  
Repository: Q-SPY  
Branch di lavoro: E-05-task  
Commit iniziale: a2fe30d28f8145d47362767ba65028288c3c16ca

## Stato conclusivo

**PARTIALLY IMPLEMENTED.** Il flusso software E-05 è stato adeguato in modo circoscritto, i test isolati passano e l’API G-03 effettiva è stata verificata da una copia temporanea del sorgente della branch G-03-task. La branch E-05-task non contiene ancora G-03, non ci sono dati H-02, il valore gamma non è approvato né selezionato tramite un artefatto del procedimento training-only e le approvazioni D-05/L-05 sono ancora documentate come pendenti.

Il criterio ufficiale E-05 esportato è “Figures produced by scripts only”. Le figure vengono generate via script e i test dimostrano la generazione deterministica su fixture sintetiche. Non sono state generate figure hardware: con i prerequisiti attuali non è possibile dichiarare E-05 validata su una campagna reale né assegnare lo stato VALIDATED.

Non sono stati inviati job QPU, effettuati commit, push, merge o cambi di branch. Non sono stati modificati contratti, configurazioni, dati, run store o file di altre task. Le figure, il riepilogo e il JSON sotto results/e05 già presenti nel working tree non sono stati rigenerati o alterati.

## A. Stato iniziale e fonti consultate

### Stato iniziale del checkout

La branch corrente era E-05-task al commit indicato sopra. Non risultavano file tracciati modificati. Erano già presenti come non tracciati:

- docs/E05.md
- experiments/e05/ (analysis.py, figures.py, summary.py, cli.py e __init__.py)
- tests/e05/test_e05.py e __init__.py
- results/e05/runs/20261124-e05-final-v2/

Questi materiali precedevano questo intervento. Sono stati modificati solo i file E-05 indicati nella sezione C; i risultati preesistenti non sono stati toccati.

### Documentazione letta

Sono stati letti integralmente la proposta Q-SPY in documentazione/Q-Spy_Project.pdf; la scheda originale tasks/E/E-05.md; tasks/indice.md; documentazione/Gestione_Task_G.txt; docs/contracts.md; docs/decisions.md; la documentazione corrente G-01 e D-05; le schede esportate complete G-01, G-03, H-02, I-04, M-01, D-05 e L-05; il codice e i test E-05 già presenti; il writer H-02, I-04, G-01 e M-01 effettivi.

La scheda originale E-05 non è sostituita da docs/E05.md: è stata consultata separatamente. L’esportazione dell’issue è datata 13 novembre 2026 e ne riporta lo stato open; lo stato remoto attuale non è stato verificato. Le schede e la knowledge base sono esportazioni esterne al checkout, non file versionati in questa branch.

Nel checkout E-05-task non sono presenti tasks/ né docs/G03_documentation.md o diagnostics/compare.py. Le schede originali delle task sono disponibili nelle esportazioni workspace. La documentazione e il codice G-03 sono stati trovati nella branch locale/remota G-03-task. Non è stata trovata una documentazione I-04 o M-01 autonoma dentro docs; per tali interfacce sono stati consultati i contratti, il codice effettivo e le schede originali esportate.

### Dipendenze e relativi confini

| Tipo | Task/modulo | Ruolo E-05 | Stato nel checkout o branch di riferimento |
|---|---|---|---|
| Dichiarata | H-02 | Fornisce la campagna hardware e la convenzione dei run ID. | Writer presente in scripts/qpu_campaign.py; nessuna campagna H-02 nel run store. |
| Dichiarata | G-03 | Calcola errore Bloch, kernel alignment e drift. | Non integrato in E-05-task; sorgente disponibile in G-03-task. |
| Funzionale | I-04 | Legge i run persistiti come (ids, B, meta). | store.io.read_bloch disponibile e usato. |
| Funzionale | G-01 | Calcola valori fuori diagonale, varianza, spettro ed effective rank. | diagnostics.concentration disponibile e riusato. |
| Funzionale | M-01 | Costruisce il projected kernel. | kernels.projected.projected_kernel disponibile e riusato. |
| Funzionale | D-05 | Definisce i subset e i reference ID ammessi. | JSON reale bound_to_hw_freeze; approvazione documentale ancora pending. |
| Contrattuale | L-02 | Definisce forma, dtype e intervallo dei vettori Bloch e le API condivise. | docs/contracts.md presente; nessuna modifica. |
| Approvazione/configurazione | L-05 | Fissa la configurazione hardware e la procedura gamma. | YAML collegato al tag, ma docs/decisions.md lo dichiara draft; gamma assente. |
| Sviluppo/infrastruttura | Qiskit, Aer, IBM Runtime e PyYAML | Servono al writer per produrre campagne H-02. | Assenti dall’ambiente attivo; non sono richiesti per le figure SVG E-05. |

## B. Analisi dell’implementazione e integrazioni

### Stato per programma

| Componente | Stato | Evidenza e limitazione |
|---|---|---|
| experiments/e05/analysis.py | PARTIALLY_IMPLEMENTED | Scopre i gruppi main, control, concentration-n4/n8/n12 e drift-start/end, verifica ID/run/config e coordina I-04, G-01, G-03 e M-01. Sono stati rafforzati i controlli L-02/H-02 e rimossa la scelta gamma da CLI. La reportistica reale resta bloccata dai prerequisiti. |
| experiments/e05/figures.py | IMPLEMENTED_AND_VERIFIED | Produce SVG tramite Python/stdlib. La suite sintetica genera 12 SVG: errore Bloch per qubit, alignment, varianza, effective rank, drift e uno spettro per ciascuno dei sette gruppi. I byte dei file SVG sono risultati riproducibili sulle stesse fixture. Nessuna figura hardware è stata prodotta. |
| experiments/e05/summary.py | IMPLEMENTED_AND_VERIFIED | Genera il riepilogo Markdown dai dati del report. È stato aggiornato il testo gamma. Verificata la generazione su fixture; non è stata eseguita una verifica di impaginazione/rendering grafico. |
| experiments/e05/cli.py | IMPLEMENTED_AND_VERIFIED | Le azioni analyze, figures e summary funzionano nei test. È stata rimossa l’opzione --gamma arbitraria. Analyze genera il report di blocco, ma restituisce codice 0 anche quando il report ha status blocked_by_prerequisites: gli script di automazione devono leggere lo status JSON. |
| tests/e05/test_e05.py | IMPLEMENTED_AND_VERIFIED | Fixture I-04 e StubG03 per test isolati; verifiche su subset, gamma assente, dtype, metadati, source_kind, generazione e riproducibilità. Due test parametrizzati contro G-03 sono saltati quando il modulo non era disponibile e passati quando è stata caricata la sorgente G-03 di riferimento. |
| docs/E05.md | IMPLEMENTED_AND_VERIFIED | Aggiornata per riflettere API, blocker e comportamento effettivi; non presenta fixture come risultati hardware. |

### Integrazioni effettive

**I-04 — IMPLEMENTED_AND_VERIFIED.** E-05 legge esclusivamente con store.io.read_bloch e usa la tupla pubblica (ids, B, meta). I-04 aggiunge run_id, created_at e git_commit quando scrive i dati. I test E-05 verificano round trip e matching; test_store.py è stato eseguito. Non sono richiesti campi job_status, che non fanno parte del writer H-02.

**G-01 — IMPLEMENTED_BUT_NOT_VERIFIED.** E-05 invoca le funzioni reali off_diagonal_values, off_diagonal_variance, eigen_spectrum ed effective_rank, senza copiarne gli algoritmi. La varianza è confrontata con il risultato di G-01 nei test; spettro ed effective rank sono esercitati e passano nei grafici sintetici, ma i test E-05 non verificano ancora un caso analitico atteso per ciascuno. Non è presente un file sorgente dedicato di test G-01 in questo checkout.

**M-01 — IMPLEMENTED_AND_VERIFIED.** E-05 usa kernels.projected.projected_kernel; s non viene riapplicato. La prova tests/contracts/test_l02_contracts.py::test_projected_kernel è passata e verifica il calcolo contro la formula condivisa. E-05 non implementa un secondo kernel e non seleziona gamma.

**G-03 — BLOCKED_BY_DEPENDENCY nel checkout; IMPLEMENTED_AND_VERIFIED sulla branch di riferimento.** La branch G-03-task è disponibile localmente e segue origin/G-03-task. Il commit più recente è 439939f, “Fix compatibilità dtype Bloch”; il merge-base con E-05-task è b5a096efc0dbd9bc119af7fc38a746531071123b. Sono stati ispezionati diagnostics/compare.py, docs/G03_documentation.md e tests/test_compare.py. Le API effettive sono bloch_frobenius_error, kernel_alignment, compare_runs e drift_report. G-03 accetta float32 e float64 secondo L-02, controlla ID/dimensioni/range e riordina per ID. Non sono stati copiati file G né eseguiti merge/cherry-pick. E-05 usa un import statico opzionale: la mancanza esatta del modulo viene riportata esplicitamente; errori interni all’import non vengono nascosti. La suite integrata è stata eseguita con una copia temporanea del file G-03 in /tmp, non dentro il repository.

**H-02 — BLOCKED_BY_DATA.** scripts/qpu_campaign.py genera run ID hw-<timestamp UTC>-<esperimento>-<tier>, con i sette esperimenti previsti. L0 e HW condividono gli stessi D-05 ID; control usa mitigazione disattivata. Il writer salva layout, profondità, calibrazione, job ID, usage, stima, precisione e mitigazione. I controlli E-05 ora confrontano questi campi con il contratto e con il freeze, verificano la dimensione Bloch e il source_kind; non richiedono job_status. Il writer non registra campaign_id/experiment nei metadati: E-05 usa la convenzione ufficiale del run ID e verifica l’eventuale run_id inserito da I-04. Non è stato eseguito H-02 né inviato un job.

**D-05/L-05 — BLOCKED_BY_APPROVAL.** E-05 chiama il loader D-05 in modalità stretta e passa freeze_repo, così il tag live viene confrontato con il commit registrato. Il caricamento reale ha confermato status bound_to_hw_freeze e selection SHA-256 7fee3e50508c951f1161db9e3cd593ea66e609e572d3e38dc7bdbbbc58a0c7c7. Tuttavia docs/hardware_subsets.md e docs/validation_d05.md continuano a descrivere una proposta/pending; docs/decisions.md dice “draft for lead approval”. Il binding al tag non è trattato come approvazione.

### Gamma

Il kernel M-01 richiede gamma. Nel file configs/hardware_frozen.yaml gamma non esiste. docs/decisions.md descrive una procedura su distanze quadrate training-only, gamma = c / median(d²_train), con c dalla griglia documentata. Il run store non contiene un artefatto di selezione gamma training-only né esiste un’API condivisa che consegni a E-05 valore, metodo e provenienza. E-05 non sceglie gamma dai punti D-05 hardware/test e non accetta più un valore numerico arbitrario da CLI. Se gamma manca, i confronti Bloch possono essere eseguiti quando G-03 e gli altri prerequisiti sono disponibili, ma kernel alignment, concentrazione e spettri sono esplicitamente bloccati.

Classificazione: selezione gamma E-05 = BLOCKED_BY_DEPENDENCY; approvazione dei parametri L-05 = BLOCKED_BY_APPROVAL; assenza di un contratto per trasferire la selezione training-only = ARCHITECTURAL_ISSUE da risolvere con il Project Owner.

## C. Modifiche effettuate e codice conservato

| File E-05 | Modifica | Impatto |
|---|---|---|
| experiments/e05/analysis.py | Import statico G-03 con blocker esplicito; validazione Bloch per shape, qubit, dtype float32/float64, finitezza e range; verifica di precisione, backend, layout, profondità, calibrazione, usage, estimate e mitigazione H-02; source_kind limitato a hardware/synthetic_fixture; verifica D-05 contro il tag live; gamma solo da config congelata; stati approvazione interpretati in modo esplicito. | Nessun algoritmo G-01/G-03/M-01 duplicato; gli input malformati o non provenienti da H-02 valido vengono respinti prima delle metriche. |
| experiments/e05/cli.py | Rimossa l’opzione --gamma non approvata. | La CLI non consente di far passare per pre-registrato un valore arbitrario. |
| experiments/e05/summary.py | Gamma assente etichettato “not approved/frozen”. | Il riepilogo espone il blocker corretto. |
| tests/e05/test_e05.py | Fixture H-02 arricchite con metadati conformi al writer reale; test nuovi per dtype, shape/provenienza, tag D-05, approval parsing, gamma e API G-03 reale condizionale. Confronto byte per byte anche per SVG. | Separa test isolati con StubG03 da test d’integrazione con il modulo vero. |
| docs/E05.md | Aggiornata architettura, dipendenze, metadata attesi, gamma e limiti. | La documentazione non dichiara completata la validazione hardware. |
| docs/E05_IMPLEMENTATION_AUDIT.md | Nuovo report. | Contiene audit, modifiche, prove e checklist residua. |

Sono stati conservati il flusso di caricamento separato dalla logica metrica, i renderer SVG, la generazione del riepilogo, la struttura JSON dei report e gli output per-run isolati. experiments/e05/figures.py non è stato modificato. Non sono stati creati sostituti per G-01/G-03/I-04/M-01.

## D. Test scritti ed eseguiti

I test E-05 usano fixture sintetiche marcate synthetic_fixture e un test double esplicito StubG03 quando G-03 non è integrato. Questi test dimostrano il comportamento del codice E-05, non una misura hardware. Il test condizionale d’integrazione usa le API pubbliche G-03 vere e viene saltato se diagnostics.compare non è disponibile.

| Comando/prova | Esito | Tipo e dipendenze |
|---|---|---|
| PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/e05-audit-final-isolated tests/e05 tests/test_store.py tests/test_hardware_subsets.py -q | 40 passed, 2 skipped, 1 warning, 3 subtests passed | Test E-05, fixture sintetiche, I-04 e D-05 reali. I due skip sono le varianti d’integrazione G-03 perché il modulo non è nella branch. |
| PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/e05-g03-ref .venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/e05-audit-final-g03 tests/e05 tests/test_store.py tests/test_hardware_subsets.py /tmp/e05-g03-ref/tests/test_compare.py -q | 75 passed, 1 warning, 3 subtests passed | Stesse prove più test G-03 presi dal commit 439939f e import reale da /tmp. Le due varianti E-05 float32/float64 hanno entrambe eseguito G-03. |
| PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/contracts/test_l02_contracts.py::test_projected_kernel -q | 1 passed | Test del contratto L-02/M-01 sulla formula projected kernel. |
| build_report() sul run store corrente, senza scrivere output | blocked_by_prerequisites; 4 run caricati, 0 errori I-04; nessuna campagna trovata | Verifica read-only di D-05, L-05, gamma, H-01 e disponibilità G-03 nel checkout. |

Il warning è pandas Pandas4Warning su float_precision in data/hardware_subsets.py:333; non è stato introdotto da E-05. La suite completa del repository non è stata lanciata: qiskit, qiskit-aer, qiskit-ibm-runtime e PyYAML non sono installati nell’ambiente. Le versioni attive osservate sono NumPy 2.5.5, pandas 3.1.1 e scikit-learn 1.9.1, diverse dai pin in requirements.txt (2.2.6, 2.3.3 e 1.7.2). E-05 SVG non richiede Matplotlib; il writer H-02 richiede Qiskit/IBM Runtime e PyYAML.

## E. Validazione, dati e riproducibilità

### Dati disponibili

Il run store contiene soltanto quattro run H-01 da 10 punti: HW, L0, L1 e L2, con prefisso hw-20261124T005515Z. Il pilot usa s=1.0, mentre il freeze documentale indica s=0.1778; non corrisponde ai sette gruppi H-02 e ai subset D-05. E-05 lo esclude correttamente. I metadati HW del pilot includono job ID, backend, calibrazione, layout, profondità, usage e precisione; la mancanza di una campagna completa, non l’assenza generale di metadati pilot, è il blocker dati.

La campagna H-02 attesa contiene main, control, tre gruppi concentration, drift-start e drift-end, ciascuno confrontato tra L0 e HW. Nessuno di questi gruppi è presente per la campagna reale. Non sono stati creati campioni sintetici per colmare il vuoto nel run store.

### Figure e pipeline

L’analisi fixture ha prodotto automaticamente 12 SVG e riepilogo; JSON, riepilogo e tutti gli SVG sono risultati identici tra due output directory con input uguali. Ciò verifica la riproducibilità della generazione sintetica. Non dimostra riproducibilità su una campagna hardware o visiva, e non ci sono figure nella reportistica reale corrente. L’esecuzione CLI analyze scrive un report di blocco; la sua invocazione read-only tramite build_report non ha prodotto o modificato file.

Una volta presenti approvazioni, valore gamma ammesso, G-03 integrato e run H-02 validi, analyze genera report, figure disponibili e summary senza modifiche manuali. Questa riproduzione hardware end-to-end non è ancora verificabile con i dati correnti.

### Approvazioni e configurazione

Il file hardware_frozen.yaml contiene n=8, layers=2, s=0.1778, precision=0.03, concentration_qubits 4/8/12, budget 587 s, dispositivo ibm_kingston e scaler P2 seed 0. Non contiene gamma. Il commento YAML e il tag hw-freeze suggeriscono freeze, mentre docs/decisions.md e i documenti D-05 dicono draft/pending. Questo conflitto non viene risolto dal tag né da E-05.

## F. Problemi tecnici e decisioni da chiedere

1. **Integrazione G-03:** la branch corretta è disponibile ma non è stata unita. Il test contro la copia temporanea prova la compatibilità delle API; il normale checkout E-05 continua a registrare diagnostics.compare come dipendenza assente.
2. **Provenienza gamma:** manca un artefatto/API approvato per trasferire il gamma selezionato secondo il procedimento training-only, con c e riferimento alle distanze/ID di training. Non va scelto sui punti H-02 di valutazione. Il Project Owner deve indicare la sorgente ufficiale prima di estendere il flusso oltre un valore congelato.
3. **D-05/L-05:** il JSON D-05 è bound e il tag live è verificato, ma i documenti di approvazione sono in conflitto/pending. Il Project Owner deve registrare le approvazioni e rendere coerenti YAML, decision record, validazione e lifecycle D-05.
4. **Formato della prova di approvazione D-05:** L-02 non definisce uno schema di approval. E-05 usa un indicatore testuale esplicito; il Project Owner deve confermare come registrare l’approvazione D-05 in modo stabile. La logica non deve dedurre l’approvazione dalla sola esistenza di hw-freeze.
5. **Dati H-02:** servono i run reali registrati attraverso I-04 secondo H-02, con sette esperimenti, coppie L0/HW, ID D-05 e metadati completi. Non sono stati inviati job: la richiesta di implementazione vieta nuove esecuzioni hardware.
6. **Ambiente H-02:** nell’ambiente attivo mancano Qiskit, Aer, IBM Runtime e PyYAML; serve l’ambiente fissato da I-01 prima di eseguire il writer. Questo non impedisce i test E-05 di sola lettura e figure.
7. **Copertura G-01:** aggiungere casi analitici E-05 per eigen_spectrum ed effective_rank oppure usare la suite G-01 integrata quando disponibile; ora l’esercizio è indiretto attraverso i fixture e il rendering.
8. **Esito CLI:** analyze restituisce 0 anche se il report segnala blocchi. Non invalida la scrittura corretta del report, ma chi automatizza deve interrogare status nel JSON. Una modifica futura del codice di uscita richiede una decisione d’interfaccia del CLI.

Non sono state riscontrate incompatibilità tra la forma/dtype Bloch E-05 e il contratto L-02 effettivo: entrambi i dtype float32 e float64 sono accettati senza conversione dei dati archiviati e sono stati provati contro G-03. L’implementazione non richiede job_status. Non è stata rilevata una necessità di modificare contratti condivisi.

## G. Attività residue ordinate

| # / Priorità | Attività e motivazione | Dipendenze | File/aree coinvolte | Criterio verificabile di completamento |
|---|---|---|---|---|
| 1 / P0 | Risolvere e registrare approvazioni L-05/D-05 e riconciliare YAML, tag e documenti; il binding corrente non prova approvazione. | Project Owner e gruppo; policy D-05/L-05. | configs/hardware_frozen.yaml, docs/decisions.md, docs/hardware_subsets.md, docs/validation_d05.md, data/splits/hardware_subsets.json. | Stato approvato esplicito e coerente; loader strict e controllo tag live passano; nessuna fonte continua a dichiarare pending. Nessuna approvazione deve essere inventata da E-05. |
| 2 / P0 | Stabilire e rendere disponibile il risultato gamma della procedura ufficiale training-only; E-05 rifiuta valori CLI arbitrari per prevenire leakage. | Decisione Project Owner e componente che seleziona gamma sui dati training. | Output/artifact della selezione M-02/M-03 o configurazione L-05 approvata; eventuale adattamento limitato E-05 solo dopo accordo sull’interfaccia. | Report E-05 registra gamma, metodo, c e provenienza di training; test dimostra che nessun ID D-05 valutato entra nella selezione. |
| 3 / P0 | Integrare G-03 nel flusso della branch con il processo gestito dal proprietario della categoria G; questa attività non è stata fatta qui per il vincolo no merge/copy. | G-03-task commit 439939f e workflow di integrazione del repository. | diagnostics/compare.py, docs/G03_documentation.md, tests/test_compare.py, branch E-05-task. | Import G-03 disponibile nel checkout E-05; suite G-03 e test integrazione E-05 passano senza StubG03 per il percorso reale. |
| 4 / P1 | Preparare ambiente conforme ai pin prima di produrre H-02. | I-01 e Qiskit/IBM Runtime/PyYAML. | requirements.txt, ambiente locale gestito da I-01; nessun file E-05 necessario. | Versioni installate corrispondono ai pin richiesti e il writer H-02 è importabile; dry-run locale non invia job. |
| 5 / P1 | Produrre e registrare H-02 solo dopo l’approvazione e l’autorizzazione/budget del Team Lead. | Attività 1–4, accesso IBM e budget approvato. | Script H-02, store/runs e risultati campagna; non eseguita in questo intervento. | Sette esperimenti con run L0/HW, ID D-05 esatti, run ID/meta coerenti e usage/calibrazione/layout/mitigazione validi. Nessun run H-01 usato come sostituto. |
| 6 / P1 | Eseguire il test end-to-end read-only sui run H-02 e rigenerare l’output E-05. | Attività 1–5 e G-03 integrato. | experiments/e05, tests/e05 e results/e05. | Report hardware non bloccato; metriche, figura e riepilogo generati da CLI e riferiti a IDs/campagna corretti; nessun mock nel test d’integrazione. |
| 7 / P2 | Rafforzare copertura numerica G-01 e controllo visivo delle figure effettive. | Fixture/G-01 integrato e output H-02. | tests/e05, eventualmente test condivisi G-01; risultati SVG. | Casi analitici per varianza, spettro ed effective rank; SVG hardware renderizzati e controllati; nessuna figura ritoccata manualmente. |

## H. Checklist per il completamento definitivo

- [ ] Il Project Owner conferma lo stato effettivo di L-05 e D-05 e rende coerenti il tag, il file congelato e i record di approvazione.
- [ ] Il proprietario del gamma definisce l’interfaccia ufficiale per trasferire a E-05 il valore selezionato solo su dati training, con provenienza e c.
- [ ] G-03 viene integrato nella branch E-05 tramite il workflow del repository; nessun duplicato locale viene creato.
- [ ] L’ambiente H-02 rispetta i pin del progetto; esecuzione dry-run verificata senza QPU.
- [ ] Dopo autorizzazione esplicita e budget, H-02 registra una campagna completa con metadati validi; non usare il pilot H-01.
- [ ] E-05 viene eseguita sulla campagna completa; tutte le metriche e figure previste sono prodotte automaticamente, con output ripetibile e tracciabile.
- [ ] Suite d’integrazione reale, test figure e controlli G-01 sono passati; viene aggiornato questo report con le prove effettive.
- [ ] Solo dopo tali verifiche si assegna VALIDATED; fino ad allora lo stato resta PARTIALLY IMPLEMENTED.

## Addendum — sincronizzazione con `origin/main` (25 novembre 2026)

Le sezioni precedenti registrano l'audit eseguito il 24 novembre 2026 e
descrivono correttamente lo stato di quel momento. Il 25 novembre il branch
`E-05-task` è stato aggiornato con un merge commit `6b237e3` di `origin/main`
(`df1e155`), che include G-03. La dipendenza `diagnostics.compare` ora è
importabile nel branch e i test E-05 d'integrazione float32/float64 sono stati
eseguiti contro l'API reale.

Verifiche consentite eseguite sul tree sincronizzato:

- `tests/e05`: 35 passed, 0 failed, 0 skipped;
- `tests/test_compare.py`: 33 passed, 0 failed, 0 skipped;
- `tests/test_store.py`: 4 passed, 0 failed, 0 skipped;
- `tests/contracts/test_l02_contracts.py::test_projected_kernel`: 1 passed,
  0 failed, 0 skipped.

Il file `data/splits/hardware_subsets.json` ricevuto da `origin/main` registra
ora `draft_pending_l05` e `l05_freeze: null`; non va quindi descritto come
approvato o associato al freeze. D-05/L-05 restano pending secondo la
documentazione E-05, non è presente un gamma selezionato e non è disponibile
una campagna H-02 reale. Il risultato preesistente
`results/e05/runs/20261124-e05-final-v2/` è un'istantanea del 24 novembre che
riporta G-03 assente; è conservato come output storico, non come stato corrente.
Nessun job QPU è stato inviato e nessuna validazione hardware è dichiarata.
