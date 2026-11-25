## Contesto

Follow-up della review E-03 / PR #74 e della task M-03 (#17). E-03 confronta attualmente OC-SVM proiettata, OC-SVM RBF, Isolation Forest e LOF. La review richiede anche alternative che consumino la **stessa matrice kernel**, per separare il contributo della rappresentazione da quello del rilevatore.

## Implementazione richiesta al proprietario M-03

- Rilevazione tramite densità a kernel.
- kNN nello spazio indotto dal kernel.
- Errore di ricostruzione della kernel PCA.

I tre metodi devono usare gli stessi ID e le stesse matrici di Gram di training e valutazione. Parametri, centratura e trasformazioni devono essere stimati solo sul training del rispettivo fold; il test esterno e Fast_Scattering non possono partecipare alla selezione.

## Criteri di accettazione

- API e convenzione condivise: score più alto = più anomalo; valutazione solo su osservazioni non usate per il fit.
- Scelta degli iperparametri con anomalie surrogate ottenute lasciando fuori classi note, coerente con `docs/decisions.md §4`; budget e numero di fit dichiarati.
- Supporto e controlli per Gram train/train e eval/train, identità/ordine degli ID, centratura e casi degeneri.
- Test sintetici di orientamento, esclusione del held-out, determinismo e compatibilità della stessa matrice kernel fra tutti i detector.
- AUROC, Average Precision e convenzione esplicita di TPR a basso FPR; nessun risultato ricavato invertendo il segno con etichette del test.
- Integrazione E-03/E-04 documentata, con scaler e metadati del kernel comparabili.

Questa issue separa il lavoro aggiuntivo M-03 dalla correzione bloccante della selezione OC-SVM nella PR #74. Non costituisce evidenza che i tre detector siano già implementati.
