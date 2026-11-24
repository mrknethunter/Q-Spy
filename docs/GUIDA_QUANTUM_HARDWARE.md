# Guida: parte quantistica e hardware di Q-Spy

Questa guida spiega cosa fa la parte quantistica del progetto, perché è fatta così, dove sta il codice, come si esegue
e cosa abbiamo già misurato sull'hardware IBM. Riguarda le task Q-01…Q-06, M-01, M-05, I-04, H-01, H-02 ed E-01.

## 1. Il quadro in una pagina

```
trigger Omicron (7 numeri)                         D-02 curation, D-03 scaling in [-1, 1]
        │
        ▼
angoli θ = s·π·x̃  ──► circuito U(x) su n qubit     Q-01  quantum/feature_map.py
        │
        ▼
misura di X, Y, Z su ogni qubit ──► vettori di Bloch b(x), forma (n, 3)
        │                                          Q-02  quantum/observables.py
        │                                          Q-03/04/05  quantum/embed.py (L0, L1, L2, HW)
        ▼
store/runs/<run_id>/  (B.npy, ids.json, meta.json)  I-04  store/io.py
        │
        ▼
kernel proiettato k(x, x') = exp(-γ/n · Σ_q ||b_q(x) − b_q(x')||²)   M-01  kernels/projected.py
        │
        ▼
SVM few-shot, One-Class SVM per le classi nuove    M-02, M-03 (ML)
```

L'idea chiave: **la parte quantistica produce una rappresentazione dei dati** (i vettori di Bloch). Tutto quello che viene
dopo è machine learning classico. Così il costo sull'hardware cresce come N (un circuito per punto e per base di misura)
invece che come N² (un circuito per ogni coppia di punti, come col fidelity kernel).

## 2. I concetti, uno per uno

**Feature map (Q-01).** Ogni qubit riceve una feature: il qubit q prende la feature q mod 7. La feature entra come angolo di
due rotazioni, R_Y(θ) e R_Z(θ). Poi porte CZ tra qubit vicini creano entanglement. Il blocco si ripete L volte
("re-uploading"). Le CZ sono applicate prima sulle coppie pari e poi sulle dispari: commutano, quindi lo stato è identico,
ma la profondità a due qubit si dimezza (sul chip Kingston: 4 invece di 9 per n=8, L=2).

**Bandwidth s.** Moltiplica gli angoli. Con s piccolo il circuito resta vicino allo stato iniziale; con s grande gli stati
diventano molto diversi tra loro. Serve a regolare la concentrazione (vedi sotto).

**Vettore di Bloch.** Per ogni qubit, i valori medi di X, Y, Z: tre numeri in [-1, 1] che descrivono completamente lo stato
ridotto di quel qubit. n qubit → 3n numeri per punto.

**Kernel proiettato (M-01).** Confronta due punti attraverso i loro vettori di Bloch. Matematicamente è un kernel RBF sui
vettori di Bloch appiattiti, con γ diviso per n: per questo nel codice è una riga che usa `rbf_kernel` di scikit-learn.
Conseguenza importante: il "modello quantistico" è lo stesso modello delle baseline classiche, applicato a una
rappresentazione diversa. Il confronto è equo per costruzione.

**Fidelity kernel (M-05).** Il kernel "da manuale": k(x, x') = |⟨ψ(x)|ψ(x')⟩|². Serve solo come riferimento in simulazione.

**Concentrazione.** Al crescere di qubit e profondità i valori del fidelity kernel tendono tutti allo stesso numero:
la loro varianza crolla, e con un numero finito di misure diventano indistinguibili. Il modello non vede più i dati
(Thanasilp et al., 2024). Il kernel proiettato ne soffre molto meno. E-01 misura proprio questo sui nostri dati.

## 3. I quattro livelli di esecuzione (Q-03, Q-04, Q-05)

Una sola funzione, `embed(X, cfg, livello)`, costruisce circuito e osservabili e lancia **un solo job** su tutti i punti.
Cambia solo chi esegue:

| Livello | Chi esegue | A cosa serve |
|---|---|---|
| L0 | Aer, valori esatti | sviluppo, scelta della configurazione, mappa di concentrazione |
| L1 | Aer con campionamento | isola il rumore di misura (dipende dalla `precision`) |
| L2 | Aer + modello di rumore del chip Kingston, circuito compilato per il chip vero | prevede l'hardware; è pessimistico perché manca la mitigazione |
| HW | IBM Runtime su ibm_kingston | hardware vero, con mitigazione |

**Precisione e shot.** `precision=0.03` significa errore statistico atteso 0,03 su ogni valore medio: servono
⌈1/0,03²⌉ = 1112 ripetizioni (shot) per circuito. Ogni punto richiede 3 circuiti (basi X, Y, Z).

**Mitigazione degli errori** (solo HW, applicata da IBM sul server):
- *TREX*: corregge gli errori di lettura dei qubit;
- *twirling*: rende casuali gli errori delle porte, così si mediano invece di accumularsi in una direzione;
- *dynamical decoupling*: impulsi sui qubit inattivi per proteggerli dalla decoerenza.

`embed(..., mitigation=False)` la spegne: serve al controllo di H-02. Le opzioni applicate finiscono nei metadati.

## 4. Lo store e la provenienza (I-04)

Ogni corsa è una cartella `store/runs/<run_id>/` con `B.npy` (i vettori di Bloch), `ids.json` (quali punti) e
`meta.json`: livello, configurazione, data, commit git e, per l'hardware, job ID, layout fisico, profondità, calibrazione
dei qubit usati (T1, T2, errore di lettura, errori delle CZ) e secondi QPU misurati. Le corse hardware (`hw-*`)
**non si possono sovrascrivere** e sono committate in git; quelle simulate no.

## 5. Le sicurezze

- `embed(..., "HW")` è un **dry run** se non si passa `submit=True`: niente rete, niente QPU, solo la stima.
- Gli script inviano solo con `--submit`, e la policy (`docs/qpu_policy.md`) chiede il commento `QPU approved` del lead.
- La campagna rifiuta l'invio se la stima supera il budget o se i sottoinsiemi D-05 non sono legati al tag `hw-freeze`.

## 6. Cosa abbiamo misurato: il pilota H-01

Corsa `hw-20261124T005515Z`, 10 punti sintetici, 8 qubit, L=2, ibm_kingston, mitigazione attiva.

| Misura | Valore | Lettura |
|---|---|---|
| secondi QPU | 13 misurati, 13,7 dalla formula IBM | la formula è affidabile; il nostro margine del 30% è prudente |
| errore HW contro esatto | 0,033 (correlazione 0,997) | |
| errore del solo campionamento (L1) | 0,030 | l'hardware mitigato è quasi al limite statistico; errore proprio ≈ 0,014 |
| errore HW contro emulatore L2 | 0,047 | quello atteso da due rumori indipendenti: L2 è un buon sostituto |

Decisione: **GO** per la campagna. Limiti: punti sintetici, circuito poco profondo.

## 7. La campagna principale (H-02) e perché ogni esperimento

| Esperimento | Punti | Domanda a cui risponde |
|---|---|---|
| drift-start / drift-end | 5 + 5 | il chip è cambiato durante la campagna? |
| main (con mitigazione) | 120 | i kernel hardware danno le stesse decisioni di L0? (RQ1, E-04) |
| control (senza mitigazione) | 40 | quanto conta la mitigazione? (studio di mitigazione, Q-06) |
| concentration n = 4, 8, 12 | 3 × 30 | la concentrazione sull'hardware segue quella simulata? (RQ2, E-05) |

Stima ~315 s su 587 disponibili. Tutti i job partono in un solo Batch, sulla stessa calibrazione. Per ogni esperimento lo
script salva anche L0 e L2 degli stessi punti: sono il confronto per l'hardware e il primo deliverable di Q-06.

## 8. Come si esegue

```bash
source ~/.venvs/qspy/bin/activate
python -m pytest tests -q                                      # tutto verde prima di toccare l'hardware
python scripts/check_ibm_access.py                             # processori e budget, senza inviare nulla
python experiments/e01_concentration.py                        # E-01 sui dati veri (minuti); --synthetic per provare
python scripts/qpu_pilot.py [--submit]                         # H-01
python scripts/qpu_campaign.py [--submit]                      # H-02, dopo il tag hw-freeze
```

## 8bis. Cosa ha detto E-01 sui dati veri e cosa abbiamo congelato (L-05)

Mappa di concentrazione su 196 glitch reali del training (figura `results/e01/e01_gravityspy.png`):
- **la bandwidth s conta più di tutto**: a 8 qubit e 2 layer l'allineamento con le etichette passa da 0,36 (s=1) a 0,53
  (s≈0,18) e crolla sopra s=1 (pannello d);
- il fidelity kernel si concentra al crescere di n, il proiettato no (pannello a);
- da 8 a 12 qubit l'allineamento non cambia; a 14–16 cresce di poco (+0,018 sui punti hardware), ma non è ancora
  verificato col rumore del chip: per ora restiamo a 8, validati dal pilota (dettagli in `docs/RISULTATI_QUANTUM_HARDWARE.md`).

Emulazione sui 120 punti hardware veri (Q-06): con s=0,1778 e precisione 0,03 l'allineamento resta 0,53 anche col rumore
del chip, perché i vettori di Bloch dei diversi trigger differiscono 8 volte più del rumore di misura.

Configurazione proposta per l'hardware: **n=8, L=2, s=0,1778, precisione 0,03**, stima 413 s su 587. Tutti i dettagli,
le ipotesi pre-registrate e la procedura di scelta degli iperparametri sono in `docs/decisions.md`; i valori in
`configs/hardware_frozen.yaml`. Diventano definitivi con il tag `hw-freeze`.

## 9. Decisioni prese e deviazioni dalle issue

- Un'unica `embed()` in `quantum/embed.py` invece di tre file in `backends/`: il circuito è lo stesso, cambia solo
  l'esecutore.
- L0 usa Aer in modalità esatta: 70 volte più veloce di `StatevectorEstimator`, stesso risultato.
- L2 usa Aer con il modello di rumore di FakeKingston: l'Estimator di IBM Runtime in locale non riesce a simulare 156
  qubit. Quindi niente `estimator_options()` condivisa e L2 senza mitigazione (pessimistico).
- Store in `.npy` + JSON invece di parquet: un array (N, n, 3), nessuna dipendenza nuova.
- Studio della mitigazione (Q-06): fatto con il controllo senza mitigazione di H-02, su hardware vero; niente ZNE (triplica
  il costo, e il pilota è già vicino al limite statistico).
- L-05 pre-registra la **procedura** di scelta di γ, C e ν, non i valori: si scelgono dopo, sui vettori di Bloch, senza QPU.

## 10. Cosa manca

1. Merge dei dati (D-02, D-03 con la correzione `central_freq`, D-04, D-05).
2. Approvazione di L-05 e tag `hw-freeze`; poi D-05 lega i sottoinsiemi al tag (`python -m data.hardware_subsets --bind-freeze-from .`).
4. H-02 su IBM, poi E-05 (diagnostiche hardware, con G-03) ed E-04 (modelli sui kernel hardware).
