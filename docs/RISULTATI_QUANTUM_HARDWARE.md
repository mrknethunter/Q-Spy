# Risultati: parte quantistica e hardware

Questo documento raccoglie tutto ciò che abbiamo misurato finora, in simulazione e sull'hardware IBM, con le figure
spiegate pannello per pannello. Ogni numero esce da uno script della repo e si può rigenerare con i comandi della
sezione 5. I concetti (vettori di Bloch, kernel proiettato, concentrazione, livelli L0–HW) sono spiegati in
`docs/GUIDA_QUANTUM_HARDWARE.md`; le scelte che ne derivano in `docs/decisions.md`.

Una misura che ricorre ovunque, la **kernel-target alignment (KTA)**: dice quanto il kernel "vede" le classi, cioè quanto
i punti della stessa classe risultano simili tra loro e diversi da quelli delle altre classi. Va da 0 (il kernel ignora le
classi) a 1 (le separa perfettamente). È una misura del kernel, non di un classificatore: dice quanto è promettente una
configurazione prima ancora di allenare una SVM.

---

## 1. Pilota hardware (H-01)

**Domanda.** Il nostro codice funziona sull'hardware vero, quanto costa davvero, e quanto rumore aggiunge il chip?

**Come.** 10 punti sintetici, 8 qubit, 2 layer, s = 1, precisione 0,03, su `ibm_kingston` con mitigazione (TREX,
twirling, dynamical decoupling). Gli stessi punti calcolati anche in modo esatto (L0), con solo rumore di misura (L1) e
con il modello di rumore di Kingston (L2). Corsa `hw-20261124T005515Z`, salvata in `store/runs/`.

**Numeri.**

| | Valore |
|---|---|
| Secondi QPU misurati / previsti dalla formula IBM / nostra stima con margine | 13 / 13,7 / 17,8 |
| Errore (RMSE) hardware contro esatto | 0,033 |
| Errore del solo campionamento (L1) | 0,030 |
| Errore dell'emulatore L2 contro esatto | 0,032 |
| Errore medio dell'hardware (sbilanciamento) | −0,0065 ± 0,0021 |

![Pilota H-01](../results/pilot/h01_pilot.png)

**Come leggere la figura** (`results/pilot/h01_pilot.png`):
- **(a)** ogni punto è uno dei 240 valori misurati (10 punti × 8 qubit × 3 assi X, Y, Z): in ascissa il valore esatto,
  in ordinata quello dell'hardware (rosso) o dell'emulatore (arancio). Stanno tutti sulla diagonale, da −1 a +1:
  l'hardware riproduce l'intero intervallo senza distorsioni visibili.
- **(b)** la distribuzione degli errori (misurato meno esatto) per i tre livelli. Le tre curve sono quasi sovrapposte e
  larghe circa ±0,05: l'hardware mitigato si comporta come il solo rumore statistico delle 1112 ripetizioni.
- **(c)** le barre sono l'errore dell'hardware per ciascun qubit fisico del chip; la linea tratteggiata è l'errore di
  lettura dichiarato dalla calibrazione di IBM al momento della corsa. Il qubit 72 ha l'errore di lettura peggiore (5,2%)
  ma non è il più impreciso: la correzione TREX funziona. Con 30 valori per qubit, ogni barra ha un'incertezza di circa
  ±0,004, quindi le differenze tra qubit sono in parte solo statistiche.

**Conclusioni.** La formula di costo IBM è affidabile (sbaglia di meno di un secondo). L'hardware mitigato è quasi al
limite del rumore di misura: il suo errore proprio è circa 0,014. C'è un piccolo sbilanciamento sistematico verso il
basso, compatibile con la decoerenza residua che "accorcia" i vettori di Bloch; è 5 volte più piccolo del rumore e
l'emulatore L2 lo riproduce in parte. L'emulatore è quindi un buon sostituto dell'hardware per preparare la campagna.

**Limiti.** Punti sintetici (verifica la macchina, non la classificazione), un solo job, circuito poco profondo.

---

## 2. Mappa di concentrazione sui dati veri (E-01)

**Domanda.** Con quale numero di qubit n, profondità L e bandwidth s il kernel rappresenta meglio i glitch, senza
"concentrarsi" (cioè senza diventare quasi costante e quindi inutile)?

**Come.** 196 glitch veri dal training dello split P2 (28 per ognuna delle 7 classi note), stessa normalizzazione che
userà la campagna. Simulazione esatta (L0) di tutta la griglia n ∈ {4, 6, …, 16}, L ∈ {1, 2, 4, 8}, 9 valori di s tra
0,03 e 3: 252 configurazioni per il kernel proiettato (γ = 1) e 216 per il fidelity kernel (fino a 14 qubit).

![E-01](../results/e01/e01_gravityspy.png)

**Come leggere la figura** (`results/e01/e01_gravityspy.png`):
- **(a) concentrazione contro n**, a s = 1: in ordinata la varianza dei valori del kernel fuori dalla diagonale (scala
  logaritmica). Se scende verso zero, il kernel dà lo stesso valore a tutte le coppie e non distingue più nulla. Il
  fidelity kernel (rosso) scende di circa 8 volte da 4 a 14 qubit; il proiettato (verde) resta piatto. È la
  concentrazione prevista dalla teoria, più lieve del caso peggiore perché le feature distinte sono solo 7 e oltre 7
  qubit si ripetono.
- **(b) concentrazione contro bandwidth**, a 14 qubit e L = 2: con s troppo piccolo tutti gli stati restano vicini a
  quello iniziale e la varianza crolla per entrambi i kernel; con s grande il fidelity kernel crolla di nuovo, il
  proiettato no.
- **(c) allineamento contro n**, a s = 1: la KTA cresce fino a 8 qubit e poi si stabilizza. Servono almeno 7 qubit per
  usare tutte e 7 le feature.
- **(d) allineamento contro bandwidth**, a n = 8 e L = 2: **è la figura chiave**. La KTA ha un picco netto tra s ≈ 0,18
  e 0,56 e crolla sopra s = 1. A s = 0,1778 il proiettato vale 0,534 contro 0,357 a s = 1 (il valore usato nel pilota).

**Conclusioni.**
- La bandwidth conta più di qubit e profondità: scegliere s bene vale il 50% di allineamento in più.
- Oltre 8 qubit non si guadagna nulla (0,534 a n = 8, 0,535 a n = 10, 0,530 a n = 12), e più qubit significano più
  rumore sull'hardware. Circuiti più profondi (L = 4, 8) non migliorano.
- **Onestamente:** a n = 8 il fidelity kernel si allinea un po' meglio del proiettato (0,556 contro 0,534). Il
  proiettato resta la scelta per l'hardware per il costo: 360 circuiti per 120 punti invece di 7140.

**Limiti.** Un solo campione di 196 punti e un solo valore di γ per il kernel proiettato; la KTA misura la qualità del
kernel, non l'accuratezza di un classificatore (che arriverà con E-02/E-03).

---

## 3. Previsione del rumore sui punti hardware veri (Q-06)

**Domanda.** Con la configurazione scelta, il rumore dell'hardware cancellerà le differenze tra glitch che servono per
classificare? E quanta precisione serve davvero?

**Come.** I 120 punti "main" della campagna (15 per classe, Fast_Scattering compresa, mai visti in training), 8 qubit,
2 layer. Per quattro combinazioni di bandwidth e precisione: esatto (L0), solo rumore di misura (L1), modello di rumore
di Kingston (L2).

![Q-06](../results/q06/q06_main.png)

**Come leggere la figura** (`results/q06/q06_main.png`):
- **(a)** la KTA dei tre livelli per ogni configurazione. Con s = 0,1778 e precisione 0,03 le tre barre sono uguali
  (0,531 / 0,532 / 0,530): il rumore non toglie nulla. A precisione 0,1 il calo diventa visibile (0,512 in L2). Con s = 1
  tutto parte già basso (0,32).
- **(b)** l'errore dei vettori di Bloch rispetto all'esatto; il trattino nero è la precisione richiesta. L'errore
  coincide sempre con la precisione chiesta, anche col rumore del chip: le stime di costo e di qualità si possono fare
  partendo dalla precisione.

**Conclusioni.** I vettori di Bloch di glitch diversi differiscono in media di 0,23, otto volte il rumore di misura:
la configurazione scelta è robusta. La KTA sui punti mai visti (0,531) coincide con quella del campione di training di
E-01 (0,534), quindi la scelta generalizza. Corse ripetute variano di circa ±0,005: precisione 0,05 è già dentro questa
variabilità, 0,1 no. Teniamo 0,03, validata dal pilota e compatibile con il budget.

**Limiti.** L2 non include la mitigazione del server IBM, quindi è pessimistico; il confronto con e senza mitigazione
sull'hardware vero lo darà il gruppo di controllo di H-02.

---

## 4. Cosa ne deriva

Configurazione proposta per la campagna, in `configs/hardware_frozen.yaml` e motivata in `docs/decisions.md`:
**8 qubit, 2 layer, s = 0,1778, precisione 0,03**, circa 413 secondi QPU su 587 disponibili.

## 5. Come riprodurre

```bash
source ~/.venvs/qspy/bin/activate
python experiments/h01_pilot_plot.py              # figura del pilota, dallo store (niente QPU)
python experiments/e01_concentration.py           # E-01: dati veri, ~15 minuti; --synthetic per una prova
python experiments/q06_noise_prediction.py        # Q-06: ~1-2 minuti
```

E-01 e Q-06 richiedono i dati scaricati (`python data/download.py`) e i moduli di D-02 e D-03 in `main`.
