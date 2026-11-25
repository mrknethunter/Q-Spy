# Report tecnico (L-06)

`qspy_report.pdf` è il report: proposta aggiornata con i risultati, positivi e negativi, e i limiti. Ogni numero è
nella tabella dell'appendice A con lo script e il run da cui viene.

```bash
cd report && latexmk -pdf qspy_report.tex        # ricompila il PDF (serve una distribuzione TeX con l'italiano)
python report/make_figures.py                    # rigenera le figure dei risultati dalla root della repo
python report/make_circuit_figures.py            # rigenera le figure dei circuiti (serve pylatexenc)
```

Le figure di `fig/` dipendono da risultati che arrivano con le PR di H-02 (`results/h02`, store `hw-20261125*`),
L-04 (`results/e2e`) ed E-03 (`results/e03`): finché non sono tutte in `main`, `make_figures.py` non trova i file.
`q06.png` e `h01.png` sono copie di `results/q06/q06_main.png` e `results/pilot/h01_pilot.png`.
