# D-02 — Curation Gravity Spy

Questa proposta riprende D-02 dopo la revisione della PR #45, su un nuovo ramo
`feat/D-02-curation-fix` creato da main. Il codice di curation e le regole
scientifiche restano quelli della proposta originale.

## Integrazione

- Codice: `data/curate.py`, funzione `load_curated()` secondo contratto.
- Test D-02: `tests/test_data_curate.py` e `tests/contracts/test_curated_contract.py`.
- Test D-01: usare quelli canonici in `data/tests/`; non aggiungerne copie in `tests/`.
- Guida solo in `docs/GUIDA_D02.md`.
- Dipendenze nel `requirements.txt` condiviso: integrare `numpy>=1.24,<3` e
  `pandas>=2.0,<3` nelle voci esistenti, conservando le altre dipendenze e i loro
  vincoli. Non creare `requirements-d02.txt`.

Lo ZIP di upload contiene solo i file D-02: `requirements.txt` va aggiornato
separatamente sul ramo, prima di presentare la nuova proposta come completa.
Non contiene i dati raw, i moduli degli altri o gli split da sovrascrivere.

## Utilizzo dalla radice del repository

```powershell
python -m pip install -r requirements.txt
python data/download.py
python -m data.curate
python -m pytest -q data/tests/test_data_download.py data/tests/test_data_profile.py tests/test_data_curate.py tests/contracts/test_curated_contract.py
```

Il download riutilizza gli originali validi; il CSV generato in `data/curated/`
non va caricato su GitHub. Non usare quel CSV completo per addestrare senza
applicare i protocolli di split condivisi.

## Convalida della proposta

Sul main ricevuto il 24 novembre 2026 e i file D-02 forniti sono stati verificati
40 test D-01/D-02 (22 subtest), inclusa la rimozione delle copie identiche dei
test D-01. Sul progetto completo ricomposto in precedenza sono passati anche
205 test e 190 subtest. Queste verifiche sono locali su Linux, non una nuova
esecuzione sul ramo remoto o su Windows.

La curation verificata conserva 318.801 righe, di cui 56.509 Fast_Scattering.
I dati raw e i conteggi descritti nei report si riferiscono agli originali
verificati per la proposta, non a dati nuovi scaricati da questo ZIP.
