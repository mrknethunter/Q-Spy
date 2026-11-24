"""Deterministic, independently sized input for D-04 protocol tests."""

import pandas as pd


def curated_fixture() -> pd.DataFrame:
    known = ("Blip", "Extremely_Loud", "Koi_Fish", "Low_Frequency_Burst",
             "Low_Frequency_Lines", "Scattered_Light", "Tomte")
    rows = []
    for label in (*known, "Fast_Scattering"):
        for ifo in ("H1", "L1"):
            for i in range(120):
                rows.append(dict(
                    id=f"{ifo}:{label}:{i:03d}", ifo=ifo, label=label,
                    run=("O3a", "O3b")[i % 2] if label == "Fast_Scattering" else ("O1", "O2", "O3a", "O3b")[i % 4],
                    gps=1_200_000_000. + i, confidence=.99, duration=.1,
                    peak_frequency=100., central_freq=90., bandwidth=20., amplitude=1e-21,
                    snr=10., q_value=8., is_held_out=(label == "Fast_Scattering"),
                ))
    return pd.DataFrame(rows)
