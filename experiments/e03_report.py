"""Independently check E-03 saved predictions and produce its human-readable report."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.splits import DEFAULT_SPLIT_DIR, load_split, verify_dataset
from features.transforms import make_features
from experiments.e03_novelty import HELD_OUT, MODELS, _hash, _id_hash, _json, anomaly_metrics, summarize


def verify_results(directory, split_dir=None, curated_csv=None, progress=False):
    """Recalculate metrics from saved float64 scores; do not trust the tables."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("only complete experiments may be reported")
    for relative, expected in manifest["files"].items():
        if _hash(directory / relative) != expected:
            raise ValueError(f"result hash mismatch: {relative}")
    with gzip.open(directory / "selected_ids.json.gz", "rt") as file:
        selections = json.load(file)
    selected = {(case["held_out"], case["seed"]): case for case in selections}
    audit = {(case["held_out"], case["seed"]): case for case in json.loads((directory / "audit.json").read_text())}
    settings = manifest["settings"]
    keys = {(family, seed) for family in settings["held_outs"] for seed in settings["seeds"]}
    if set(selected) != keys or set(audit) != keys or manifest["completed_cases"] != len(keys):
        raise ValueError("incomplete or duplicated case coverage")
    splits = Path(split_dir) if split_dir is not None else DEFAULT_SPLIT_DIR
    split_manifest = json.loads((splits / "manifest.json").read_text())
    if _hash(splits / "manifest.json") != manifest["split_manifest_sha256"]:
        raise ValueError("split version differs from the experiment")
    entries = {(entry["held_out"], entry["seed"]): entry for entry in split_manifest["splits"]
               if entry["protocol"] == "P2"}
    with np.load(splits / "catalog.npz", allow_pickle=False) as file:
        labels = dict(zip(file["id"].tolist(), file["label"].tolist()))
    curated = None
    if curated_csv is not None:
        curated = pd.read_csv(curated_csv, float_precision="round_trip")
        verify_dataset(curated, split_dir=splits)
        curated = curated.set_index("id", drop=False, verify_integrity=True)
    metrics = pd.read_csv(directory / "e03_metrics.csv", float_precision="round_trip")
    recalculated = 0
    for family in settings["held_outs"]:
        if progress:
            print(f"Verifying {family}: scores, frozen partitions and scaler states", flush=True)
        predictions = pd.read_csv(directory / "predictions" / (family + ".csv.gz"), float_precision="round_trip")
        if set(predictions["seed"]) != set(settings["seeds"]) or not predictions["held_out"].eq(family).all():
            raise ValueError("prediction fold/seed mismatch")
        fixed_test = None
        for seed in settings["seeds"]:
            case = selected[family, seed]
            info = audit[family, seed]
            frozen = load_split("P2", seed, held_out=family, split_dir=splits)
            unknown = info["unknown_classes"]
            if unknown != entries[family, seed]["unknown_classes"]:
                raise ValueError("audited unknown families differ from the frozen manifest")
            for name, partition in zip(("train", "val", "test"), (frozen.train_ids, frozen.val_ids, frozen.test_ids)):
                if len(case[name]) != len(set(case[name])) or not set(case[name]) <= set(partition):
                    raise ValueError("sample is not a subset of the frozen partition")
                if _id_hash(case[name]) != info["id_sha256"][name]:
                    raise ValueError("selected-ID audit mismatch")
                if name != "test" and any(labels[id_] in unknown for id_ in partition):
                    raise ValueError("held-out label in a FULL train/validation partition")
            if set(case["train"]) & set(case["val"]) or set(case["train"]) & set(case["test"]) or set(case["val"]) & set(case["test"]):
                raise ValueError("selected ID leakage")
            if curated is not None:
                _, scaler = make_features(curated.loc[case["train"]])
                if not np.array_equal(scaler.center_, np.array(info["scaler_center"])) or not np.array_equal(scaler.scale_, np.array(info["scaler_scale"])):
                    raise ValueError("scaler does not reproduce from selected training rows")
            test = predictions[predictions["seed"] == seed]
            if test["id"].tolist() != case["test"] or len(test) != len(set(test["id"])):
                raise ValueError("prediction IDs differ from the recorded test selection")
            if test["label"].tolist() != [labels[id_] for id_ in test["id"]]:
                raise ValueError("saved prediction labels differ from the frozen catalog")
            expected_unknown = test["label"].isin(unknown).to_numpy()
            if not np.array_equal(expected_unknown, test["is_unknown"].to_numpy(dtype=bool)):
                raise ValueError("unknown targets disagree with the P2 manifest")
            if fixed_test is not None and fixed_test != case["test"]:
                raise ValueError("test selection changed across training seeds")
            fixed_test = case["test"]
            for model in MODELS:
                model_audit = info["models"][model]
                if model_audit["n_configs_evaluated"] != settings["budget"] or model_audit["validation_fpr"] > .01:
                    raise ValueError("budget or known-validation calibration mismatch")
                scores = test[model + "_score"].to_numpy()
                threshold = model_audit["threshold"]
                if not np.all(test[model + "_threshold"].to_numpy() == threshold):
                    raise ValueError("saved thresholds disagree with the audit")
                for scope in ("target_family", "protocol_unknown"):
                    mask = ~expected_unknown | test["label"].eq(family).to_numpy() if scope == "target_family" else np.ones(len(test), bool)
                    y = expected_unknown[mask].astype(int)
                    values, _ = anomaly_metrics(y, scores[mask])
                    flags = scores[mask] > threshold
                    values.update(tpr_validation_threshold=float(flags[y == 1].mean()),
                                  fpr_validation_threshold=float(flags[y == 0].mean()))
                    for metric, expected in values.items():
                        rows = metrics[(metrics["held_out"] == family) & (metrics["seed"] == seed)
                                       & (metrics["scope"] == scope) & (metrics["model"] == model)
                                       & (metrics["metric"] == metric)]
                        if len(rows) != 1 or not np.isclose(rows.iloc[0]["value"], expected, rtol=0, atol=1e-12):
                            raise ValueError("a metric does not reproduce from saved predictions")
                        if rows.iloc[0]["n_test"] != int(mask.sum()) or not np.isclose(rows.iloc[0]["positive_fraction"], y.mean()):
                            raise ValueError("metric test support/prevalence mismatch")
                        recalculated += 1
        if progress:
            print(f"Verified {family}; metrics recalculated so far: {recalculated}", flush=True)
    if recalculated != len(metrics) or manifest["metric_rows"] != len(metrics):
        raise ValueError("extra or missing metric rows")
    actual_summary, actual_pairs = summarize(metrics, bootstrap=settings["bootstrap"])
    for name, expected in (("e03_summary.csv", actual_summary), ("e03_paired_differences.csv", actual_pairs)):
        actual = pd.read_csv(directory / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    return dict(status="pass", full_frozen_cases_checked=len(keys), models_per_case=len(MODELS),
                metric_values_recalculated=recalculated, all_result_hashes_checked=True,
                full_partition_class_exclusion_checked=True, selected_ids_and_labels_checked=True,
                fixed_test_across_seeds_checked=True, equal_budgets_checked=True,
                scalers_refitted_from_selected_training=curated is not None,
                curated_csv_sha256=_hash(curated_csv) if curated_csv is not None else None,
                paired_bootstrap_recalculated=True, manifest_sha256=_hash(directory / "manifest.json"),
                verifier_sha256=_hash(__file__))


def render_report(directory, verification):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    settings = manifest["settings"]
    summary = pd.read_csv(directory / "e03_summary.csv")
    metric_rows = pd.read_csv(directory / "e03_metrics.csv")
    errors = pd.read_csv(directory / "family_errors.csv")
    neighbors = pd.read_csv(directory / "neighbors_fast_misses.csv")
    pairs = pd.read_csv(directory / "e03_paired_differences.csv")
    primary = summary[(summary["held_out"] == HELD_OUT) & (summary["scope"] == "target_family")]

    def interval(frame, model, metric):
        row = frame[(frame["model"] == model) & (frame["metric"] == metric)].iloc[0]
        return f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"

    data_label = "Gravity Spy D-02" if manifest["data_kind"] == "gravityspy_d02" else "dataset alternativo (non risultati Gravity Spy)"
    lines = [f"# E-03 — Risultati L0: {data_label} (issue #29)", "",
             "## Ambito della misura", "",
             f"Eseguiti **{manifest['completed_cases']} casi P2**, con {len(settings['held_outs'])} famiglie held-out, "
             f"{len(settings['seeds'])} seed e quattro modelli. Input: {manifest['input_rows']:,} righe, "
             "verificate rispetto al bundle D-04 scelto. Questa esecuzione usa sottoinsiemi stratificati: "
             f"al massimo **{settings['max_train']} train / {settings['max_val']} validation / {settings['max_test']} test** "
             "per caso, comuni ai quattro modelli. I risultati NON rappresentano un fit sull'intero training P2.", "",
             f"Embedding esatto L0: {settings['embed_config']['n_qubits']} qubit, "
             f"{settings['embed_config']['layers']} layer, s={settings['embed_config']['s']}, CZ attivi. "
             f"Budget uguale: {settings['budget']} configurazioni per modello, "
             f"{settings['cv_folds']} fold interni. Training, tuning e scaling escludono le famiglie sconosciute; "
             "la soglia operativa usa soltanto validation di classi note. Nessuna esecuzione hardware.", "",
             f"Durata registrata: {manifest['runtime_seconds']/60:.2f} minuti. Versioni: " +
             ", ".join(f"{key} {value}" for key, value in manifest['versions'].items()) + ".", "",
             "## Fast_Scattering: confronto principale", "",
             "Media sui dieci seed e CI bootstrap 95% tra parentesi, condizionati al test fissato. "
             "TPR@1%FPR è una statistica descrittiva della ROC del test, senza interpolare punti non raggiunti. "
             "Le ultime due colonne usano invece la soglia scelta sulla validation nota, senza usare il test per sceglierla.", "",
             "| Modello | AUROC [CI] | AP [CI] | TPR@1%FPR [CI] | TPR soglia validation [CI] | FPR test soglia validation [CI] |",
             "| --- | --- | --- | --- | --- | --- |"]
    main_metrics = ("auroc", "ap", "tpr_at_1pct_fpr", "tpr_validation_threshold", "fpr_validation_threshold")
    for model in MODELS:
        lines.append("| " + model + " | " + " | ".join(interval(primary, model, metric) for metric in main_metrics) + " |")
    baseline = metric_rows[(metric_rows["held_out"] == HELD_OUT) & (metric_rows["scope"] == "target_family")].iloc[0]
    lines += ["", f"Test principale per seed: {int(baseline['n_test'])} eventi, "
              f"frazione positiva {baseline['positive_fraction']:.4f}. AP dipende da questa prevalenza.", "",
              "![ROC Fast Scattering](../results/e03/roc_fast_scattering.png)", "",
              "La fascia grafica è il percentile empirico 2,5–97,5% delle curve dei seed. "
              "Non è una banda di confidenza su eventi indipendenti. La curva è interpolata solo per visualizzazione.", "",
              "## Rotazione della famiglia held-out", "",
              "Ogni riga confronta soltanto la famiglia indicata con le famiglie note di quel fold "
              "(`target_family`), omettendo dal calcolo l'altra famiglia sconosciuta. Fast_Scattering "
              "resta comunque fuori da ogni training/validation. I risultati con entrambe le famiglie "
              "sconosciute positive sono disponibili separatamente come `protocol_unknown` nei CSV.", "",
              "| Famiglia held-out | Modello | AUROC [CI] | AP [CI] | TPR@1%FPR [CI] |",
              "| --- | --- | --- | --- | --- |"]
    for family in settings["held_outs"]:
        group = summary[(summary["held_out"] == family) & (summary["scope"] == "target_family")]
        for model in MODELS:
            lines.append(f"| {family} | {model} | " + " | ".join(interval(group, model, metric) for metric in main_metrics[:3]) + " |")
    lines += ["", "![ROC rotazioni](../results/e03/roc_rotations.png)", "",
              "## Differenze appaiate nel replay principale", "",
              "Differenza projected meno baseline sullo stesso seed. I CI ricampionano gli stessi "
              "indici di seed per i quattro modelli. Sono descrittivi della variabilità di training/tuning "
              "su questo test; non dimostrano superiorità statistica generale o vantaggio quantistico.", "",
              "| Confronto | Metrica | Differenza media [CI 95%] |", "| --- | --- | --- |"]
    for _, row in pairs[(pairs["held_out"] == HELD_OUT) & (pairs["scope"] == "target_family")
                         & pairs["metric"].isin(main_metrics[:3])].iterrows():
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")
    lines += ["", "## Nota fisica e analisi degli errori", "",
              "Un OC-SVM/Isolation Forest/LOF decide noto contro anomalo: non predice quale famiglia "
              "nota sia Fast_Scattering. Perciò separiamo le famiglie note erroneamente segnalate come "
              "anomale dai vicini noti degli eventi Fast_Scattering mancati.", "",
              "### Famiglie note più spesso segnalate come anomalie", "",
              "Falso positivo alla soglia calibrata su validation, nel fold Fast_Scattering. "
              "Le tre famiglie con tasso più alto per modello sono mostrate sotto. Le decisioni "
              "sono sommate sui seed, che riusano gli stessi eventi: questi denominatori NON sono "
              "numeri di eventi fisici indipendenti.", "",
              "| Modello | Famiglia nota | Decisioni anomale / decisioni totali | Tasso |",
              "| --- | --- | --- | --- |"]
    normal = errors[(errors["held_out"] == HELD_OUT) & ~errors["is_unknown"]]
    for model in MODELS:
        group = normal[normal["model"] == model].groupby("label")[["flagged", "n"]].sum()
        group["rate"] = group["flagged"] / group["n"]
        for label, row in group.sort_values("rate", ascending=False).head(3).iterrows():
            lines.append(f"| {model} | {label} | {int(row['flagged'])} / {int(row['n'])} | {row['rate']:.4f} |")
    lines += ["", "### Famiglie vicine ai Fast_Scattering non rilevati", "",
              "Per ogni Fast_Scattering sotto la soglia, cerchiamo il vicino più prossimo nel "
              "training noto, nello spazio classico delle sette feature trasformate (distanza "
              "euclidea, non nello spazio quantistico). Le due famiglie più frequenti per modello "
              "sono un indizio di sovrapposizione delle feature, non una matrice di classificazione. "
              "La frequenza dei vicini dipende anche dalla numerosità e densità delle famiglie nel "
              "training sbilanciato; da sola non prova una maggiore confusione morfologica.", "",
              "| Modello | Famiglia del vicino noto | Fast_Scattering mancati associati | Quota dei mancati |",
              "| --- | --- | --- | --- |"]
    for model in MODELS:
        group = neighbors[neighbors["model"] == model].groupby("nearest_known_family")["missed_fast"].sum().sort_values(ascending=False)
        for label, count in group.head(2).items():
            lines.append(f"| {model} | {label} | {int(count)} | {count/group.sum():.4f} |")
    nearest_families = neighbors.groupby("nearest_known_family")["missed_fast"].sum().sort_values(ascending=False).head(2).index.tolist()
    lines += ["", "In questa esecuzione, aggregando le decisioni dei seed e dei quattro modelli, "
              "le famiglie dei vicini più frequenti sono **" + " e ".join(nearest_families) + "**. "
              "È il risultato misurato della diagnostica di vicinato, non un'etichetta predetta dai detector."]
    if "Tomte" in nearest_families:
        lines += ["", "Per Tomte una spiegazione plausibile è la sovrapposizione di descrittori a bassa "
                  "frequenza: Glanzer et al. descrivono eventi Tomte spesso attorno a 40 Hz e "
                  "Fast Scattering O3 quasi sempre sotto circa 60 Hz [1]. Questo non stabilisce "
                  "una causa comune: lo stesso articolo segnala che l'accoppiamento fisico dei "
                  "Tomte non è identificato. Lo scarto di informazione morfologica passando "
                  "dalle immagini a sette numeri è una possibile spiegazione della vicinanza, "
                  "non una verifica sperimentale del meccanismo."]
    lines += ["", "**Interpretazione, distinta dal dato misurato.** Fast_Scattering e Scattered_Light "
              "sono popolazioni legate allo scattering; la letteratura collega Fast Scatter a moto "
              "di superfici ottiche e disturbi sismici a frequenze superiori a 1 Hz [1,2]. Una "
              "vicinanza a Scattered_Light è quindi fisicamente plausibile, ma deve essere verificata "
              "nei numeri sopra e non assunta come risultato. Vicini di altre famiglie indicano "
              "somiglianza dei descrittori scalari, non l'identificazione dello stesso accoppiamento fisico.", "",
              "Le sette feature riassumono durata, frequenze, banda, ampiezza, SNR e Q; non contengono "
              "gli archi tempo-frequenza delle immagini usate dalle classificazioni Gravity Spy. "
              "La trasformazione log/RobustScaler/clipping può inoltre rendere uguali gli estremi di "
              "popolazioni diverse. L'embedding R3 è una trasformazione di queste stesse feature: "
              "non aggiunge dati sui canali ambientali o sulla morfologia delle immagini. Sono "
              "spiegazioni possibili dei fallimenti, non una misura causale eseguita qui.", "",
              "Nel dataset curato Fast_Scattering è quasi tutto L1 (55.229 su 56.509) e soltanto O3; "
              "le famiglie note mescolano detector e run. La letteratura riporta differenze fra siti "
              "e una maggiore prevalenza di Fast Scattering a Livingston [1]. Pur senza usare ifo/run "
              "come input, le feature possono riflettere questi domini. La selezione stratificata "
              "preserva, non elimina, questo confondimento. P2 non equivale a una prova di "
              "generalizzazione temporale O3a→O3b o di trasferimento tra detector.", "",
              "## Controlli e limiti", "",
              f"Verifica indipendente dai file delle predizioni: **{verification['full_frozen_cases_checked']} casi**, "
              f"**{verification['metric_values_recalculated']} valori metrici** ricalcolati. "
              "Controllati hash degli output, appartenenza ai partition ID originali, esclusione "
              "delle famiglie sconosciute da TUTTO il train/validation P2, etichette del catalogo, "
              "test fisso fra seed, budget uguali e CI/differenze appaiate ricalcolate.", "",
              "La configurazione è resource-bounded e non include un'analisi di sensibilità ai cap "
              "o ai parametri del circuito. Gli intervalli sono condizionati al test e non includono "
              "l'incertezza della scelta del circuito, del dataset o delle etichette ML. Le rotazioni "
              "condividono dati e non sono otto dataset indipendenti. Le etichette sono quelle ML "
              "della release, non verità fisica verificata caso per caso.", "",
              "Metodo, comandi di riproduzione e significato di ogni output: [E03.md](E03.md). "
              "Suite di test e provenienza dello ZIP di base: [E03_validation.json](E03_validation.json).", "",
              "## Riferimenti primari", "",
              "1. Glanzer et al. (2023), *Data quality up to the third observing run of advanced "
              "LIGO: Gravity Spy glitch classifications*, "
              "[DOI 10.1088/1361-6382/acb633](https://doi.org/10.1088/1361-6382/acb633).", "",
              "2. Soni et al., *Modeling and Reduction of High Frequency Scatter Noise at LIGO "
              "Livingston*, [arXiv:2311.05730](https://arxiv.org/abs/2311.05730).", ""]
    if len(primary):
        performance = primary[primary["metric"] == "auroc"].set_index("model")["mean"]
        interpretation = ("\n## Lettura del confronto\n\nIn questa configurazione il miglior AUROC medio è di **" +
                          performance.idxmax() + "**. Il confronto con le baseline complete è essenziale: "
                          "un miglioramento sul solo OC-SVM RBF non dimostra un vantaggio sui migliori "
                          "metodi classici né un vantaggio computazionale.\n")
        if performance["rbf_ocsvm"] < .5:
            interpretation += ("\nL'AUROC dell'OC-SVM RBF è sotto 0,5. Il segno dello score resta quello "
                               "del contratto condiviso (più alto = più anomalo): non viene invertito "
                               "a posteriori usando le etichette held-out. Questo risultato segnala "
                               "un fallimento di ranking nel replay, non viene corretto ottimizzando sul test.\n")
        lines.insert(lines.index("## Controlli e limiti"), interpretation)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--curated-csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/E03_results.md"))
    args = parser.parse_args()
    result = verify_results(args.directory, args.split_dir, args.curated_csv, progress=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(args.directory, result), encoding="utf-8")
    args.output.with_name("E03_validation.json").write_text(_json(result), encoding="utf-8")
    print(_json(result))
