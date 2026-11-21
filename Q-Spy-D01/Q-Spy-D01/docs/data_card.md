# Gravity Spy raw data card (D-01)

## Scope and provenance

Pinned release: [Zenodo 5649212, v1.0.0](https://zenodo.org/records/5649212), DOI [10.5281/zenodo.5649212](https://doi.org/10.5281/zenodo.5649212). Published in 2021; companion paper: Glanzer et al. (2023). These are tabular machine-learning classifications and Omicron metadata, not detector strain time series, volunteer votes, or downloaded spectrogram images. H1 = LIGO Hanford; L1 = LIGO Livingston. O3 is split into O3a and O3b.

This card profiles **676,613 unfiltered rows** in all eight CSVs. Counts include low-confidence events, No_Glitch and None_of_the_Above wherever present. No deduplication, label remapping, normalization or class selection is applied (D-02 handles curation).

## Reproduce

Python 3.10+; no third-party packages. From the repository root:

```sh
python data/download.py
```

This one command downloads all eight original CSVs into `data/raw/` (742,195,211 bytes total), checks size and the publisher's pinned MD5 checksums, and skips valid local files on subsequent runs. A damaged file is replaced only after a verified download; failed transfers exit nonzero. Allow about 1 GB free space for the dataset and a temporary file. Raw files are git-ignored. The pinned `data/manifest.json` avoids silently following future release changes. MD5 detects accidental corruption; HTTPS and the pinned release supply provenance, and MD5 is not a cryptographic authenticity guarantee.

```sh
# Offline integrity check
python data/download.py --verify-only
# Regenerate this card and exact machine-readable statistics
python data/profile.py
# Download a single file for a smoke check
python data/download.py --files H1_O1.csv
```

Default paths are relative to the scripts, so invocation also works from another working directory. `--output-dir` selects a raw download location; pass the same path to `profile.py --raw-dir`. The profiler refuses incomplete or checksum-mismatched datasets. All ranges below are observed finite min/max, not imposed selection limits. Markdown values use 12 significant digits; `docs/data_summary.json` retains Python float precision.

## Files and row counts

| File | Rows | Bytes | MD5 |
| --- | --- | --- | --- |
| H1_O1.csv | 15305 | 15963563 | 91963313b1574e083bc58915e0aa8ca1 |
| H1_O2.csv | 70871 | 73098153 | 1c3eb9432107c6fee671cc76ad632e69 |
| H1_O3a.csv | 80763 | 90238691 | 29aea278b622cd97496971f7c07f7d6a |
| H1_O3b.csv | 99568 | 111575139 | 590290fe3c7ddf8cd9fa85c3a09cd5e4 |
| L1_O1.csv | 26412 | 27212877 | 34c7d08070f302d3b366767610748f40 |
| L1_O2.csv | 63501 | 65785621 | df0a831cb79509e2aae38a4651c796fc |
| L1_O3a.csv | 127500 | 142258875 | 3d3d0409817499c84fb2e0a2b211b5a5 |
| L1_O3b.csv | 192693 | 216062292 | 90d0bc16448bca72818ed9abf267979c |

## Seven numeric features

**Selection is provisional.** The supplied D-01 screenshot requires seven features but does not name them, and the supplied repository snapshot contains no proposal or feature contract. The following seven non-timestamp Omicron parameters form a proposed input vector, in this exact order. The team must confirm it against the proposal before D-01's seven-feature acceptance criterion is signed off. All source columns are retained regardless of this choice. `ml_label` is the target; confidence scores, identifiers and image URLs are excluded from this proposed input vector.

| Column | Unit | Meaning | Raw min | Raw max | Empty | Invalid | NaN/Inf | <=0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `duration` | s | Duration reported by Omicron | 0.000730000028852 | 64.5 | 0 | 0 | 0 | 0 |
| `peak_frequency` | Hz | Frequency of the peak Omicron tile | 10.0087795258 | 2047.10644531 | 0 | 0 | 0 | 0 |
| `central_freq` | Hz | Central frequency reported by Omicron | 7.84056997299 | 4933.75292969 | 0 | 0 | 0 | 0 |
| `bandwidth` | Hz | Frequency extent reported by Omicron | 0.131430000067 | 7946.48291016 | 0 | 0 | 0 | 0 |
| `amplitude` | pipeline units (see note) | Omicron amplitude, retained without conversion | 1.59848005407e-23 | 1.4298899502e-16 | 0 | 0 | 0 | 0 |
| `snr` | dimensionless | Omicron signal-to-noise ratio | 7.5 | 156629.9375 | 0 | 0 | 0 | 0 |
| `q_value` | dimensionless | Omicron quality factor Q | 5.65685 | 1e+20 | 0 | 0 | 0 | 0 |

Amplitude caveat: the release identifies `amplitude` as Omicron output but does not specify its physical normalization/unit. The underlying channel is calibrated gravitational-wave strain; that alone does not establish whether this derived amplitude is strain, an integrated quantity, or another normalization. This card therefore keeps the published values and marks the unit as pipeline units pending confirmation from the release authors or the exact pipeline configuration. Do not substitute a strain spectral-density unit without evidence.

`central_freq` and `bandwidth` describe the reported event extent and can exceed the release's stated 10–2048 Hz peak-frequency selection window. Keep these distinct from `peak_frequency`. Nonpositive and non-finite inputs are reported, not silently removed. Feature transforms and scaling must be fitted on training data in later tasks.

## Observed quality flags

| File | q_value == 1e20 | ml_confidence == 0 | snr == 7.5 |
| --- | --- | --- | --- |
| H1_O1.csv | 0 | 0 | 0 |
| H1_O2.csv | 0 | 0 | 0 |
| H1_O3a.csv | 0 | 0 | 0 |
| H1_O3b.csv | 0 | 0 | 0 |
| L1_O1.csv | 14047 | 52 | 0 |
| L1_O2.csv | 36745 | 0 | 0 |
| L1_O3a.csv | 0 | 0 | 1 |
| L1_O3b.csv | 0 | 0 | 0 |

The repeated Q value `1e20` is an extreme raw value that may be a sentinel; its meaning is not established by the release description. It is finite and therefore is not counted as NaN/Inf above. Preserve it in raw data and decide its treatment in D-02 before scaling or fitting a model. Zero confidence is also retained. One or more exact boundary SNR values, if listed, should not be silently removed on the basis of the release's textual SNR > 7.5 selection statement.

## Complete observed schema

| Column | Unit / type | Meaning | Present in files |
| --- | --- | --- | --- |
| `event_time` | GPS s | Event timestamp, including a fractional part | 8 |
| `ifo` | category | Detector identifier H1 or L1 | 8 |
| `peak_time` | GPS s | Integer seconds of trigger peak | 8 |
| `peak_time_ns` | ns | Nanosecond component of peak time | 8 |
| `start_time` | GPS s | Integer seconds of trigger start | 8 |
| `start_time_ns` | ns | Nanosecond component of start time | 8 |
| `duration` | s | Duration reported by Omicron | 8 |
| `peak_frequency` | Hz | Frequency of the peak Omicron tile | 8 |
| `central_freq` | Hz | Central frequency reported by Omicron | 8 |
| `bandwidth` | Hz | Frequency extent reported by Omicron | 8 |
| `channel` | text | Analysed strain channel | 8 |
| `amplitude` | pipeline units (see note) | Omicron amplitude, retained without conversion | 8 |
| `snr` | dimensionless | Omicron signal-to-noise ratio | 8 |
| `q_value` | dimensionless | Omicron quality factor Q | 8 |
| `gravityspy_id` | identifier | Published Gravity Spy event identifier | 8 |
| `1400Ripples` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `1080Lines` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Air_Compressor` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Blip` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Chirp` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Extremely_Loud` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Helix` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Koi_Fish` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Light_Modulation` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Low_Frequency_Burst` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Low_Frequency_Lines` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `No_Glitch` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `None_of_the_Above` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Paired_Doves` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Power_Line` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Repeating_Blips` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Scattered_Light` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Scratchy` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Tomte` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Violin_Mode` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Wandering_Line` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `Whistle` | dimensionless [0,1] | Original 22-class classifier score | 8 |
| `ml_label` | category | Predicted morphology label; count these strings as published | 8 |
| `ml_confidence` | dimensionless [0,1] | Confidence attached to ml_label; not calibrated certainty | 8 |
| `url1` | URL | Omega-scan image link, 0.5 s window | 8 |
| `url2` | URL | Omega-scan image link, 1 s window | 8 |
| `url3` | URL | Omega-scan image link, 2 s window | 8 |
| `url4` | URL | Omega-scan image link, 4 s window | 8 |

Times are GPS, not Unix/UTC. Keep the separate integer seconds and nanoseconds when exact time matching matters; a floating-point `event_time` can lose subsecond precision. The 22 class-score columns are the original network outputs even in O3; they must not be assumed to be a score vector aligned with all O3 labels. The upstream description says None_of_the_Above was removed for O3 but also lists it among O3 labels; the observed counts below are authoritative for this release. `ml_label`/`ml_confidence` are used as published, without reconstructing them from the original score columns.

## Class counts by detector and run

| ml_label | H1_O1 | H1_O2 | H1_O3a | H1_O3b | L1_O1 | L1_O2 | L1_O3a | L1_O3b | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1080Lines | 15 | 31684 | 23 | 354 | 13 | 31 | 161 | 1343 | 33624 |
| 1400Ripples | 3 | 838 | 113 | 208 | 1439 | 26 | 8340 | 185 | 11152 |
| Air_Compressor | 128 | 116 | 441 | 139 | 24 | 127 | 931 | 2164 | 4070 |
| Blip | 2372 | 8461 | 5094 | 4123 | 868 | 7584 | 2895 | 3321 | 34718 |
| Blip_Low_Frequency | 0 | 0 | 2583 | 2845 | 0 | 0 | 11473 | 11945 | 28846 |
| Chirp | 30 | 11 | 32 | 23 | 33 | 15 | 10 | 20 | 174 |
| Extremely_Loud | 431 | 2475 | 11388 | 7082 | 195 | 1189 | 5897 | 4412 | 33069 |
| Fast_Scattering | 0 | 0 | 583 | 2154 | 0 | 0 | 30219 | 52374 | 85330 |
| Helix | 8 | 115 | 98 | 44 | 274 | 65 | 142 | 106 | 852 |
| Koi_Fish | 751 | 3078 | 9334 | 5049 | 270 | 3112 | 7534 | 4744 | 33872 |
| Light_Modulation | 642 | 124 | 108 | 93 | 21 | 111 | 428 | 372 | 1899 |
| Low_Frequency_Burst | 1712 | 1850 | 19212 | 2785 | 7267 | 7182 | 2375 | 4458 | 46841 |
| Low_Frequency_Lines | 2566 | 3700 | 2330 | 2702 | 9413 | 27402 | 2901 | 11938 | 62952 |
| No_Glitch | 850 | 7928 | 8377 | 1075 | 2424 | 1692 | 12688 | 4201 | 39235 |
| None_of_the_Above | 82 | 782 | 14 | 0 | 110 | 399 | 10 | 0 | 1397 |
| Paired_Doves | 228 | 440 | 110 | 265 | 11 | 369 | 1442 | 3066 | 5931 |
| Power_Line | 863 | 836 | 225 | 144 | 1538 | 1230 | 1484 | 627 | 6947 |
| Repeating_Blips | 332 | 1496 | 1430 | 972 | 91 | 339 | 414 | 806 | 5880 |
| Scattered_Light | 2968 | 2698 | 10376 | 64968 | 727 | 9939 | 9208 | 52935 | 153819 |
| Scratchy | 983 | 1619 | 602 | 276 | 1280 | 82 | 312 | 153 | 5307 |
| Tomte | 100 | 1192 | 1017 | 1772 | 54 | 1073 | 24262 | 26562 | 56032 |
| Violin_Mode | 137 | 582 | 285 | 744 | 6 | 75 | 1417 | 641 | 3887 |
| Wandering_Line | 46 | 61 | 39 | 4 | 1 | 17 | 41 | 32 | 241 |
| Whistle | 58 | 785 | 6949 | 1747 | 353 | 1442 | 2916 | 6288 | 20538 |
| TOTAL | 15305 | 70871 | 80763 | 99568 | 26412 | 63501 | 127500 | 192693 | 676613 |

## ML confidence distribution

Quantiles use sorted finite values and linear interpolation at `(n-1)*p`. These are classifier scores, not independently calibrated probabilities.

| File | Min | P10 | Median | P90 | P99 | Max | Mean | Empty | Invalid | NaN/Inf | Outside [0,1] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1_O1.csv | 0.21672052145 | 0.656330227852 | 0.98961108923 | 0.999996638298 | 1 | 1 | 0.910455540195 | 0 | 0 | 0 | 0 |
| H1_O2.csv | 0.147637084126 | 0.551173388958 | 0.987898230553 | 0.999981760979 | 0.999999821186 | 1 | 0.879781931769 | 0 | 0 | 0 | 0 |
| H1_O3a.csv | 0.173653423786 | 0.747247564793 | 0.997213065624 | 0.999999165535 | 1 | 1 | 0.935978532173 | 0 | 0 | 0 | 0 |
| H1_O3b.csv | 0.172644808888 | 0.772720068693 | 0.9959025383 | 0.999867320061 | 0.999998927116 | 1 | 0.941041484017 | 0 | 0 | 0 | 0 |
| L1_O1.csv | 0 | 0.638051146269 | 0.973272860051 | 0.999598801136 | 0.999999880791 | 1 | 0.89483273093 | 0 | 0 | 0 | 0 |
| L1_O2.csv | 0.158588960767 | 0.610067725182 | 0.968358874321 | 0.999946594238 | 0.999999880791 | 1 | 0.884467971917 | 0 | 0 | 0 | 0 |
| L1_O3a.csv | 0.141466319561 | 0.603103166819 | 0.984488785267 | 0.999993562698 | 0.999999880791 | 1 | 0.889833579221 | 0 | 0 | 0 | 0 |
| L1_O3b.csv | 0.169134914875 | 0.618798947334 | 0.991668105125 | 0.999995112419 | 1 | 1 | 0.900375191397 | 0 | 0 | 0 | 0 |

| File | [0,0.5) | [0.5,0.8) | [0.8,0.9) | [0.9,0.95) | [0.95,0.99) | [0.99,1] |
| --- | --- | --- | --- | --- | --- | --- |
| H1_O1.csv | 436 | 2288 | 1222 | 1123 | 2646 | 7590 |
| H1_O2.csv | 5071 | 11691 | 5445 | 4578 | 9742 | 34344 |
| H1_O3a.csv | 1275 | 8713 | 5318 | 4934 | 11464 | 49059 |
| H1_O3b.csv | 1128 | 10035 | 6081 | 6233 | 15890 | 60201 |
| L1_O1.csv | 596 | 4836 | 2728 | 2638 | 6226 | 9388 |
| L1_O2.csv | 2388 | 11986 | 6736 | 6471 | 13842 | 22078 |
| L1_O3a.csv | 5382 | 23222 | 11814 | 9728 | 18086 | 59268 |
| L1_O3b.csv | 7583 | 31304 | 15868 | 13017 | 25723 | 99198 |

## Feature ranges by detector and run

| File | Feature | Raw min | Raw max |
| --- | --- | --- | --- |
| H1_O1.csv | duration | 0.00243999995291 | 64.5 |
| H1_O2.csv | duration | 0.00098 | 60 |
| H1_O3a.csv | duration | 0.000980000011623 | 39.5 |
| H1_O3b.csv | duration | 0.000730000028852 | 59 |
| L1_O1.csv | duration | 0.00390999997035 | 54.21875 |
| L1_O2.csv | duration | 0.00147 | 60 |
| L1_O3a.csv | duration | 0.000980000011623 | 49.5 |
| L1_O3b.csv | duration | 0.000980000011623 | 60 |
| H1_O1.csv | peak_frequency | 10.0717697144 | 2027.84204102 |
| H1_O2.csv | peak_frequency | 10.0087795258 | 2046.2811 |
| H1_O3a.csv | peak_frequency | 10.0895795822 | 2046.28112793 |
| H1_O3b.csv | peak_frequency | 10.0473403931 | 2032.88879395 |
| L1_O1.csv | peak_frequency | 10.0717697144 | 2047.10644531 |
| L1_O2.csv | peak_frequency | 10.0087795258 | 2046.28112793 |
| L1_O3a.csv | peak_frequency | 10.0473403931 | 2046.28112793 |
| L1_O3b.csv | peak_frequency | 10.0473403931 | 2032.88879395 |
| H1_O1.csv | central_freq | 8.96879959106 | 4037.09521484 |
| H1_O2.csv | central_freq | 7.95143985748 | 4838.56982422 |
| H1_O3a.csv | central_freq | 9.65067005157 | 4711.93847656 |
| H1_O3b.csv | central_freq | 10.6750898361 | 4780.50683594 |
| L1_O1.csv | central_freq | 7.84056997299 | 4727.69189453 |
| L1_O2.csv | central_freq | 10.0566101074 | 4684.56445312 |
| L1_O3a.csv | central_freq | 11.048789978 | 4794.63330078 |
| L1_O3b.csv | central_freq | 10.3268604279 | 4933.75292969 |
| H1_O1.csv | bandwidth | 0.401410013437 | 7946.48291016 |
| H1_O2.csv | bandwidth | 0.79646 | 7946.48291016 |
| H1_O3a.csv | bandwidth | 0.403699994087 | 7946.48291016 |
| H1_O3b.csv | bandwidth | 0.508459985256 | 7940.48291016 |
| L1_O1.csv | bandwidth | 0.131430000067 | 7946.48291016 |
| L1_O2.csv | bandwidth | 0.495629996061 | 7946.48291016 |
| L1_O3a.csv | bandwidth | 0.527719974518 | 7946.48291016 |
| L1_O3b.csv | bandwidth | 0.474209994078 | 7940.48291016 |
| H1_O1.csv | amplitude | 4.07195989105e-23 | 2.70256995673e-17 |
| H1_O2.csv | amplitude | 3.90265014595e-23 | 7.42827959689e-18 |
| H1_O3a.csv | amplitude | 2.59378996027e-23 | 8.79328973501e-18 |
| H1_O3b.csv | amplitude | 2.36363994894e-23 | 5.79898980256e-18 |
| L1_O1.csv | amplitude | 4.09531995572e-23 | 1.4298899502e-16 |
| L1_O2.csv | amplitude | 3.14222988518e-23 | 2.37894993501e-17 |
| L1_O3a.csv | amplitude | 2.42046002222e-23 | 4.73245993027e-18 |
| L1_O3b.csv | amplitude | 1.59848005407e-23 | 6.25423981962e-17 |
| H1_O1.csv | snr | 7.50006008148 | 81178.7265625 |
| H1_O2.csv | snr | 7.50013017654 | 105073.03125 |
| H1_O3a.csv | snr | 7.50001001358 | 14597.0146484 |
| H1_O3b.csv | snr | 7.50016021729 | 21292.2539062 |
| L1_O1.csv | snr | 7.50001001358 | 13013.9384766 |
| L1_O2.csv | snr | 7.50002002716 | 156629.9375 |
| L1_O3a.csv | snr | 7.5 | 18300.9355469 |
| L1_O3b.csv | snr | 7.50001001358 | 22107.9804688 |
| H1_O1.csv | q_value | 5.65685 | 45.2548 |
| H1_O2.csv | q_value | 5.65685 | 45.2548 |
| H1_O3a.csv | q_value | 5.65685424949 | 45.2548339959 |
| H1_O3b.csv | q_value | 5.65685424949 | 45.2548339959 |
| L1_O1.csv | q_value | 5.65685 | 1e+20 |
| L1_O2.csv | q_value | 5.65685 | 1e+20 |
| L1_O3a.csv | q_value | 5.65685424949 | 45.2548339959 |
| L1_O3b.csv | q_value | 5.65685424949 | 45.2548339959 |

## Limitations and downstream use

Labels are model predictions; rare classes, imperfect labels and class imbalance need explicit treatment. Different detectors/runs have different noise populations and observing conditions, so an unrestricted random split can conceal domain shifts. D-02 should decide confidence filtering, deduplication and class selection; later split/feature tasks should prevent event and preprocessing leakage. The release description gives SNR > 7.5 and peak frequency 10–2048 Hz as selection criteria, but this profiler reports actual CSV values without enforcing these cuts. No waveform or auxiliary channel data is downloaded by this task.

## Licence and citation

The dataset's Zenodo API metadata declares `cc-by-4.0`: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Provide attribution and identify any later modifications. This dataset licence is separate from the project's software licence.

Dataset: Glanzer, J., et al. (2021). *Gravity Spy Machine Learning Classifications of LIGO Glitches from Observing Runs O1, O2, O3a, and O3b* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.5649212

Paper: Glanzer, J., et al. (2023). *Data quality up to the third observing run of Advanced LIGO: Gravity Spy glitch classifications*. Classical and Quantum Gravity, 40, 065004. https://doi.org/10.1088/1361-6382/acb633

Sources: [release and column definitions](https://zenodo.org/records/5649212), [release API (licence, sizes and MD5)](https://zenodo.org/api/records/5649212), [Omicron documentation](https://virgo.docs.ligo.org/virgoapp/Omicron/).

Generated by `data/profile.py` from checksum-verified originals; exact statistics and observed column order are in `docs/data_summary.json`.
