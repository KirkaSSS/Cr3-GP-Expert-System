# Cr³⁺ Phosphor Discovery — Gaussian Process Expert System

A machine learning expert system for computational screening and discovery of Cr³⁺-doped phosphor materials, developed at the Institute of Nuclear Sciences "Vinča", Belgrade.

The system uses **Gaussian Process Regression (GPR)** to predict the crystal field parameter **Dq/B** for candidate phosphor host lattices, enabling targeted selection of materials with emission in the desired near-infrared spectral range.

---

## Background

The Dq/B ratio (crystal field strength to Racah parameter) is the key descriptor governing the emission wavelength of Cr³⁺-doped phosphors. Materials with Dq/B in the range 2.2–2.8 emit in the physiological transparency window (650–900 nm), making them candidates for bioimaging, night-vision, and plant-growth lighting applications.

Experimental determination of Dq/B requires synthesis and spectroscopic characterization of each candidate — a slow and resource-intensive process. This expert system predicts Dq/B from structural and chemical descriptors alone, allowing large candidate spaces to be screened computationally before any synthesis is attempted.

---

## Features

- Gaussian Process Regression with **Matérn 5/2 kernel**
- **Automatic Relevance Determination (ARD)** — one length-scale per feature, automatically ranking descriptor importance
- **Uncertainty quantification** — posterior standard deviation σ and 95% credible intervals for every prediction
- **RobustScaler** preprocessing — handles extreme feature distributions (e.g. 1/r², SGR number)
- **Tier classification** of candidates based on predicted Dq/B and confidence
- **Streamlit web GUI** — interactive interface with Plotly visualizations and Excel/CSV export
- **5-fold cross-validation** with RMSE, R², and uncertainty calibration metrics

---

## Input Descriptors

The model uses 15 structural and chemical descriptors per host lattice composition:

| Descriptor | Description |
|---|---|
| `avg_Mulliken EN` | Average Mulliken electronegativity of constituent atoms |
| `avg_First ionization energy (kJ/mol)` | Average first ionization energy |
| `1/r²` | Inverse squared ionic radius descriptor |
| `avg_Metallic valence` | Average metallic valence |
| `avg_Martynov-Batsanov EN` | Average Martynov-Batsanov electronegativity |
| `beta` | Nephelauxetic ratio β |
| `SGR No.` | Structure-type classification number |
| `avg_Number of outer shell electrons` | Average number of outer shell electrons |
| `X` | Cr³⁺ dopant concentration |
| `max_metal_ligand_bond_length` | Maximum metal-ligand bond length (Å) |
| `std_Mendeleev number` | Standard deviation of Mendeleev numbers |
| `volume_per_atom` | Unit cell volume per atom (Å³) |
| `max_First ionization energy (kJ/mol)` | Maximum first ionization energy among constituents |
| `volume_per_fu` | Volume per formula unit (Å³) |
| `polyhedron volume` | Volume of the CrO₆ coordination polyhedron (Å³) |

---

## File Format

**Training file** (`.xlsx`):
```
Formula | Dq/B | avg_Mulliken EN | avg_First ionization energy (kJ/mol) | ... | polyhedron volume
```

**Prediction file** (`.xlsx`):
```
Formula | avg_Mulliken EN | avg_First ionization energy (kJ/mol) | ... | polyhedron volume
```

---

## Tier Classification

| Tier | Condition |
|---|---|
| Tier 1 — Strong | Predicted Dq/B in target range, σ < 0.2 |
| Tier 2 — Promising | Predicted Dq/B in target range, σ < 0.4 |
| Tier 3 — Uncertain | Predicted Dq/B in target range, σ ≥ 0.4 |
| Tier 3 — Edge | Predicted Dq/B within 0.3 of target boundary |
| Tier 4 — Out of range | Outside target range |

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Streamlit GUI (recommended):**
```bash
streamlit run gp_streamlit_gui.py
```

**Command-line script:**
```bash
python gp_dqb_prediction.py
```
Edit the `TRAIN_PATH` and `PREDICT_PATH` variables at the top of the script to point to your files.

---

## Requirements

```
streamlit
scikit-learn
pandas
numpy
plotly
openpyxl
```

---

## Training Data

The model was trained on 192 experimentally characterized Cr³⁺-doped phosphor host lattices spanning oxide, fluoride, phosphate, germanate, silicate, and garnet structure types, with Dq/B values ranging from 1.17 to 3.43.

---

## Authors

**Snežana Đurković**  
Institute of Nuclear Sciences "Vinča", University of Belgrade  
OMAS Group — Optical Materials and Spectroscopy  
Belgrade, 2026

*Developed under the supervision of:*  
Prof. Dr. Miroslav Dramićanin  
Dr. Zoran Ristić  
Institute of Nuclear Sciences "Vinča", University of Belgrade

---

## License

MIT License — free to use, modify, and distribute with attribution.
