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
| `avg_Mulliken EN` | Average Mulliken electronegativity of constituent elements in the host composition |
| `avg_First ionization energy (kJ/mol)` | Average first ionization energy of constituent elements in the host composition |
| `1/r²` | Inverse square of the average Cr³⁺–ligand bond length |
| `avg_Metallic valence` | Average metallic valence of constituent elements in the host composition |
| `avg_Martynov-Batsanov EN` | Average Martynov–Batsanov electronegativity of constituent elements |
| `beta` | Lattice angle β of the host crystal structure (°); equals 90° for cubic, tetragonal, orthorhombic, and hexagonal systems; deviates for monoclinic structures |
| `SGR No.` | Space group number of the host crystal structure (categorical — integer values carry no ordinal meaning) |
| `avg_Number of outer shell electrons` | Average number of outer shell electrons of constituent elements |
| `X` | Cr³⁺ dopant concentration (molar fraction) |
| `max_metal_ligand_bond_length` | Maximum metal–ligand bond length in the coordination polyhedron (Å) |
| `std_Mendeleev number` | Standard deviation of Mendeleev numbers of constituent elements |
| `volume_per_atom` | Unit cell volume per atom of the host crystal structure (Å³) |
| `max_First ionization energy (kJ/mol)` | Maximum first ionization energy of constituent elements in the host composition |
| `volume_per_fu` | Unit cell volume per formula unit of the host crystal structure (Å³) |
| `polyhedron volume` | Volume of the CrO₆ octahedral coordination polyhedron (Å³) |

> **Note on feature types:** The majority of descriptors are continuous numerical features. `SGR No.` is categorical — space group 62 (Pnma) has no mathematical relationship to space group 225 (Fm-3m). `beta` is numerical but quasi-categorical in practice, as it equals exactly 90.0° for all higher-symmetry crystal systems. CatBoost and the GP model handle both types without requiring one-hot encoding.

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
| Tier 1 — Strong | Predicted Dq/B in target range [2.2–2.8], σ < 0.2 |
| Tier 2 — Promising | Predicted Dq/B in target range [2.2–2.8], σ < 0.4 |
| Tier 3 — Uncertain | Predicted Dq/B in target range [2.2–2.8], σ ≥ 0.4 |
| Tier 3 — Edge | Predicted Dq/B within 0.3 of target boundary |
| Tier 4 — Out of range | Outside target range |

---

## Installation

```
pip install -r requirements.txt
```

## Usage

**Streamlit GUI (recommended):**

```
streamlit run gp_streamlit_gui.py
```

**Command-line script:**

```
python gp_dqb_prediction_scaled.py
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

The model was trained on **243 experimentally characterized Cr³⁺-doped phosphor host lattices** spanning oxide and fluoride structure types — including garnets, perovskites, spinels, elpasolites, corundum-type, rutile-type, and related families — with Dq/B values ranging from 1.17 to 3.43 (mean: 2.25).

The dataset was assembled through an exhaustive survey of the peer-reviewed literature. Coverage is limited not by the scope of the search, but by the availability of experimentally characterized compounds with sufficiently documented spectroscopic data for rigorous Dq/B determination.

A companion datasheet documenting the dataset composition, collection process, preprocessing, and recommended use is available in the repository as `Datasheet_Cr3_DqB_Dataset.docx`.

---

## Related Repository

This system is the successor to the CatBoost and neural network models developed in:
[KirkaSSS/phD-AI](https://github.com/KirkaSSS/phD-AI) — CatBoost (white-box) and MLP neural network (black-box) models with uncertainty estimation, trained on an earlier dataset of 174 compounds.

---

## Authors

**Snežana Đurković**
ORCID: [0009-0007-6638-0682](https://orcid.org/0009-0007-6638-0682)
Institute of Nuclear Sciences "Vinča", University of Belgrade
OMAS Group — Optical Materials and Spectroscopy
Belgrade, 2026

*Developed under the supervision of:*
Prof. Dr. Miroslav Dramićanin
Dr. Zoran Ristić
Institute of Nuclear Sciences "Vinča", University of Belgrade

---

## Citation

If you use this code or dataset in your research, please cite:

```
Đurković S., Dramićanin M.D. Gaussian Process Expert System for Cr³⁺ Phosphor Discovery.
OMAS Group, Institute of Nuclear Sciences Vinča, University of Belgrade, 2026.
https://github.com/KirkaSSS/Cr3-GP-Expert-System
```

---

## License

MIT License — free to use, modify, and distribute with attribution.
