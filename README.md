# Uni-PVT-pro: Physics-Informed--Mixture-of-Experts for Universal Fluid Thermodynamic Property Prediction

**Uni-PVT-pro** is an advanced deep learning framework designed to predict thermodynamic properties (such as Compressibility Factor , Fugacity Coefficient , Enthalpy , and Entropy ) from PVT data.

It combines **Mixture-of-Experts (MoE)** architecture to handle complex phase behaviors (Gas, Liquid, Transition, Critical, etc.) with **Physics-Informed Neural Networks (PINNs)** to ensure the predictions obey fundamental thermodynamic laws.



## 🌟 Key Features & Advantages

1. **Physics-Informed (PINN) Loss**:
* Unlike standard black-box models, this model enforces thermodynamic consistency using soft constraints derived from Equations of State (EOS).
* It minimizes residuals for derivatives like $(\frac{\partial ln \phi}{\partial p})_T$, $(\frac{\partial H}{\partial p})_T$, and $(\frac{\partial S}{\partial p})_T$.


2. **Mixture-of-Experts (MoE) Architecture**:
* **Specialized Experts**: Uses distinct residual networks (Experts) for different phase regions (Gas, Liquid, Critical, Extra).
* **Gating Network**: A learnable gate dynamically routes inputs to the correct expert(s) based on state variables (p, T).
* **Hard & Soft Routing**: Supports hard routing during pre-training (using known labels) and soft (probabilistic) routing during fine-tuning.


3. **Cascade Prediction Mode**:
* Supports a physical dependency chain: .
* The model can predict the Z-factor first and inject it as a feature to predict derived properties ($\phi$, $H$, $S%), mimicking physical calculation pathways.


4. **Three-Stage Training Strategy**:
* **Stage 1**: Pre-train experts individually using region labels.
* **Stage 2**: Train the Gating Network while freezing experts.
* **Stage 3**: Joint fine-tuning with PINN constraints enabled.


5. **Comprehensive Analysis Tools**:
* **SHAP Analysis**: Explainability for feature importance.
* **PT Visualization**: High-quality 3D and 2D contour plots of property distributions across the Pressure-Temperature landscape.
* **Optuna**: Built-in hyperparameter optimization.



---

## 📂 Project Structure

```text
Uni-PVT-pro/
├── config/                 # Configuration files
│   ├── current.yaml        # Active configuration
│   └── config_total.yaml   # Template with all options
├── data/                   # Input datasets (CSV format)
├── models/                 # Neural Network Definitions
│   ├── experts.py          # Residual Blocks & Activation Logic
│   ├── gate.py             # Gating Network
│   └── fusion_model.py     # Main MoE Model (Standard & Cascade)
├── utils/                  # Utility Functions
│   ├── dataset.py          # Data loading, splitting, scaling
│   ├── physics_loss.py     # Loss calculation (Data + Regularization)
│   ├── thermo_pinn.py      # Thermodynamic derivative calculations
│   ├── trainer.py          # 3-Stage Training Loop
│   ├── logger.py           # Logging utility
│   └── visualize.py        # Basic plotting
├── train.py                # Main training script
├── predict.py              # Inference & Evaluation script
├── run_all.py              # Orchestrator (Train -> Predict -> Viz)
├── shap_analysis.py        # Feature attribution analysis
├── pt_viz.py               # Advanced PT-diagram visualization
├── optimize.py             # Hyperparameter tuning (Optuna)
└── requirements.txt        # Python dependencies

```

---

## 🚀 Installation

1. **Prerequisites**: Python 3.10+ and CUDA (optional, for GPU acceleration).
2. **Clone the repository**:
```bash
git clone https://github.com/your-username/Uni-PVT-pro.git #SSH is recommended, http is also OK.
cd Uni-PVT-pro

```


3. **Install dependencies**:
```bash
pip install -r requirements.txt

```



---

## 📖 Manual / Usage

### 1. Quick Start (Run All)

The easiest way, also the most recommended way to run the full pipeline (Train  Predict  SHAP  Visualization) is using the orchestrator script:

```bash
python run_all.py --config config/current.yaml

```

### 2. Training

To train the model manually using the 3-stage strategy:

```bash
python train.py --config config/current.yaml

```

* **Outputs**: Saved in `results/{timestamp}/`, including `best_model.pt`, logs, and scalars.

### 3. Prediction & Evaluation

To generate predictions on the test set and calculate metrics (MAE, MSE, ):

```bash
python predict.py --config results/{timestamp}/config_used.yaml

```

* **Outputs**: `test_predictions.csv`, scatter plots, and `metrics_summary.yaml`.

### 4. Visualization (PT Diagrams)

To generate 3D and 2D contour maps of the predictions relative to Pressure and Temperature:

```bash
python pt_viz.py --csv results/{timestamp}/eval/test_predictions.csv --outdir results/{timestamp}/pt_viz

```

### 5. Hyperparameter Optimization

To search for the best hyperparameters using Optuna:

```bash
python optimize.py --config config/current.yaml --n-trials 50

```

---

## ⚙️ Configuration Guide

The behavior of the model is controlled entirely by the YAML config file. Here are key sections in `config_total.yaml`:

### Data Selection

```yaml
feature_cols: # Data inputs, you can choose feature you want to input
  -T_r (-)
  -p_r (-)  
targets: # Outputs to predict, you can also choose properties you want to predict
  -Z (-)
  -H (J/mol)       
expert_col: # Column containing Phase ID (1-4), should be included
  -no                      

```

### PINN Settings

```yaml
pinn:
  enabled: true               # Turn on the mode
  input_mode: "reduced"       # Use reduced T_r, p_r for calculation
  lambda_H: 0.05              # Weight for Enthalpy constraint
  lambda_S: 0.05              # Weight for Entropy constraint
  p_to_Pa: 1.0e6              # Unit conversion factor to Pascals

```

### Training Strategy

```yaml
training:
  pretrain_epochs: 30         # Stage 1: Expert pre-training
  gate_epochs: 50             # Stage 2: Gate training
  finetune:
    enabled: true
    epochs: 80                # Stage 3: Joint training

```

### Loss Function

```yaml
loss:
  supervised: "mse"           # or "huber"
  region_weights:             # Weight specific phase regions higher
    enabled: true
    weights: {1: 1.0, 3: 2.0} # e.g., Focus more on Critical region (3)

```

---

## 🧠 Methodology Details

### The Physics Loss

The total loss function is defined as:


1. Weighted MSE/Huber loss against ground truth labels.
2. Soft constraints derived from thermodynamics. For example, Enthalpy residual:


3. Regularization terms, including:
* **Non-negative penalty**: Penalizes unphysical negative outputs.
* **Entropy penalty**: Encourages sharp or smooth gating decisions.
* **Extreme region penalty**: Extra weight for difficult regions (e.g., ).



### Cascade Model

When configured, the model uses a two-step approach:

1. **Step 1**: `x -> [Z-Head] -> Z_pred`
2. **Step 2**: `[x, Z_pred] -> [Props-Head] -> H, S, phi`
This ensures that property predictions are consistent with the predicted state of the fluid ().
