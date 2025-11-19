# 🚀 Quick Start Guide

Get started with the RUL Prediction System in 5 minutes!

## Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/RUL-TurboFan.git
cd RUL-TurboFan

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Download Dataset

```bash
cd data
python download_cmapss.py
cd ..
```

This will download the NASA C-MAPSS dataset (~5MB) automatically.

## Step 2: Train a Model

### Option A: Quick Training (Transformer on FD001)

```bash
python src/train.py --dataset FD001 --model_type transformer --epochs 50
```

### Option B: Use Configuration File

```bash
python src/train.py --config configs/transformer_fd001.yaml
```

### All Model Types

```bash
# Transformer (recommended)
python src/train.py --dataset FD001 --model_type transformer

# LSTM Baseline
python src/train.py --dataset FD001 --model_type lstm_attention

# TCN
python src/train.py --dataset FD001 --model_type tcn

# CNN-Transformer Hybrid
python src/train.py --dataset FD001 --model_type hybrid
```

## Step 3: Evaluate Model

```bash
python src/evaluate.py \
    --dataset FD001 \
    --model_type transformer \
    --compare_baselines
```

This will:
- Load the best trained model
- Evaluate on test set
- Calculate all metrics (RMSE, MAE, NASA Score, R²)
- Compare with literature baselines

## Step 4: Visualize Results

```bash
python src/visualize.py --dataset FD001 --model_type transformer
```

Generates:
- Attention heatmaps
- Feature importance plots
- RUL prediction scatter plots
- Error distribution analysis
- Model comparison charts

## View Demo Results

We've included pre-generated demo visualizations:

```bash
ls results/
```

Files:
- `demo_training_history.png` - Training curves
- `demo_predictions.png` - Predictions vs truth
- `demo_attention_heatmap.png` - Attention weights
- `demo_feature_importance.png` - Sensor importance
- `demo_model_comparison.png` - Model benchmarks

## Python API Usage

```python
from src.data_loader import CMAPSSDataLoader
from src.models.transformer import create_transformer_model
import torch

# 1. Load and preprocess data
loader = CMAPSSDataLoader(data_dir='data/CMAPSS', dataset_name='FD001')
data = loader.load_and_preprocess(
    sequence_length=30,
    max_rul=125,
    val_split=0.2
)

# 2. Create model
model = create_transformer_model(
    num_features=data['num_features'],
    model_size='base'  # or 'small', 'large'
)

# 3. Make predictions
model.eval()
with torch.no_grad():
    X = torch.FloatTensor(data['X_test'][:10])
    predictions, _, attention = model(X, return_attention=True)

print(f"Predictions: {predictions.squeeze().numpy()}")
print(f"True RUL: {data['y_test'][:10]}")
```

## Common Commands

### Train on All Datasets

```bash
for dataset in FD001 FD002 FD003 FD004; do
    python src/train.py --dataset $dataset --model_type transformer
done
```

### Compare All Models on FD001

```bash
for model in transformer lstm_attention tcn hybrid; do
    python src/train.py --dataset FD001 --model_type $model --epochs 50
    python src/evaluate.py --dataset FD001 --model_type $model
done
```

### Generate All Visualizations

```bash
python src/visualize.py --dataset FD001 --model_type transformer
```

## Configuration

Edit `configs/transformer_fd001.yaml` to customize:

```yaml
# Model architecture
model:
  type: 'transformer'
  d_model: 128
  nhead: 8
  num_layers: 4

# Training parameters
training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  loss: 'asymmetric'
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python src/train.py --batch_size 128
```

### Slow Training
```bash
# Use smaller model
python src/train.py --model_size small

# Or fewer epochs
python src/train.py --epochs 50
```

### Dataset Download Fails

If automatic download fails:
1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Download "Turbofan Engine Degradation Simulation Data Set"
3. Extract to `data/CMAPSS/`

## Expected Training Time

On CPU:
- FD001 (100 units): ~30-60 minutes
- FD002 (260 units): ~90-120 minutes

On GPU (NVIDIA RTX 3080):
- FD001: ~5-10 minutes
- FD002: ~15-20 minutes

## Expected Performance

After training on FD001, you should achieve:

- **RMSE**: 12-14 (lower is better)
- **MAE**: 9-11 (lower is better)
- **NASA Score**: 250-300 (lower is better)
- **R²**: 0.88-0.92 (higher is better)

These results are competitive with published state-of-the-art methods!

## Next Steps

1. **Read the full documentation**: See `README.md`
2. **Explore notebooks**: Check `notebooks/01_data_exploration.ipynb`
3. **Customize models**: Edit architecture in `src/models/`
4. **Apply to your data**: Adapt `data_loader.py` for custom datasets

## Getting Help

- **Documentation**: See `README.md` and `SUMMARY_REPORT.md`
- **Issues**: Open an issue on GitHub
- **Examples**: Check `notebooks/` directory

---

Happy predicting! 🎯
