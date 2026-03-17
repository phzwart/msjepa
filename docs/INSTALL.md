# Installing MSJEPA

## Get the code on GitHub

From your local `msjepa` project directory:

1. **Initialize git and make the first commit** (if not already done):

   ```bash
   cd /path/to/msjepa
   git init
   git add .
   git status   # review what will be committed
   git commit -m "Initial commit: MSJEPA dense representation learning"
   ```

2. **Create a new repository on GitHub**

   - Go to [github.com/new](https://github.com/new).
   - Repository name: `msjepa` (or any name you prefer).
   - Choose Public (or Private).
   - Do **not** initialize with a README, .gitignore, or license (you already have them).
   - Click **Create repository**.

3. **Add the remote and push**

   Replace `YOUR_USERNAME` with your GitHub username:

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/msjepa.git
   git branch -M main
   git push -u origin main
   ```

   If you use SSH:

   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/msjepa.git
   git branch -M main
   git push -u origin main
   ```

---

## Installing at NERSC

On NERSC (e.g. Perlmutter), use a Python environment (conda or venv) and install from GitHub.

### Option A: Install from GitHub (recommended)

```bash
# Load modules and create env as needed, e.g.:
module load python
python -m venv myenv
source myenv/bin/activate

# Install from GitHub (replace YOUR_USERNAME with your GitHub username)
pip install git+https://github.com/YOUR_USERNAME/msjepa.git
```

For a specific branch or tag:

```bash
pip install git+https://github.com/YOUR_USERNAME/msjepa.git@main
# or
pip install git+https://github.com/YOUR_USERNAME/msjepa.git@v0.1.0
```

### Option B: Clone and install in editable mode

Useful if you want to change code on the system:

```bash
git clone https://github.com/YOUR_USERNAME/msjepa.git
cd msjepa
pip install -e .
```

### Requirements at NERSC

- Python ≥ 3.10
- PyTorch: load the NERSC PyTorch module or install in your env, e.g.:

  ```bash
  module load pytorch
  # or install in your venv/conda env
  ```

Then install msjepa as above; `pip` will install `numpy`, `Pillow`, and `PyYAML` from `pyproject.toml`.

### Run training

```bash
msjepa-train --config configs/default.yaml --train-root /path/to/train --val-root /path/to/val
# or
python -m msjepa.train --config configs/default.yaml --train-root /path/to/train --val-root /path/to/val
```
