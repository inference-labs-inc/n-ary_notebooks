# n-ary notebooks

**n-ary notebooks** is a collection of mathematical notebooks exploring ideas that arise naturally in the study of zero-knowledge proofs and related areas of cryptography. The emphasis is on *understanding*, *derivation*, and *mechanism* rather than on production-ready implementations.

At present, the notebooks focus primarily on the **sum-check protocol**, **multilinear extensions**, **arithmetic circuis**, and the **GKR protocol**, with an emphasis on why these constructions work, what guarantees they provide, and how their complexity behaves. Over time, the scope may broaden to include other mathematical topics that are relevant to cryptography, verification, or interactive proof systems.

This is best thought of as a set of **working notes with executable examples**, not as a comprehensive textbook or a complete cryptographic framework.

---

## What these notebooks are (and are not)

**They are:**
- Mathematically precise explanations of core ideas behind sum-check and GKR  
- Interactive demonstrations designed to make abstract arguments concrete  
- A place to experiment with small-scale implementations and counterexamples  
- Written with clarity and pedagogy in mind 

**They are not:**
- A full zk-SNARK system  
- A performance-optimized prover or verifier  
- A general-purpose cryptographic library  
- A substitute for peer-reviewed protocol specifications  

---

## Current notebook topics

The current notebooks include (but are not limited to):

- **Sum-check protocol**
  - Correctness, soundness, and error bounds
  - Degree tracking and verifier efficiency
  - Concrete examples over finite fields

- **Multilinear extensions**
  - Boolean hypercubes and polynomial interpolation
  - Why multilinearity matters in interactive proofs

- **Arithmetic circuits**
  - Circuits as structured polynomials
  - How circuit structure feeds into sum-check and GKR

- **GKR protocol**
  - High-level structure and intuition
  - Relationship to sum-check and circuit depth

Future notebooks may cover additional mathematical tools as they become relevant.

---

## Installation

The project is packaged as a standard Python package so that the notebooks remain focused on mathematics rather than setup code.

The only **system-level dependency** is **Graphviz**, which is required for circuit and protocol visualizations.

---

### 1. System prerequisite: Graphviz

Graphviz must be installed **at the OS level** (the Python package alone is not sufficient).

#### macOS

```bash
brew install graphviz
```

#### Linux (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install graphviz
```

#### Windows

1. Download Graphviz from: [https://graphviz.org/download/](https://graphviz.org/download/)
2. Install it and ensure the `dot` executable is added to your `PATH`
3. Verify installation:

```powershell
dot -V
```

---

### 2. Create a Python environment (recommended)

You can use either **conda** or **pip + venv**. Conda is recommended if you want a fully reproducible environment.

#### Option A: Conda (recommended)

```bash
conda env create -f environment.yaml
conda activate gkr
```

#### Option B: pip + venv

```bash
# Create a virtual environment
python -m venv .venv
```

Activate it:

* **Windows (PowerShell)**

  ```powershell
  .venv\Scripts\Activate.ps1
  ```

* **macOS / Linux**

  ```bash
  source .venv/bin/activate
  ```

---

### 3. Install the package (editable mode)

From the project root:

```bash
pip install -e .
```

Editable mode (`-e`) ensures that changes to the source code are immediately reflected in notebooks and scripts without reinstallation.

#### Optional extras

* Notebook support (recommended):

  ```bash
  pip install -e ".[notebook]"
  ```

* Colored terminal output (nice to have):

  ```bash
  pip install -e ".[colors]"
  ```

* Everything:

  ```bash
  pip install -e ".[notebook,colors]"
  ```

---

### 4. Launch Jupyter

```bash
jupyter notebook
```

or, if you prefer:

```bash
jupyter lab
```

The notebooks can import the package directly; no additional initialization is required.

---

### Notes

* The package also installs a small CLI entry point:

  ```bash
  gkr
  ```

  (This will expand over time.)

* Visualization features depend on Graphviz being available on your.venv\Scripts\activate
 system. If you encounter rendering errors, verify that `dot` is accessible from the command line.

---
## License

This project is released under the *MIT License*.
See the `LICENSE` file at the repository root for full terms.
