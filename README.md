## Tiny LLM Trainer

The experiment implements a tiny language model trainer using [PyTorch](https://pytorch.org/). I designed it to train on Wikipedia data and generate text based on the learned patterns.

### Features

- PyTorch-based implementation
- Transformer architecture
- Configurable model size and training parameters
- Text generation with temperature and top-k sampling

### Requirements

- Python 3.7+
- PyTorch
- NumPy

### Project Structure

```sh
.
├── data
├── models
├── tiny_llm_trainer.py
└── wikipedia_data.py
```

### Files

- `wikipedia_data.py`: Script for downloading and preprocessing [Wikipedia](https://www.wikipedia.org) data.
- `data/`: Directory where preprocessed training data from Wikipedia is saved.
- `tiny_llm_trainer.py`: The main script for training the model.
- `models/`: Directory where trained models are saved.

### Usage

1. Python Package Installer:

   ```sh
   pip3 install uv
   ```

2. Prerequisites:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   python3 -m pip install --upgrade pip
   deactivate # deactivate virtual environment
   ```

3. Prepare Data:

   ```sh
   python3 wikipedia_data.py
   ```

4. Train LLM:

   ```sh
   python3 tiny_llm_trainer.py
   ```

### License

This project is licensed under the [Apache License 2.0](./LICENSE).

### Citation

```tex
@misc{tlt2024,
  author       = {Oketunji, A.F.},
  title        = {Tiny LLM Trainer},
  year         = 2024,
  version      = {0.0.2},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.12579634},
  url          = {https://doi.org/10.5281/zenodo.12579634}
}
```

### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.