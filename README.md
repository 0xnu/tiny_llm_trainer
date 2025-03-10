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
- Pillow

### Project Structure

```sh
.
├── data
├── models
├── wikipedia_data.py
├── tiny_llm_trainer.py
├── flickr_data.py
├── tiny_llm_trainer_vqa.py
├── cvc_data.py
└── tiny_llm_trainer_cvc.py
```

### Files

- `data/`: Directory where preprocessed training data from Wikipedia is saved.
- `models/`: Directory where trained models are saved.
- `wikipedia_data.py`: Script for downloading and preprocessing [Wikipedia](https://www.wikipedia.org) data.
- `tiny_llm_trainer.py`: The main script for training the model.
- `flickr_data.py`: Script for downloading and preprocessing [Flickr](https://flickr.com) image data.
- `tiny_llm_trainer_vqa.py`: Script for training the model on Visual Question Answering (VQA) tasks using Flickr data.
- `cvc_data.py`: Script for downloading and preprocessing [Common Voice Corpus 1](https://commonvoice.mozilla.org/en/datasets) data.
- `tiny_llm_trainer_cvc.py`: Script for training a TTS model using Common Voice Corpus 1 data.

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

### Text Generation

1. Prepare Data:

   ```sh
   python3 wikipedia_data.py
   ```

2. Train LLM:

   ```sh
   python3 tiny_llm_trainer.py
   ```

### Visual Question Answering (VQA)

1. Prepare Data:

   ```sh
   python3 flickr_data.py
   ```

2. Train VQA — Multimodal:

   ```sh
   python3 tiny_llm_trainer_vqa.py
   ```

### Text-to-Speech (TTS)

1. Prepare Data:

   ```sh
   python3 cvc_data.py
   ```

2. Train TTS:

   ```sh
   tiny_llm_trainer_cvc.py
   ```

### References

+ [Large Language Model (LLM) AI text generation detection based on transformer deep learning algorithm](https://arxiv.org/abs/2405.06652)
+ [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
+ [From Images to Textual Prompts: Zero-shot VQA with Frozen Large Language Models](https://arxiv.org/abs/2212.10846)
+ [Enhancing Image Caption Generation Using Reinforcement Learning with Human Feedback](https://arxiv.org/abs/2403.06735)
+ [VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468)
+ [Meta Learning Text-to-Speech Synthesis in over 7000 Languages](https://arxiv.org/abs/2406.06403)
+ [Text to Speech Synthesis](https://arxiv.org/abs/2401.13891)

### License

This project is licensed under the [Apache License 2.0](./LICENSE).

### Citation

```tex
@misc{tlt2024,
  author       = {Oketunji, A.F.},
  title        = {Tiny LLM Trainer},
  year         = 2024,
  version      = {0.0.6},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.12593929},
  url          = {https://doi.org/10.5281/zenodo.12593929}
}
```

### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.