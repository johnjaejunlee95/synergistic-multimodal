# Can One Modality Model Synergize Training of Other Modality Models?

This is the official PyTorch implementation of our paper "Can One Modality Model Synergize Training of Other Modality Models?"

> **Can One Modality Model Synergize Training of Other Modality Models?**  
> Jae-Jun Lee, Sung Whan Yoon  
> **Accepted by: ICLR 2025**
>
> [arXiv: Upload Soon] [[ICLR 2025](https://openreview.net/pdf?id=5BXWhVbHAK)]

## Installation

Install the required dependencies by running:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm transformers huggingface-hub
```

Install any additional dependencies as needed for your specific setup.

## Experiments

For detailed instructions on running experiments, please refer to the README files in the following directories:
- `vision-language-models/`
- `multimodal/`

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{lee2025can,
  title={Can One Modality Model Synergize Training of Other Modality Models?},
  author={Jae-Jun Lee and Sung Whan Yoon},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=5BXWhVbHAK}
}
```