# VISREAS

Code for the paper [VISREAS: Complex Visual Reasoning with Unanswerable Questions](https://arxiv.org/abs/2403.10534) by Syeda Nahida Akter, Sangwu Lee, Yingshan Chang, Yonatan Bisk, Eric Nyberg. We release minimal code to train and evaluate LLaVA, BLIP2, InstructBLIP, and various VLM baselines on the VISREAS dataset.

**Keep posted for more updates!**

## Model

Logic2Vision is a [LLaVA-1.5-13B](https://huggingface.co/llava-hf/llava-1.5-13b-hf) model finetuned on [VisReas dataset](https://arxiv.org/abs/2403.10534) for complex visual reasoning tasks.
The model has been finetuned using LoRA to generate python pseudocode outputs to solve a complex visual reasoning tasks. We released the model on Huggingface models [here](https://huggingface.co/RE-N-Y/logic2vision).

## TODOS

- [ ] Release VISREAS dataset on Huggingface Datasets.
- [x] Release Logic2Vision on Huggingface Models.
- [ ] Make code compatible with HF datasets library.

## Citation

```
@misc{akter2024visreas,
    title={VISREAS: Complex Visual Reasoning with Unanswerable Questions},
    author={Syeda Nahida Akter and Sangwu Lee and Yingshan Chang and Yonatan Bisk and Eric Nyberg},
    year={2024},
    eprint={2403.10534},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
