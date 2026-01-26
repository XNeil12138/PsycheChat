# PsychēChat

## Introduction

Welcome to **PsychēChat** 🎉
We propose a framework **PsychēChat**: An Empathic Framework Focused on *Emotion Shift Tracking* and *Safety Risk Analysis* in Psychological Counseling.

<div align="center">
  <img src="figure/framework.svg">
</div>

---

## 📦 Installation

Clone repository:

```bash
git clone https://github.com/JOHNNY-fans/PsycheChat.git
cd PsycheChat
pip install -r requirements.txt
```

---

## Resources

### 📦 Models
| Model | Description |
|:---------|:------------|
| [PsycheChat-Counselor-Agent-Mode-Qwen3-8B](https://huggingface.co/Johnnyfans/PsycheChat-Counselor-Agent-Mode-Qwen3-8B) | Built on Qwen3-8B with full SFT data in Agent Mode. |
| [PsycheChat-Counselor-LLM-Mode-Qwen3-8B](https://huggingface.co/Johnnyfans/PsycheChat-Counselor-LLM-Mode-Qwen3-8B) | Built on Qwen3-8B with full SFT data in LLM Mode.|
| [PsycheChat-Counselor-Agent-Mode-Qwen2.5-7B-Instruct](https://huggingface.co/Johnnyfans/PsycheChat-Counselor-Agent-Mode-Qwen2.5-7B-Instruct) | Built on Qwen2.5-7B-Instruct with full SFT data in Agent Mode. |
| [PsycheChat-Counselor-LLM-Mode-Qwen2.5-7B-Instruct](https://huggingface.co/Johnnyfans/PsycheChat-Counselor-LLM-Mode-Qwen2.5-7B-Instruct) | Built on Qwen2.5-7B-Instruct with full SFT data in LLM Mode. |
| [PsycheChat-Seeker-Qwen3-8B](https://huggingface.co/Johnnyfans/PsycheChat-Seeker-Qwen3-8B) | Built on Qwen3-8B with Dialogue-Guided Seeker data.|

Additional, stronger models will be released progressively. *To be released soon...*

### 📂 Datasets

We provide the counseling dialogue dataset PsychēDialog constructed in PsychēChat, along with training data for the two corresponding modes.

| Dataset | Description |
|:---------|:------------|
| [PsycheDialog](https://huggingface.co/datasets/Johnnyfans/PsycheDialog)   | The counseling dialogue data constructed through role-playing. |
| [PsycheDialog-Counselor-Agent-Mode](https://huggingface.co/datasets/Johnnyfans/PsycheDialog-Counselor-Agent-Mode) | Training data dataset in the Agent Mode. |
| [PsycheDialog-Counselor-LLM-Mode](https://huggingface.co/datasets/Johnnyfans/PsycheDialog-Counselor-LLM-Mode) | Training data in the LLM Mode. |
| [PsycheDialog-Seeker](https://huggingface.co/datasets/Johnnyfans/PsycheDialog-Seeker) | Training data for Dialogue-Guided Seeker in the Agent Mode. |

---

# 📚 Citation

If you use PsychēChat in your research, please cite:

```bibtex
@article{xia2026psych,
  title={Psych$\backslash$= eChat: An Empathic Framework Focused on Emotion Shift Tracking and Safety Risk Analysis in Psychological Counseling},
  author={Xia, Zhentao and Fan, Yongqi and Chu, Yuxiang and Yin, Yichao and Chen, Liangliang and Ruan, Tong and Zhang, Weiyan},
  journal={arXiv preprint arXiv:2601.12392},
  year={2026}
}
```