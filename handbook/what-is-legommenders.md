# What is Legommenders?

Legommenders is an open-source library for content-based recommendation that supports “Lego-style” modular composition 🧱. It enables joint training of content encoders and user behavior models, integrating content understanding seamlessly into the recommendation pipeline. Researchers and developers can use Legommenders to easily assemble **thousands of different recommendation models** and run experiments across more than 15 datasets. Notably, Legommenders **pioneers integration with Large Language Models (LLMs)**, allowing LLMs to be used both as content encoders and as generators for data augmentation to build more personalized and effective recommenders 🎉.

## 🧠 Project Overview

The name "Legommenders" comes from "LEGO + Recommenders," symbolizing the idea of building recommendation models like Lego bricks 🫠. This project, proposed by The Hong Kong Polytechnic University, is the official implementation of the WWW 2025 paper, _Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support_. The key goal is to provide **a unified and flexible research framework** for content-driven recommendation. Traditional recommender systems often rely on static ID representations and struggle with cold-start problems. Legommenders focuses on content features (e.g., article texts, product descriptions) to enhance recommendations.

### Highlights:

- **Joint Modeling of Content & Behavior**: Supports end-to-end training of content encoders and user behavior models, ensuring content representations are task-aware.
- **Modular Design**: Provides LEGO-style composable modules for content processing, behavior modeling, prediction, etc.
- **Rich Built-in Models**: Includes classic models like NAML, NRMS, LSTUR, DeepFM, DCN, DIN, enabling rapid experimentation and comparison.
- **LLM Integration**: Enables LLMs (e.g., BERT, GPT, LLaMA) as content encoders or for data generation. Includes LoRA support for efficient fine-tuning.
- **Widely Adopted**: Already supports multiple research projects such as [ONCE](https://arxiv.org/abs/2305.06566), [SPAR](https://arxiv.org/abs/2402.10555),[GreenRec](https://arxiv.org/abs/2403.04736), and [UIST](https://arxiv.org/abs/2403.08206).