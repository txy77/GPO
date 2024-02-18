# Unleashing the Potential of Large Language Models as Prompt Optimizers: An Analogical Analysis with Gradient-based Model Optimizers

This repo provides the source code & data of our paper: Unleashing the Potential of Large Language Models as Prompt Optimizers: An Analogical Analysis with Gradient-based Model Optimizers.



## 😀 Overview

**Highlights**:

- 1️⃣ We are the first to conduct a systematic study for LLM-based prompt optimizers.
- 💡 By drawing inspiration from gradient-based model optimization techniques, we develop a capable Gradient-inspired LLM based Prompt Optimizer called GPO.
- 🔝 GPO brings an additional improvement of up to 56.8% on Big-Bench Hard and 55.3% on MMLU compared to baseline methods.

We propose a novel perspective to investigate the design of LLM-based prompt optimizers, by drawing an analogy with gradient-based model optimizers. By systematically analyzing a rich set of improvement strategies, we further develop a capable Gradient-inspired LLM-based Prompt Optimizer called GPO. At each step, it first retrieves relevant prompts from the optimization trajectory as the update direction. Then, it utilizes the generation-based refinement strategy to perform the update, while controlling the edit distance through a cosine-based decay strategy.