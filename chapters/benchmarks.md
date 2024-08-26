# 📊 LLM Benchmarks

## Benchmark Datasets

### MMLU

- Measuring Massive Multitask Language Understanding

### Open Medical-LLM Leaderboard

- For health
- By Hugging Face
- Uses multiple health datasets
- [News article](https://www.heise.de/news/LLMs-als-Arzthelfer-Benchmark-von-Hugging-Face-gibt-Zeugnisse-fuer-GPT-und-Co-9692203.html) by Heise

## Benchmark Tools

### [LMSYS Chatbot Arena](https://chat.lmsys.org)

- Other projects by LMSYS on their [website](https://lmsys.org)

### [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)

- Ratings of AlpacaEval correlate strongly with human evaluation.

### Prometheus 2

- Open-source LLM to benchmark other LLMs (similar to a human assessment)
- [GitHub](https://github.com/prometheus-eval/prometheus-eval)
- [Hugging Face](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0)
- [News article](https://the-decoder.de/open-source-llm-prometheus-2-soll-andere-sprachmodelle-bewerten-und-verbessern/) by The Decoder
- Note: GPT and other LLMs can benchmark other LLMs, too. But Prometheus is open-source and reaches a similar performance.

## Synthetical Dataset Generators

### Nvidia

- Nemotron
  - Open-source model family to generate synthetical datasets
  - [Announcement](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/) by Nvidia (2024/06/14)
  - [News article](https://the-decoder.de/nvidia-veroeffentlicht-kostenlose-sprachmodelle-optimiert-fuer-die-datengenerierung/) by The Decoder (2024/06/15)

## Critique on LLM Benchmarks

- Paper ["Alice in Wonderland"](https://doi.org/10.48550/arXiv.2406.02061) by Nezhurina et al. (June 2024)
  - Current LLMs fail in reasoning tasks, even if they are very simple.
  - Current benchmarks (also the reasoning-related ones) do not sufficiently detect such flaws
  - [News article](https://www.heise.de/news/Reasoning-Fail-Gaengige-LLMs-scheitern-an-kinderleichter-Aufgabe-9755034.html) by Heise
