# 👷 Prompt Engineering

## Definition

- Prompt engineering enabled by the model's ability of in-context learning (ICL)
- ICL vs. fine-tuning: The former adds the examples to the prompt instead of training the model
- E.g., few-shot / many-shot prompting
- In-context learning is much cheaper and less effortful than fine-tuning.
- Many-shot:
  - \>100 examples
  - In a test by Google, the performance of Gemini Pro 1.5 with many-shot prompts was significantly better than with few-shots.
  - [Pre-print](https://arxiv.org/abs/2404.11018) by Agarwal et al. on arXiv
  - [News article](https://the-decoder.de/prompts-mit-vielen-beispielen-verbessern-die-leistung-grosser-sprachmodelle/) by The Decoder

## Prompt Engineering Guidelines and Overview Papers

- The Prompt Report: A Systematic Survey of Prompting Techniques (Schulhoff et al., 2024)
  - [Pre-print](https://doi.org/10.48550/arXiv.2406.06608) on arXiv
  - [Project website](https://trigaten.github.io/Prompt_Survey_Site/)
  - [News article](https://the-decoder.de/der-prompt-report-ist-ein-umfassender-prompting-ueberblick-mit-kuriosen-erkenntnissen/) by The Decoder (2024/06/15)
- [Prompt Engineering Guide](https://www.promptingguide.ai) (Elvis Saravia)
  - Definitions and examples for various prompt engineering techniques
