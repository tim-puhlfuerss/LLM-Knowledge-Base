# LLM Knowledge Base

Last update: 2024/05/02 (YYYY/MM/DD)

## Purpose

The world of large language models (LLMs) is changing rapidly. Here is some help keeping track.

The LLM Knowledge Base is a personal collection of helpful information and sources about LLMs created by [tim-puhlfuerss](https://github.com/tim-puhlfuerss). 
This collection can help you explore the domain of LLMs and related research and toolkits.

While it may resemble one of those 'Awesome' lists, the LLM Knowledge Base is currently a personal compilation of notes and news, not claiming to be exhaustive, but aiming to provide you with a wealth of insights.

## LLM Models

### Text

- OpenAI
  - GPT
  - GPT-4V
    - Text + image to text
    - E.g., to generate code for a mock-up
- Meta
  - Llama3
  - LLMLingua-2
    - Compresses prompts up to 80% to make model calls more cost-efficient
    - Released on 19.03.2024
    - Pre-print: <https://doi.org/10.48550/arXiv.2403.12968>
    - News article: <https://the-decoder.de/neues-tool-von-microsoft-kann-ki-prompts-um-bis-zu-80-prozent-komprimieren/>
  - Open source
- Google
  - Gemini / Bard
- Anthropic
  - Claude 3 (<https://www.anthropic.com/claude>)
  - Advertizes the high security of Claude
- Reka AI
  - Multimodal input (text, image, video, audio) to text output
  - Models: Reka Core, Flash, Edge
  - Reka Core performs similarly to GPT-4 (April 2024).
  - Start-up founded by DeepMind, Google Brain, Baidu, and Meta researchers in 2023
  - [https://www.reka.ai](https://www.reka.ai/)
  - [https://showcase.reka.ai](https://showcase.reka.ai/)
  - News article: <https://the-decoder.de/reka-core-ist-das-naechste-multimodale-ki-modell-das-gpt-4-weniger-speziell-macht/>
- Mistral
  - Mistral-7B
  - Mixtral 8x22B (Mixture of Experts)
  - Open source
- Microsoft
  - Phi 3
  - “Small Language Model (SLM)”
  - Optimized for low resource consumption
    - Only 1.8 GB large
    - Reaches 12 tokens per second on an iPhone 14
  - News articles:
    - <https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/>
    - <https://the-decoder.de/microsofts-effizientes-ki-modell-phi-3-soll-metas-llama-3-und-kostenloses-chatgpt-schlagen/>
- Cognition AI
  - Devin
- xAI
  - Grok 1 (soon 2)
  - Grok 1.5 Vision (text + documents/images to text)
- Apple
  - OpenELM family
  - “Efficient Language Model” for edge devices
  - Preprint: <https://doi.org/10.48550/arXiv.2404.14619>
  - Model: <https://huggingface.co/apple/OpenELM>
- Aleph Alpha
  - From Heidelberg (Germany)
  - [https://aleph-alpha.com](https://aleph-alpha.com/)
  - <https://github.com/Aleph-Alpha>
  - Research organization focusing on LLMs, especially concerning GDPR-save usage
  - Model family: Luminous (<https://docs.aleph-alpha.com/docs/introduction/luminous/>)

### Programming

- GitHub:
  - Copilot
  - Copilot Workspaces
    - Motto: “From feature proposal to code”
    - News article: <https://www.heise.de/news/GitHub-veroeffentlicht-KI-gestuetzte-Entwickungsumgebung-Copilot-Workspaces-9698383.html>
- GitLab:
  - Duo (similar to Copilot)
  - Include rights management on the manager level to restrict the tool from reading sensitive data
  - News article: <https://www.heise.de/news/Duo-GitLab-gibt-KI-Developer-Chat-offiziell-fuer-Pro-Nutzer-frei-9690844.html>
- Meta
  - Code Llama
  - <https://llama.meta.com/code-llama/>
- SWE Agent:
  - <https://the-decoder.de/swe-agent-freie-alternative-zu-ki-entwickler-devin-von-cognition-ai/>
- Cognition AI: Devin
- Google: Gemini Code Assist (e.g., as VS Code extension)

### UI Testing

- Apple: Ferret UI
  - News article: <https://www.heise.de/news/Frettchen-voraus-Apple-KI-Ferret-UI-will-bei-App-Bedienung-helfen-9680770.html>
  - Pre-print: <https://doi.org/10.48550/arXiv.2404.05719>

### Images

- Ideogram AI:
  - <https://ideogram.ai/login>
  - News article: <https://the-decoder.de/bild-ki-ideogram-bekommt-ein-update-und-ist-wirklich-richtig-gut/>
  - Offers “negative prompts” to exclude certain elements from the image
  - Offers a “describe” feature to generate image captions
- Stability AI: Stable Diffusion (z.B. SDXL)
- Midjourney
- Leonardo.ai
- OpenAI: DALL-E
- Google: Imagen 2
- Meta: Imagine (based on Emu model)
  - News article: <https://www.heise.de/news/Meta-bringt-mit-Imagine-eigenstaendigen-KI-Bildgenerator-9566722.html>
- Amazon: Titan
  - News article: <https://www.heise.de/news/Titan-fuer-Geschaeftskunden-Amazon-mit-eigenem-Modell-fuer-KI-Bildgeneratoren-9544056.html>

### Videos

- Open AI: Sora
- Higgsfield AI:
  - [https://higgsfield.ai](https://higgsfield.ai/)
  - iOS app Diffuse
  - News article: <https://www.heise.de/news/Diffuse-Ex-KI-Chef-von-Snap-veroeffentlicht-KI-Videogenerator-9677577.html>
- Microsoft:
  - VASA:
    - Text + face image to animated face video
    - Not public due to deepfake potential
    - News article: <https://the-decoder.de/vasa-1-microsoft-zeigt-lebensechte-ki-avatare-in-echtzeit/>

### Audio

- Stability AI: Stable Audio 2 ([https://stableaudio.com](https://stableaudio.com/))
- Google: Gemini
- Suno:
  - [https://suno.com](https://suno.com/)
  - Generate songs, incl. lyrics
  - E.g., the MIT License as ballad: <https://suno.com/song/da6d4a83-1001-4694-8c28-648a6e8bad0a>
- Udio:
  - Generate songs, incl. lyrics
  - Support for multiple languages
  - [https://www.udio.com](https://www.udio.com/)
  - News article: <https://www.heise.de/news/Kuenstliche-Intelligenz-Udio-Mit-wenigen-Stichworten-zum-eigenen-Song-9681717.html>

## In-Context Learning

- In-context learning (ICL) vs. fine-tuning: The former adds the examples to the prompt instead of training the model
- E.g., few-shot / many-shot prompting
- Much cheaper and less effortful
- Many-shot:
  - >100 examples
  - In a test by Google, the performance of Gemini Pro 1.5 with many-shot prompts was significantly better than with few-shots
  - Paper: <https://arxiv.org/abs/2404.11018>
  - News article: <https://the-decoder.de/prompts-mit-vielen-beispielen-verbessern-die-leistung-grosser-sprachmodelle/>

## Multi-Agent LLMs

- Multi-agent LLMs enable automated generation of hypotheses, experiment design and simulation
  - Paper: <https://arxiv.org/abs/2404.11794>
  - News article: <https://the-decoder.de/llms-koennten-praezise-grossangelegte-sozialwissenschaftliche-experimente-ermoeglichen/>

## LLM Benchmarks

- MMLU (Measuring Massive Multitask Language Understanding)
- Open Medical-LLM Leaderboard
  - For health
  - By Hugging Face
  - Uses multiple health datasets
  - News article: <https://www.heise.de/news/LLMs-als-Arzthelfer-Benchmark-von-Hugging-Face-gibt-Zeugnisse-fuer-GPT-und-Co-9692203.html>

## AI Tools and Platforms

### LLM Frameworks

- LangChain
<https://github.com/hwchase17/langchain>
- LlamaIndex
- Llama.cpp to run LLMs on end devices, like MacBooks (<https://github.com/ggerganov/llama.cpp>)

### Platforms

- Google: Vertex AI (<https://cloud.google.com/vertex-ai?hl=en>)

### Meta AI

- Llama-based Assistant that refers to search engine results to provide up-to-date information
- Also integrated in search features of Facebook, Instagram, WhatsApp and Messenger

### Perplexity AI

- [https://www.perplexity.ai](https://www.perplexity.ai/)
- A search engine that summarizes the results of a search request in small text chunks, incl. hyperlinks to its sources.

## Embedding Models

### Models

- Snowflake
  - Arctic-embed-m
    - SOTA (April 2024)
    - <https://huggingface.co/Snowflake/snowflake-arctic-embed-m>
    - <https://www.snowflake.com/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/>
- OpenAI
  - OpenAI Text Embedding
- Google
  - Gecko

### Benchmarks

- MTEB

## Helpful Libraries

### Aider

- <https://github.com/paul-gauthier/aider>
- Send code change request via the CLI to a context-aware GPT model, which automatically conducts the change and creates a git commit

### Microsoft's TypeChat

- News article: <https://www.heise.de/news/LLM-Ausgaben-strukturieren-Microsoft-Library-TypeChat-0-1-0-nutzt-TypeScript-9666782.html>
- Purpose: Ensures that LLM output is always returned in the same JSON format
- How it works:
  - You pre-define a schema for the output format (e.g., a JSON schema) and send your 'simple' prompt (as it no longer contains the format assertions) and the schema to the TypeChat function.
  - TypeChat then builds the prompt, sends it to the LLM (e.g., GPT), and returns the LLM output.
- Repo: <https://github.com/microsoft/TypeChat/tree/main?tab=readme-ov-file>
- Examples and introduction: <https://microsoft.github.io/TypeChat/docs/introduction/>

### LLM Security Libraries

- Blog article: <https://machine-learning-made-simple.medium.com/7-methods-to-secure-llm-apps-from-prompt-injections-and-jailbreaks-11987b274012>
- Rebuff:
  - Prompt injection detection
  - <https://github.com/protectai/rebuff>
- NeMo Guardrails
  - Add guardrails to conversational agents
  - <https://github.com/NVIDIA/NeMo-Guardrails>
- LangKit
  - Monitor LLMs and prevent prompt attacks
  - <https://github.com/whylabs/langkit?tab=readme-ov-file>
- LLM Guard
  - Detect harmful language, prevent data leakage and prompt injection
  - <https://github.com/protectai/llm-guard>
- LVE Repository
  - Listing of LLM vulnerabilities
  - <https://github.com/lve-org/lve>

## Related 'Awesome' Lists

- <https://github.com/underlines/awesome-ml?tab=readme-ov-file>

## Misc

### More Security by Hierarchical Instructions

- Idea:
  - Prio 1 is the system message, prio 2 is the user message, and prio 3 is tool output (prompts taken from LLM outputs or other tools).
  - The model rejects a request in prio 2 or 3 if it misaligns with the instructions in prio 1.
  - News article: <https://the-decoder.de/openai-will-ki-sicherheit-mit-prompt-hierarchie-verbessern/>
  - Paper: <https://arxiv.org/abs/2404.13208>
