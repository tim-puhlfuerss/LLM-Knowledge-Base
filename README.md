# LLM Knowledge Base

Last update: 2024/05/13 (YYYY/MM/DD)

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
- DeepSeek
  - Chinese Start-up
  - DeepSeek V2 is an open-source ChatGPT-like model published in May 2025
  - [Website](https://www.deepseek.com)
  - [GitHub](https://github.com/deepseek-ai/DeepSeek-VL)
  - [Paper pre-print](https://doi.org/10.48550/arXiv.2403.05525) on Arxiv
  - [Chat demo](https://huggingface.co/spaces/deepseek-ai/DeepSeek-VL-7B) on Hugging Face
  - [News article 1](https://www.heise.de/news/ChatGPT-4-Konkurrent-aus-China-DeepSeek-V2-ist-Open-Source-9713482.html) by Heise and [article 2](https://the-decoder.de/deepseek-v2-ist-das-neue-mixture-of-experts-spitzenmodell/) by The Decoder
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
- IBM
  - Granite
  - Open-source model for programming tasks
  - [Website](https://research.ibm.com/blog/granite-code-models-open-source)
  - [Paper pre-print](https://doi.org/10.48550/arXiv.2405.04324) on Arxiv
  - [Models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330) on Hugging Face
  - [News article](https://the-decoder.de/ibms-granite-code-schlaegt-alle-anderen-open-source-codiermodelle/) by The Decoder
- Aider:
  - [Official website](https://aider.chat)
  - [GitHub repo](https://github.com/paul-gauthier/aider)
  - Send code change request via the CLI or a [GUI](https://aider.chat/2024/05/02/browser.html) to a context-aware GPT model, which automatically conducts the change and creates a git commit
- SWE Agent:
  - <https://the-decoder.de/swe-agent-freie-alternative-zu-ki-entwickler-devin-von-cognition-ai/>
- Cognition AI:
  - Devin
  - Open-source version: [Devika](https://github.com/stitionai/devika)
- Google:
  - Gemini Code Assist (e.g., as VS Code extension)

### UI Testing

- Apple: Ferret UI
  - News article: <https://www.heise.de/news/Frettchen-voraus-Apple-KI-Ferret-UI-will-bei-App-Bedienung-helfen-9680770.html>
  - Pre-print: <https://doi.org/10.48550/arXiv.2404.05719>

### Images

- Ideogram AI:
  - Create an image that represents a concept or visualizes a text based on a text prompt 
  - [Website](https://ideogram.ai/login)
  - [News article](https://the-decoder.de/bild-ki-ideogram-bekommt-ein-update-und-ist-wirklich-richtig-gut/) by The Decoder
  - Offers “negative prompts” to exclude certain elements from the image
  - Offers a “describe” feature to generate image captions
- Stability AI: Stable Diffusion (z.B. SDXL)
- Midjourney
- Leonardo.ai
- OpenAI: DALL-E
- Google: Imagen 2
- Meta: Imagine (based on Emu model)
  - News article: <https://www.heise.de/news/Meta-bringt-mit-Imagine-eigenstaendigen-KI-Bildgenerator-9566722.html>
- Magnific AI:
  - Image resolution upscaler
  - [Website](https://magnific.ai)
  - Was acquired by Freepik in May 2024 ([News article](https://the-decoder.de/ki-upscale-start-up-magnific-ai-schafft-exit-mit-aussergewoehnlicher-gruendergeschichte/) by The Decoder)  
- Amazon: Titan
  - News article: <https://www.heise.de/news/Titan-fuer-Geschaeftskunden-Amazon-mit-eigenem-Modell-fuer-KI-Bildgeneratoren-9544056.html>
- Stable Diffusion GUIs:
  - Automatic1111
    - [GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  - Fooocus
    - [GitHub](https://github.com/lllyasviel/Fooocus)
  - ComfyUI
    - [GitHub](https://github.com/comfyanonymous/ComfyUI)
- Illusion Diffusion
  - Hide images (e.g., selfies) in other images (e.g., landscape)
  - [Hugging Face](https://huggingface.co/spaces/AP123/IllusionDiffusion)
  - [Website](https://illusiondiffusion.net)
- Story Diffusion
  - Create a comic-like storyboard
  - [Website](https://aistorydiffusion.com)

### Videos

- Open AI:
  - Sora
- Higgsfield AI:
  - [https://higgsfield.ai](https://higgsfield.ai/)
  - iOS app called Diffuse
  - [News article](https://www.heise.de/news/Diffuse-Ex-KI-Chef-von-Snap-veroeffentlicht-KI-Videogenerator-9677577.html) by Heise
- Runway:
  - [Gen-2](https://research.runwayml.com/gen2)
  - [News article](https://www.heise.de/tests/Bilder-in-Videos-verwandeln-Runway-Gen-2-im-Test-9572001.html) by Heise
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
 
### Specific Domains

#### Medicine

- AlphaFold 3 by Google for protein structures ([news article](https://www.heise.de/news/AlphaFold-3-Googles-KI-sagt-Struktur-aller-Molekuele-voraus-9713361.html) by Heise)

## In-Context Learning

- In-context learning (ICL) vs. fine-tuning: The former adds the examples to the prompt instead of training the model
- E.g., few-shot / many-shot prompting
- Much cheaper and less effortful
- Many-shot:
  - \>100 examples
  - In a test by Google, the performance of Gemini Pro 1.5 with many-shot prompts was significantly better than with few-shots
  - Paper: <https://arxiv.org/abs/2404.11018>
  - News article: <https://the-decoder.de/prompts-mit-vielen-beispielen-verbessern-die-leistung-grosser-sprachmodelle/>

## Multi-Agent LLMs

- Multi-agent LLMs enable automated generation of hypotheses, experiment design and simulation
  - Paper: <https://arxiv.org/abs/2404.11794>
  - News article: <https://the-decoder.de/llms-koennten-praezise-grossangelegte-sozialwissenschaftliche-experimente-ermoeglichen/>

## LLM Benchmarks

### Datasets

- MMLU (Measuring Massive Multitask Language Understanding)
- Open Medical-LLM Leaderboard
  - For health
  - By Hugging Face
  - Uses multiple health datasets
  - News article: <https://www.heise.de/news/LLMs-als-Arzthelfer-Benchmark-von-Hugging-Face-gibt-Zeugnisse-fuer-GPT-und-Co-9692203.html>

### Tools

- LMSYS Chatbot Arena
  - [Website](https://chat.lmsys.org)
  - Other projects by LMSYS on their [website](https://lmsys.org)
- Prometheus 2
  - Open-source LLM to benchmark other LLMs (similar to a human assessment)
  - [GitHub](https://github.com/prometheus-eval/prometheus-eval)
  - [Hugging Face](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0)
  - [News article](https://the-decoder.de/open-source-llm-prometheus-2-soll-andere-sprachmodelle-bewerten-und-verbessern/) by The Decoder
  - Note: GPT and other LLMs can do this, too. But Prometheus is open-source and reaches a similar performance

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

### Amazon Q

- [Blog article](https://www.aboutamazon.com/news/aws/amazon-q-generative-ai-assistant-aws) by Amazon
- [News article](https://the-decoder.de/amazon-veroeffentlicht-business-chatbot-q-mit-neuen-funktionen/) by The Decoder

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

### GPT4All

- Open-source ChatGPT-like UI wrapper to interact with open-source LLM models like Mistral-7b
- Focus on end-user devices (low amount of RAM and storage)
- Visit the [official website](https://gpt4all.io/index.html) or the [GitHub repo](https://github.com/nomic-ai/gpt4all?tab=readme-ov-file) for more info and installation instructions

### Microsoft's TypeChat

- News article: <https://www.heise.de/news/LLM-Ausgaben-strukturieren-Microsoft-Library-TypeChat-0-1-0-nutzt-TypeScript-9666782.html>
- Purpose: Ensures that LLM output is always returned in the same JSON format
- How it works:
  - You pre-define a schema for the output format (e.g., a JSON schema) and send your 'simple' prompt (as it no longer contains the format assertions) and the schema to the TypeChat function.
  - TypeChat then builds the prompt, sends it to the LLM (e.g., GPT), and returns the LLM output.
- Repo: <https://github.com/microsoft/TypeChat/tree/main?tab=readme-ov-file>
- Examples and introduction: <https://microsoft.github.io/TypeChat/docs/introduction/>

## Pinokio

- Open-source application that lets you browse curated AI models, like SDXL, or related libraries, like Automatic1111, in a marketplace-like UI and install them on your machine
- [Official website](https://pinokio.computer)
- [GitHub repo](https://github.com/pinokiocomputer/pinokio)

## Responsible AI (Ethics, Security, etc.)

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

### Guidelines

- AI Guideline by German Data Protection Authority in May 2024:
  - [Guideline](https://www.datenschutzkonferenz-online.de/media/oh/20240506_DSK_Orientierungshilfe_KI_und_Datenschutz.pdf)
  - [Additional resolution](https://www.datenschutzkonferenz-online.de/media/dskb/20240503_DSK_Positionspapier_Zustaendigkeiten_KI_VO.pdf)
  - [News article 1](https://www.heise.de/news/Datenschutzkonferenz-gibt-Leitfaden-fuer-DSGVO-konforme-KI-Anwendungen-9709228.html) and [article 2](https://www.heise.de/news/AI-Act-Datenschuetzer-wollen-KI-Verordnung-in-Deutschland-durchsetzen-9713089.html) by Heise
- OpenAI Model Specs
  - Set of definitions how an LLM should behave in interacting with users
  - [Article](https://openai.com/index/introducing-the-model-spec/) by OpenAI 
  - [News article](https://the-decoder.de/openai-veroeffentlicht-erstmals-richtlinien-fuer-ki-modellverhalten/) by The Decoder

### Certification for AI Models

- In Germany by TÜV Nord ([news article](https://www.heise.de/news/KI-Update-Deep-Dive-Tuev-IT-ueber-KI-Zertifizierung-9706680.html))

### BYOAI

- In early 2024, many employees already use AI for their tasks at work without their managers knowing about it ("Bring Your Own AI")
- [Survey](https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/05/2024_Work_Trend_Index_Annual_Report_663d45200a4ad.pdf) by Microsoft and LinkedIn
- [News article](https://the-decoder.de/laut-microsoft-ist-ki-scham-am-arbeitsplatz-eine-sache/) by The Decoder

## Related 'Awesome' Lists

- <https://github.com/underlines/awesome-ml?tab=readme-ov-file>

## Misc

### More Security by Hierarchical Instructions

- Idea:
  - Prio 1 is the system message, prio 2 is the user message, and prio 3 is tool output (prompts taken from LLM outputs or other tools).
  - The model rejects a request in prio 2 or 3 if it misaligns with the instructions in prio 1.
  - News article: <https://the-decoder.de/openai-will-ki-sicherheit-mit-prompt-hierarchie-verbessern/>
  - Paper: <https://arxiv.org/abs/2404.13208>

### Digital Watermark for AI-Generated Content

- [C2PA](https://c2pa.org) standard to trace the origin of AI-generated content, e.g., images
- Founded by Adobe
- OpenAI is part of the C2PA committee and developed a "Detection Classifier" to detect images generated by DALL-E.
- [News article](https://the-decoder.de/openai-unterstuetzt-c2pa-standard-fuer-ki-bilder-und-veroeffentlicht-klassifikator/) by The Decoder
- TikTok also became part of it in May 2025 ([news article](https://www.heise.de/news/Tiktok-tritt-CAI-bei-C2PA-Kennzeichnung-fuer-KI-Inhalte-9713446.html) by Heise)
 
### Cooperation StackOverlow x OpenAI

- [News article](https://the-decoder.de/chatgpt-bekommt-durch-stack-overflow-integration-zugriff-auf-validiertes-entwickler-wissen/) by The Decoder (May 2024)
