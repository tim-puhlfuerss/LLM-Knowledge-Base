# 🌍 Foundation Models

## Reasoning Focus

### OpenAI

- GPT-o1
  - Focused on reasoning.
  - Requires longer response times, but shall provide responses of higher quality
 
### DeepSeek

- DeepSeek-R1
  - [News article](https://www.heise.de/news/DeepSeek-R1-Neues-KI-Sprachmodell-mit-Reasoning-aus-China-gegen-OpenAI-o1-10082110.html) by Heise (2024-11-21)

## Multimodal Input / Output

### OpenAI

- GPT family
- GPT-4o
  - Text + image + audio to text + image + audio
  - "omni model" for multi-modal interaction
  - E.g., to generate code for a mock-up
  - News articles: [Heise](https://www.heise.de/news/OpenAI-Keine-Suche-kein-GPT-5-aber-GPT-4o-fuer-ChatGPT-und-GPT-4-9716626.html), [The Decoder](https://the-decoder.de/gpt-4o-diese-beeindruckenden-faehigkeiten-stecken-noch-im-neuen-ki-modell-von-openai/)

### Meta

- Chameleon
  - [Pre-print](https://doi.org/10.48550/arXiv.2405.09818) on arXiv
  - News articles [1](https://the-decoder.de/chameleon-meta-stellt-den-vorlaeufer-seiner-gpt-4o-alternative-vor/) and [2](https://the-decoder.de/meta-veroeffentlicht-neue-ki-modelle-fuer-text-bild-und-audio/) by The Decoder (2024-05-17, 2024-06-19)
  - [Announcement](https://about.fb.com/news/2024/06/releasing-new-ai-research-models-to-accelerate-innovation-at-scale/) by Meta (2024-06-18)

### Microsoft

- Phi-3-Vision
  - SLM
  - Text and image input
  - [Announcement](https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/) by Microsoft

### [Krea AI](https://www.krea.ai/home)

- Generate images and videos via text and/or image input
- [News article](https://the-decoder.de/krea-ai-veroeffentlicht-ki-videogenerator-mit-keyframe-unterstuetzung/) by The Decoder about the video generation feature

## Text Input / Output

### Meta

- Llama3
- LLMLingua-2
  - Compresses prompts up to 80% to make model calls more cost-efficient
  - Released on 2024-03-19
  - [Pre-print](https://doi.org/10.48550/arXiv.2403.12968) on arXiv
  - [News article](https://the-decoder.de/neues-tool-von-microsoft-kann-ki-prompts-um-bis-zu-80-prozent-komprimieren/) by The Decoder
- Open source

### Google

- Gemini (formerly Bard)
- Gemma
  - Open-source model
  - [News article](https://www.heise.de/news/KI-Update-kompakt-Proteinrevolution-Meta-AI-in-Berlin-Gemma-2-9781101.html) by Heise (2024-06-27) 

### Anthropic

- [Claude 3](https://www.anthropic.com/claude)
- Advertizes the high security of Claude
- Claude 3 can access external APIs, incl. databases, and cab work with agents internally ([announcement](https://www.anthropic.com/news/tool-use-ga) by Anthropic, [news article](https://the-decoder.de/anthropic-erweitert-claude-3-mit-tool-unterstuetzung-und-ki-agenten/) by The Decoder)

### [Reka AI](https://www.reka.ai/)

- Multimodal input (text, image, video, audio) to text output
- Models: Reka Core, Flash, Edge
- Reka Core performs similarly to GPT-4 (April 2024).
- Start-up founded by DeepMind, Google Brain, Baidu, and Meta researchers in 2023
- [Showcase](https://showcase.reka.ai/) on their website
- [News article](https://the-decoder.de/reka-core-ist-das-naechste-multimodale-ki-modell-das-gpt-4-weniger-speziell-macht/) by The Decoder

### Mistral

- Mistral-7B
- Mixtral 8x22B (Mixture of Experts)
- Open source

### Microsoft

- Phi 3
  - “Small Language Model (SLM)”
  - Optimized for low resource consumption
    - Only 1.8 GB large
    - Reaches 12 tokens per second on an iPhone 14
  - [News article](https://the-decoder.de/microsofts-effizientes-ki-modell-phi-3-soll-metas-llama-3-und-kostenloses-chatgpt-schlagen/) by The Decoder
  - [Blog article](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) by Microsoft

### xAI

- Grok 1 (soon 2)
- Grok 1.5 Vision (text + documents/images to text)

### Apple

- OpenELM family
- “Efficient Language Model” for edge devices
- [Pre-print](https://doi.org/10.48550/arXiv.2404.14619) on arXiv
- [Model](https://huggingface.co/apple/OpenELM) on Hugging Face

### DeepSeek

- Chinese Start-up
- DeepSeek V2 is an open-source ChatGPT-like model published in May 2025
- [Website](https://www.deepseek.com)
- [GitHub](https://github.com/deepseek-ai/DeepSeek-VL)
- [Paper pre-print](https://doi.org/10.48550/arXiv.2403.05525) on arXiv
- [Chat demo](https://huggingface.co/spaces/deepseek-ai/DeepSeek-VL-7B) on Hugging Face
- [News article 1](https://www.heise.de/news/ChatGPT-4-Konkurrent-aus-China-DeepSeek-V2-ist-Open-Source-9713482.html) by Heise and [article 2](https://the-decoder.de/deepseek-v2-ist-das-neue-mixture-of-experts-spitzenmodell/) by The Decoder

### Qwen / Alibaba

- Open-source LLM Qwen
- Current model, Qwen 2, performs better than other OSS models.
- [Hugging Face](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)
- [GitHub](https://github.com/QwenLM/Qwen2)
- [News article](https://the-decoder.de/qwen2-setzt-neue-massstaebe-bei-open-source-sprachmodellen/) by The Decoder, 2024-07-11

### [Aleph Alpha](https://aleph-alpha.com/)

- From Heidelberg (Germany)
- [GitHub](https://github.com/Aleph-Alpha)
- Research organization focusing on LLMs, especially concerning GDPR-save usage in industry
- Model family: [Luminous](https://docs.aleph-alpha.com/docs/introduction/luminous/)

## Programming/Coding Focus

### GitHub

- Copilot
- Copilot Workspaces
  - Motto: “From feature proposal to code”
  - [News article](https://www.heise.de/news/GitHub-veroeffentlicht-KI-gestuetzte-Entwickungsumgebung-Copilot-Workspaces-9698383.html) by Heise

### GitLab

- Duo (similar to Copilot)
- Includes rights management on the manager level to restrict the tool from reading sensitive data
- [News article](https://www.heise.de/news/Duo-GitLab-gibt-KI-Developer-Chat-offiziell-fuer-Pro-Nutzer-frei-9690844.html) by Heise

### OpenAI

- CriticGPT
  - GPT-based model to find errors in (generated) code
  - [News article](https://www.heise.de/news/CriticGPT-OpenAI-will-mit-kritischer-GPT-4-Version-Fehler-in-ChatGPT-finden-9781973.html) by Heise (2024-06-28)
  - [Blog post](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/) by OpenAI (2024-06-27)
  - [Paper](https://doi.org/10.48550/arXiv.2407.00215) on arXiv

### DeepSeek

- DeepSeek-Coder-V2
  - Programming-focused open-source model that achieves similar coding performance like GPT-4 Turbo.
  - [News article](https://the-decoder.de/deepseek-coder-v2-open-source-modell-schlaegt-gpt-4-und-claude-opus/) by The Decoder (2024-06-18)
  - [GitHub repo](https://github.com/deepseek-ai/DeepSeek-Coder-V2/tree/main), incl. pre-print paper

### Mistal

- [Codestral](https://mistral.ai/news/codestral/)
- Outperforms CodeLlama, Llama 3 and DeepSeek Coder V1
- [News article](https://the-decoder.de/mistral-stellt-neues-code-modell-vor-das-effizienter-und-kompetenter-als-llama-3-sein-soll/) by The Decoder

### Meta

- [Code Llama](https://llama.meta.com/code-llama/)

### IBM

- Granite
- Open-source model for programming tasks
- [Blog article](https://research.ibm.com/blog/granite-code-models-open-source) by IBM
- [Paper pre-print](https://doi.org/10.48550/arXiv.2405.04324) on arXiv
- [Models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330) on Hugging Face
- [News article](https://the-decoder.de/ibms-granite-code-schlaegt-alle-anderen-open-source-codiermodelle/) by The Decoder

### [Aider](https://aider.chat)

- [GitHub repo](https://github.com/paul-gauthier/aider)
- Send code change request via the CLI or a [GUI](https://aider.chat/2024/05/02/browser.html) to a context-aware GPT model, which automatically conducts the change and creates a git commit

### SWE Agent

- [News article](https://the-decoder.de/swe-agent-freie-alternative-zu-ki-entwickler-devin-von-cognition-ai/) by The Decoder

### Cognition AI

- Devin
- Open-source fork: [Devika](https://github.com/stitionai/devika)

### Google

- Gemini Code Assist (e.g., as VS Code extension)

## UI Testing

### Apple

- Ferret UI
  - Explain elements of an iOS app
  - [News article](https://www.heise.de/news/Frettchen-voraus-Apple-KI-Ferret-UI-will-bei-App-Bedienung-helfen-9680770.html) by Heise
  - [Pre-print](https://doi.org/10.48550/arXiv.2404.05719) on arXiv

## Image Output

### Ideogram AI

- Create an image that represents a concept or visualizes a text based on a text prompt
- [Website](https://ideogram.ai/login)
- [News article](https://the-decoder.de/bild-ki-ideogram-bekommt-ein-update-und-ist-wirklich-richtig-gut/) by The Decoder
- Offers “negative prompts” to exclude certain elements from the image
- Offers a “describe” feature to generate image captions

### Stability AI

- Stable Diffusion (z.B. SDXL)

### Midjourney

### Leonardo.ai

### OpenAI

- DALL-E

### Google

- Imagen

### Meta

- Imagine (based on Emu model)
- [News article](https://www.heise.de/news/Meta-bringt-mit-Imagine-eigenstaendigen-KI-Bildgenerator-9566722.html) by Heise

### [Magnific AI](https://magnific.ai)

- Image resolution upscaler
- Can also replace image backgrounds ([news article](https://the-decoder.de/magnific-ais-relight-tauscht-beleuchtung-und-hintergruende-in-bildern-per-ki-prompt-aus/) by The Decoder (2024-05-23)
- Was acquired by Freepik in May 2024 ([News article](https://the-decoder.de/ki-upscale-start-up-magnific-ai-schafft-exit-mit-aussergewoehnlicher-gruendergeschichte/) by The Decoder)  

### Amazon

- Titan
- [News article](https://www.heise.de/news/Titan-fuer-Geschaeftskunden-Amazon-mit-eigenem-Modell-fuer-KI-Bildgeneratoren-9544056.html) by Heise

### Stable Diffusion GUIs

- Automatic1111
  - [GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Fooocus
  - [GitHub](https://github.com/lllyasviel/Fooocus)
- ComfyUI
  - [GitHub](https://github.com/comfyanonymous/ComfyUI)

### Illusion Diffusion

- Hide images (e.g., selfies) in other images (e.g., landscape)
- [Hugging Face](https://huggingface.co/spaces/AP123/IllusionDiffusion)
- [Website](https://illusiondiffusion.net)

### Story Diffusion

- Create a comic-like storyboard
- [Website](https://aistorydiffusion.com)

## Video Input/Output

### Open AI

- Sora

### Luma AI

- Dream Machine
  - [News article](https://the-decoder.de/luma-ai-veroeffentlicht-ki-videogenerator-dream-machine-als-kostenlose-testversion/#google_vignette) by The Decoder (2024-06-12)
  - [Announcement](https://blog.lumalabs.ai/p/dream-machine) by Luma AI

### [Higgsfield AI](https://higgsfield.ai/)

- iOS app called Diffuse
- [News article](https://www.heise.de/news/Diffuse-Ex-KI-Chef-von-Snap-veroeffentlicht-KI-Videogenerator-9677577.html) by Heise

### RunwayML

- Gen-3 Alpha
  - Text2Video, Image2Video, Text2Image
  - [Announcement](https://runwayml.com/blog/introducing-gen-3-alpha/) by RunwayML (2024-06-17)
  - [News article](https://the-decoder.de/runway-gen-3-alpha-neues-videomodell-verkleinert-luecke-zu-openais-sora/) by The Decoder (2024-06-17)
- [Gen-2](https://research.runwayml.com/gen2)
  - Image2Video
  - [News article](https://www.heise.de/tests/Bilder-in-Videos-verwandeln-Runway-Gen-2-im-Test-9572001.html) by Heise

### Google

- V2A
  - Video2Audio
  - [Announcement](https://deepmind.google/discover/blog/generating-audio-for-video/) by Google Deepmind (2024-06-17)
  - [News article](https://the-decoder.de/deepmind-zeigt-v2a-ki-generiert-passenden-sound-fuer-stumme-videos/) by The Decoder (2024-06-18)

### Microsoft

- VASA:
  - Text + face image to animated face video
  - Not publicly available due to deepfake potential
  - [News article](https://the-decoder.de/vasa-1-microsoft-zeigt-lebensechte-ki-avatare-in-echtzeit/) by The Decoder

## Audio Output

### Stability AI

- [Stable Audio 2](https://stableaudio.com/))
  - Text-to-music (up to 3-minutes)
- [Stable Audio Open]
  - Generates up to 47-seconds-long audio samples and sound effects
  - Open-source model
  - [Announcement](https://stability.ai/news/introducing-stable-audio-open) by Stability AI
  - [News article](https://the-decoder.de/stable-audio-open-ist-ein-open-source-ki-modell-fuer-geraeusche-und-sounddesign/) by The Decoder

### [Elevenlabs](https://elevenlabs.io/)

- Text-to-speech and speech-to-speech with multiple languages ([news article](https://the-decoder.de/sprach-ki-service-elevenlabs-stellt-neues-multisprachen-modell-vor/) by The Decoder)
- Text to sound effect ([announcement](https://elevenlabs.io/blog/sound-effects-are-here/) by Elevenlabs, [news article](https://the-decoder.de/text-zu-boom-ki-sprachgenerator-elevenlabs-unterstuetzt-jetzt-soundeffekte/) by The Decoder)

### [Camb AI](https://www.camb.ai/)

- Mars5
  - Text-to-speech with support of 140 languages
  - [News article](https://the-decoder.de/mars5-soll-elevenlabs-schlagen-und-ist-fuer-englisch-kostenlos/) by The Decoder (2024-06-17)
  - [GitHub repo](https://github.com/Camb-ai/MARS5-TTS) limited to English language

### Google

- Gemini

### [Suno]((https://suno.com/))

- Text-to-music, incl. lyrics
- E.g., [the MIT License as a ballad](https://suno.com/song/da6d4a83-1001-4694-8c28-648a6e8bad0a)

### [Udio](https://www.udio.com/)

- Text-to-music, incl. lyrics
- Support for multiple languages
- [News article](https://www.heise.de/news/Kuenstliche-Intelligenz-Udio-Mit-wenigen-Stichworten-zum-eigenen-Song-9681717.html)

### [Jen](https://www.jenmusic.ai)

- Model Jen [ALPHA]
- Text-to-music (just instrumentals, up to 40 seconds)
- Trained on a fully licensed music dataset
- Stores the IDs of generated songs in a blockchain to solve legal conflicts (duplication, original authors, etc.)
- [News article](https://the-decoder.de/neue-text-zu-audio-ki-jen-verspricht-eigene-songs-ohne-copyright-aerger/) by The Decoder (2024-06-22)

### Meta

- JASCO
  - Text-to-music
  - [Announcment](https://about.fb.com/news/2024/06/releasing-new-ai-research-models-to-accelerate-innovation-at-scale/) by Meta (2024-06-18)
  - [News article](https://the-decoder.de/meta-veroeffentlicht-neue-ki-modelle-fuer-text-bild-und-audio/) by the Decoder (2024-06-19)

## Specific Domains

### Medicine

- Google's AlphaFold 3 generates protein structures ([news article](https://www.heise.de/news/AlphaFold-3-Googles-KI-sagt-Struktur-aller-Molekuele-voraus-9713361.html) by Heise)

### Legal and Compliance

- creance.ai
  - Joint venture by PwC and Aleph Alpha that will provide AI solutions for legal and compliance consulting
  - [Announcement](https://www.pwc.de/de/pressemitteilungen/2024/pwc-deutschland-und-aleph-alpha-grunden-joint-venture-creanceai.html) by PwC
  - [News article](https://www.heise.de/news/PwC-und-Aleph-Alpha-gruenden-Joint-Venture-fuer-KI-Loesungen-in-Rechtsberatung-9750737.html) by Heise
