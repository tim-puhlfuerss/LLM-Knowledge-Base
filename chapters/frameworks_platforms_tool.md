# 🛠️ LLM Frameworks, Platforms, and Tools

## Frameworks

- [LangChain](https://github.com/hwchase17/langchain)
- LlamaIndex
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) to run LLMs on end devices, like MacBooks

## Platforms

- [Azure](https://azure.microsoft.com/en-us/) by Microsoft
- [Vertex AI](https://cloud.google.com/vertex-ai) by Google
- [Lightning AI](https://lightning.ai) offers on-cloud programming, AI training, and hosting of AI web apps via their Studio application.

## Tools

### Microsoft's AI Toolkit for VS Code

- VS Code extension to explore, try, fine-tune, and integrate various models (incl. GPT, Llama, Mistral) into applications.
- [Blog post](https://techcommunity.microsoft.com/t5/microsoft-developer-community/announcing-the-ai-toolkit-for-visual-studio-code/ba-p/4146473) by Microsoft
- [Extension in the VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio&ssr=false#overview)
- [Blog post about the Azure AI Studio](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/shaping-tomorrow-developing-and-deploying-generative-ai-apps/ba-p/4143017) related to the AI Toolkit by Microsoft
- [News article](https://www.heise.de/news/Microsoft-veroeffentlicht-KI-Erweiterung-AI-Toolkit-fuer-Visual-Studio-Code-9726476.html) by Heise

### Meta AI

- Llama-based Assistant that refers to search engine results to provide up-to-date information
- Also integrated in search features of Facebook, Instagram, WhatsApp and Messenger

### [Perplexity]((https://www.perplexity.ai/))

- A search engine that summarizes the results of a search request in small text chunks, incl. hyperlinks to its sources.

### [Perplexica](https://github.com/ItzCrazyKns/Perplexica)

- Open-source, self-hosted alternative to Perplexity
- [News article](https://the-decoder.de/perplexica-ist-eine-open-source-ki-suchmaschine-als-alternative-zu-perplexity/) by The Decoder, 2024/06/10

### [GPT4All]((https://gpt4all.io/index.html))

- Open-source ChatGPT-like UI wrapper to interact with open-source LLM models like Mistral-7b
- Focus on end-user devices (low amount of RAM and storage)
- [GitHub repo](https://github.com/nomic-ai/gpt4all?tab=readme-ov-file)

### Microsoft's TypeChat

- Purpose: Ensures that LLM output is always returned in the same JSON format
- How it works:
  - You pre-define a schema for the output format (e.g., a JSON schema) and send your 'simple' prompt (as it no longer contains the format assertions) and the schema to the TypeChat function.
  - TypeChat then builds the prompt, sends it to the LLM (e.g., GPT), and returns the LLM output.
- [News article](https://www.heise.de/news/LLM-Ausgaben-strukturieren-Microsoft-Library-TypeChat-0-1-0-nutzt-TypeScript-9666782.html) by Heise
- [GitHub repo](https://github.com/microsoft/TypeChat/tree/main?tab=readme-ov-file)
- [Examples and introduction](https://microsoft.github.io/TypeChat/docs/introduction/)

### Pinokio

- Open-source application that lets you browse curated AI models, like SDXL, or related libraries, like Automatic1111, in a marketplace-like UI and install them on your machine
- [Official website](https://pinokio.computer)
- [GitHub repo](https://github.com/pinokiocomputer/pinokio)

### Amazon's Q

- [Blog article](https://www.aboutamazon.com/news/aws/amazon-q-generative-ai-assistant-aws) by Amazon
- [News article](https://the-decoder.de/amazon-veroeffentlicht-business-chatbot-q-mit-neuen-funktionen/) by The Decoder

### Retrieval-Augmented Generation (RAG) Tools

- [Khoj](https://khoj.dev) (open-source tool)
- [Nuclia](https://nuclia.com)
- [Tutorial](https://youtu.be/Yhtjd7yGGGA?si=tn6ZZjkliSK_bE27) by Rabbit Hole Syndrom on YouTube how to build a RAG with OpenAI Embeddings, Supabase, and pgvector
