import { Module } from './types';

export const CURRICULUM: Module[] = [
  {
    "id": "mod-0",
    "title": "Module 0: Orientation & Environment Setup",
    "description": "Course overview, dev environment, verify GPU/Colab options, reproducibility checklist.",
    "lessons": [
      {
        "id": "0.1",
        "title": "Course Overview & Learning Path",
        "durationMinutes": 20,
        "difficulty": "Beginner",
        "description": "Understand the course structure, final projects, and how to set up your learning environment for success.",
        "objectives": [
          "Understand course outcomes and final projects",
          "Choose an environment (local GPU, Colab, or cloud)",
          "Get a 'reproducibility checklist' to run all labs"
        ],
        "resources": [
          {
            "type": "video",
            "title": "Course Intro (example) - short orientation",
            "url": "https://www.youtube.com/watch?v=g-gu4BJ6J9o",
            "author": "Hugging Face - beginner RAG intro video"
          }
        ],
        "concepts": ["Environment Setup", "Learning Path", "Reproducibility"],
        "codeSnippets": [
          {
            "language": "bash",
            "title": "Conda Environment Setup",
            "code": "conda create -n llm python=3.10 -y"
          }
        ],
        "exercise": {
          "description": "Create a Python environment (conda or venv) and install torch & transformers. Verify CUDA with `python -c \"import torch; print(torch.cuda.is_available())\"`.",
          "expectedOutput": "True (if GPU available) or successful import of torch."
        }
      },
      {
        "id": "0.2",
        "title": "Dev Environment Options: Local GPU vs Colab vs CPU",
        "durationMinutes": 30,
        "difficulty": "Beginner",
        "description": "Evaluate and configure your compute resources. Whether using Colab Free Tier or a local RTX card, we ensure you have the right drivers and libraries.",
        "objectives": [
          "Decide between Colab, local GPU, or cloud GPU",
          "Know how to configure GPU drivers and CUDA",
          "Install core libs: torch, transformers, accelerate"
        ],
        "resources": [
          {
            "type": "doc",
            "title": "PyTorch - Get Started Locally",
            "url": "https://pytorch.org/get-started/locally/",
            "author": "PyTorch"
          },
          {
            "type": "article",
            "title": "Hugging Face - Installation and Setup",
            "url": "https://huggingface.co/docs/transformers/installation",
            "author": "Hugging Face"
          },
          {
            "type": "video",
            "title": "Verify CUDA and Setup - short tutorial",
            "url": "https://www.youtube.com/watch?v=U0s0f995w14",
            "author": "PyTorch transformer setup demo"
          }
        ],
        "concepts": ["CUDA", "PyTorch", "Accelerate", "Google Colab"],
        "codeSnippets": [
          {
            "language": "bash",
            "title": "Install Core Libraries",
            "code": "pip install torch transformers accelerate datasets sentence-transformers faiss-cpu"
          }
        ],
        "exercise": {
          "description": "Run the CUDA check script and capture the output. If no GPU, spin up a Google Colab GPU runtime and verify libraries install.",
          "expectedOutput": "Installation successful log."
        }
      }
    ]
  },
  {
    "id": "mod-1",
    "title": "Module 1: Foundations: Transformers, Tokenization, and Embeddings",
    "description": "Core theory for transformers, tokenization types, and how to compute semantic embeddings.",
    "lessons": [
      {
        "id": "1.1",
        "title": "Understanding Transformers (Attention & Architecture)",
        "durationMinutes": 50,
        "difficulty": "Beginner-Intermediate",
        "description": "Deconstruct the Transformer architecture. We dive into Self-Attention, Multi-Head Attention, and the differences between Encoder and Decoder models.",
        "objectives": [
          "Explain self-attention, multi-head attention, and positional encoding",
          "Differentiate decoder-only vs encoder-only vs encoder-decoder models",
          "Understand tradeoffs for generation vs representation tasks"
        ],
        "resources": [
          {
            "type": "article",
            "title": "The Illustrated Transformer",
            "url": "https://jalammar.github.io/illustrated-transformer/",
            "author": "Jay Alammar"
          },
          {
            "type": "article",
            "title": "Transformers from Scratch",
            "url": "https://peterbloem.nl/blog/transformers",
            "author": "Peter Bloem"
          },
          {
            "type": "video",
            "title": "Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out",
            "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY",
            "author": "Andrej Karpathy"
          }
        ],
        "concepts": ["Self-Attention", "Encoder-Decoder", "Positional Encoding", "GPT vs BERT"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "Basic Transformer Inference",
            "code": "from transformers import AutoModelForCausalLM, AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained('gpt2')\nmodel = AutoModelForCausalLM.from_pretrained('gpt2')\ninputs = tokenizer('Hello', return_tensors='pt')\nprint(model(**inputs))"
          }
        ],
        "exercise": {
          "description": "Read Jay Alammar's post and run Karpathy's minimal example. Write a 1-paragraph summary of self-attention in your own words.",
          "expectedOutput": "A clear paragraph explaining how tokens attend to each other."
        }
      },
      {
        "id": "1.2",
        "title": "Tokenization Deep Dive (BPE, WordPiece, SentencePiece, tiktoken)",
        "durationMinutes": 45,
        "difficulty": "Intermediate",
        "description": "Tokens are not words. We explore Byte Pair Encoding (BPE), how vocabularies are built, and how to calculate token costs using `tiktoken`.",
        "objectives": [
          "Understand how tokenization affects model input length and cost",
          "Use tiktoken and Hugging Face tokenizers to count tokens",
          "Train or adapt a tokenizer if needed"
        ],
        "resources": [
          {
            "type": "doc",
            "title": "Hugging Face - Tokenizers (LLM Course)",
            "url": "https://huggingface.co/learn/llm-course/en/chapter2/4",
            "author": "Hugging Face"
          },
          {
            "type": "repo",
            "title": "openai/tiktoken",
            "url": "https://github.com/openai/tiktoken",
            "author": "OpenAI"
          },
          {
            "type": "tutorial",
            "title": "How to count tokens with tiktoken (OpenAI Cookbook)",
            "url": "https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken",
            "author": "OpenAI"
          }
        ],
        "concepts": ["BPE", "Tiktoken", "Vocabulary", "Context Window"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "Tiktoken Usage",
            "code": "import tiktoken\nenc = tiktoken.get_encoding('cl100k_base')\nprint(len(enc.encode('Hello world!')))"
          }
        ],
        "exercise": {
          "description": "Compare token counts for the same paragraph using GPT-2 tokenizer (gpt2), cl100k_base (OpenAI), and a SentencePiece model.",
          "expectedOutput": "Token counts for 3 different tokenizers."
        }
      },
      {
        "id": "1.3",
        "title": "Embeddings: Principles and Practical Usage",
        "durationMinutes": 60,
        "difficulty": "Intermediate",
        "description": "Embeddings convert text into vectors where semantic meaning is distance. We use SentenceTransformers to generate and visualize these embeddings.",
        "objectives": [
          "Understand semantic embeddings and vector similarity metrics",
          "Create embeddings with SentenceTransformers and Hugging Face models",
          "Visualize and evaluate embedding quality"
        ],
        "resources": [
          {
            "type": "video",
            "title": "Sentence Transformers tutorial",
            "url": "https://www.youtube.com/watch?v=OlhNZg4gOvA",
            "author": "James Briggs"
          },
          {
            "type": "docs",
            "title": "Sentence-Transformers docs",
            "url": "https://www.sbert.net/docs/pretrained_models.html",
            "author": "SBERT"
          },
          {
            "type": "article",
            "title": "Creating and Visualizing Embeddings",
            "url": "https://www.youtube.com/watch?v=5S2Yk45xMLM",
            "author": "Datacamp"
          }
        ],
        "concepts": ["Vector Space", "Cosine Similarity", "SentenceTransformers", "Dimensionality"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "Generate Embeddings",
            "code": "from sentence_transformers import SentenceTransformer\nmodel = SentenceTransformer('all-MiniLM-L6-v2')\nemb = model.encode(['This is a sentence','Another sentence'])\nprint(len(emb), type(emb))"
          }
        ],
        "exercise": {
          "description": "Embed a small set of documents and compute cosine similarities. Visualize using PCA or TSNE.",
          "expectedOutput": "A similarity matrix or visualization plot."
        }
      }
    ]
  },
  {
    "id": "mod-2",
    "title": "Module 2: Vector Databases & Indexing",
    "description": "FAISS/Chroma/Weaviate basics, chunking strategies, metadata indexing, persistence and hybrid search.",
    "lessons": [
      {
        "id": "2.1",
        "title": "FAISS Fundamentals & Local Vector Search",
        "durationMinutes": 45,
        "difficulty": "Intermediate",
        "description": "Learn to use FAISS for efficient similarity search. We cover index types like IndexFlatL2 and IVFPQ for optimization.",
        "objectives": [
          "Install FAISS and run simple nearest-neighbour search",
          "Understand index types (Flat, IVFPQ) and their tradeoffs",
          "Persist and load indices"
        ],
        "resources": [
          {
            "type": "docs",
            "title": "FAISS - Official README and tutorials",
            "url": "https://github.com/facebookresearch/faiss",
            "author": "Facebook Research"
          },
          {
            "type": "video",
            "title": "FAISS Tutorial - Basic to Intermediate",
            "url": "https://www.youtube.com/watch?v=tcqEUSNCn8I",
            "author": "James Briggs"
          }
        ],
        "concepts": ["ANN", "FAISS", "L2 Distance", "Index Persistence"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "FAISS Simple Search",
            "code": "import faiss\nimport numpy as np\nxb = np.random.random((1000, 128)).astype('float32')\nindex = faiss.IndexFlatL2(128)\nindex.add(xb)\nD, I = index.search(xb[:5], 5)\nprint(I)"
          }
        ],
        "exercise": {
          "description": "Build a FAISS index for the embeddings from Module 1 and run similarity queries.",
          "expectedOutput": "Indices of nearest neighbors returned."
        }
      },
      {
        "id": "2.2",
        "title": "Chroma & Weaviate: Vector DBs as a Service",
        "durationMinutes": 50,
        "difficulty": "Intermediate",
        "description": "Move beyond libraries to full Vector Databases. We explore ChromaDB for local development and Weaviate for production-grade hybrid search.",
        "objectives": [
          "Understand Chroma's persistence and collections model",
          "Learn Weaviate usage and hybrid semantic search",
          "Deploy Chroma locally or use managed Weaviate"
        ],
        "resources": [
          {
            "type": "docs",
            "title": "Chroma docs",
            "url": "https://www.trychroma.com/docs/",
            "author": "Chroma"
          },
          {
            "type": "docs",
            "title": "Weaviate docs",
            "url": "https://weaviate.io/developers",
            "author": "Weaviate"
          }
        ],
        "concepts": ["ChromaDB", "Weaviate", "Metadata Filtering", "Collections"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "ChromaDB Setup",
            "code": "from chromadb import Client\nclient = Client()\ncollection = client.create_collection('my_docs')\ncollection.add(ids=['1'], metadatas=[{'title':'a'}], embeddings=[[0.1]*1536], documents=['text'])"
          }
        ],
        "exercise": {
          "description": "Index the sample docs into Chroma and compare retrieval times with FAISS for 1k documents.",
          "expectedOutput": "Comparison of setup time and query latency."
        }
      },
      {
        "id": "2.3",
        "title": "Chunking, Embedding Granularity & Metadata",
        "durationMinutes": 40,
        "difficulty": "Intermediate",
        "description": "How you split text matters. We implement Sliding Window chunking and discuss how metadata impacts retrieval quality.",
        "objectives": [
          "Design chunking strategies (sentence, sliding windows)",
          "Choose embedding granularity for quality/latency tradeoffs",
          "Store metadata for contextual filtering"
        ],
        "resources": [
          {
            "type": "article",
            "title": "Building effective RAG systems",
            "url": "https://learnopencv.com/rag-with-llms/",
            "author": "LearnOpenCV"
          },
          {
            "type": "tutorial",
            "title": "Building YouTube RAG Chat - architecture",
            "url": "https://medium.com/@komai.fares.ww/building-youtube-rag-chat-talk-to-any-video-with-local-llms-00db6fa96e6c",
            "author": "Medium"
          }
        ],
        "concepts": ["Chunking", "Overlap", "Metadata", "Granularity"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "Sliding Window Chunking",
            "code": "def chunk_text(text, chunk_size=512, overlap=128):\n    tokens = tokenizer.encode(text)\n    chunks = []\n    for i in range(0, len(tokens), chunk_size-overlap):\n        chunks.append(tokenizer.decode(tokens[i:i+chunk_size]))\n    return chunks"
          }
        ],
        "exercise": {
          "description": "Implement a chunking function for long articles and test retrieval quality with different chunk sizes.",
          "expectedOutput": "Retrieval results for different chunk sizes."
        }
      }
    ]
  },
  {
    "id": "mod-3",
    "title": "Module 3: Building a Full RAG Pipeline",
    "description": "End-to-end RAG: ingest, transcribe (if video/audio), chunk, embed, index, retrieval, generation loop with a grounded LLM.",
    "lessons": [
      {
        "id": "3.1",
        "title": "RAG Architecture & Design Patterns",
        "durationMinutes": 45,
        "difficulty": "Intermediate",
        "description": "Architectural patterns for RAG. We discuss Bi-Encoders, Cross-Encoders, and the flow from Retrieval to Context Assembly to Generation.",
        "objectives": [
          "Understand bi-encoder retriever + cross-encoder reranker patterns",
          "Know how to pipeline retrieval → context assembly → generation",
          "Design caching & freshness strategies"
        ],
        "resources": [
          {
            "type": "article",
            "title": "Illustrated Retrieval Transformer (RETRO)",
            "url": "https://jalammar.github.io/illustrated-retrieval-transformer/",
            "author": "Jay Alammar"
          },
          {
            "type": "video",
            "title": "RAG From Scratch by LangChain (playlist)",
            "url": "https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x",
            "author": "LangChain"
          }
        ],
        "concepts": ["Bi-Encoder", "Cross-Encoder", "Context Assembly", "Hallucination"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "RAG Logic Flow (Pseudocode)",
            "code": "query = \"Question\"\nemb = embed(query)\ndocs = index.search(emb)\ncontext = \"\\n\".join([d.text for d in docs])\nprompt = f\"Context: {context}\\nQuestion: {query}\"\nanswer = llm.generate(prompt)"
          }
        ],
        "exercise": {
          "description": "Sketch an architecture for a RAG assistant for your documents. Include embedding model, vector DB choice, and fallback plan.",
          "expectedOutput": "Architecture diagram or text description."
        }
      },
      {
        "id": "3.2",
        "title": "Implement a Minimal RAG App (LangChain + FAISS)",
        "durationMinutes": 90,
        "difficulty": "Intermediate",
        "description": "Hands-on implementation of RAG using LangChain and FAISS. We build a simple app that answers questions based on a text file.",
        "objectives": [
          "Follow a video tutorial and build a minimal RAG app",
          "Run local retrieval and generate answers using a HF model",
          "Understand prompts for grounding"
        ],
        "resources": [
          {
            "type": "video",
            "title": "Complete RAG Crash Course With LangChain",
            "url": "https://www.youtube.com/watch?v=o126p1QN_RI",
            "author": "LangChain"
          },
          {
            "type": "repo",
            "title": "LangChain - Python examples",
            "url": "https://github.com/langchain-ai/langchain",
            "author": "LangChain"
          }
        ],
        "concepts": ["LangChain", "Prompt Engineering", "Retriever", "QA Chain"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "LangChain FAISS Setup",
            "code": "from langchain.embeddings import HuggingFaceEmbeddings\nfrom langchain.vectorstores import FAISS\n# Implementation follows standard LangChain patterns"
          }
        ],
        "exercise": {
          "description": "Clone the LangChain RAG example, run it locally, and add two documents to the index; then ask queries and verify grounded answers.",
          "expectedOutput": "Correct answers based on the added documents."
        }
      },
      {
        "id": "3.3",
        "title": "Transcribing Audio & Video for RAG (Whisper pipeline)",
        "durationMinutes": 60,
        "difficulty": "Intermediate",
        "description": "Expand RAG to multimedia. We use OpenAI's Whisper to transcribe audio/video content, chunk it, and index it for search.",
        "objectives": [
          "Transcribe audio using Whisper",
          "Chunk transcripts and index them",
          "Construct a RAG flow for video corpora"
        ],
        "resources": [
          {
            "type": "article",
            "title": "YouTube RAG Chat - Architecture",
            "url": "https://medium.com/@komai.fares.ww/building-youtube-rag-chat-talk-to-any-video-with-local-llms-00db6fa96e6c",
            "author": "Medium"
          },
          {
            "type": "tool",
            "title": "Whisper (OpenAI)",
            "url": "https://github.com/openai/whisper",
            "author": "OpenAI"
          }
        ],
        "concepts": ["Whisper", "Transcription", "Multimedia Indexing", "yt-dlp"],
        "codeSnippets": [
          {
            "language": "bash",
            "title": "Whisper Transcription",
            "code": "# yt-dlp -x --audio-format mp3 <youtube_url>\n# python -m whisper audio.mp3 --model small"
          }
        ],
        "exercise": {
          "description": "Download a 10-minute YouTube talk, transcribe it, chunk the transcript and verify retrieval quality in your local RAG app.",
          "expectedOutput": "Searchable transcript where queries return correct video timestamps/text."
        }
      }
    ]
  },
  {
    "id": "mod-4",
    "title": "Module 4: Fine-Tuning Techniques (LoRA, PEFT, QLoRA)",
    "description": "Hands-on fine-tuning: parameter-efficient methods (LoRA/PEFT), QLoRA for memory-limited GPUs, dataset prepping, training loops.",
    "lessons": [
      {
        "id": "4.1",
        "title": "Why Fine-Tune? Full vs Parameter-Efficient Approaches",
        "durationMinutes": 35,
        "difficulty": "Intermediate",
        "description": "Compare Full Fine-Tuning vs PEFT. We introduce LoRA (Low-Rank Adaptation) and explain why it saves massive amounts of VRAM.",
        "objectives": [
          "Understand full model fine-tuning vs LoRA/PEFT",
          "Know tradeoffs in compute, accuracy, and complexity",
          "Choose the right approach for a given dataset"
        ],
        "resources": [
          {
            "type": "repo",
            "title": "Hugging Face PEFT",
            "url": "https://github.com/huggingface/peft",
            "author": "Hugging Face"
          },
          {
            "type": "article",
            "title": "LoRA paper",
            "url": "https://www.microsoft.com/en-us/research/uploads/prod/2022/05/lora.pdf",
            "author": "Microsoft Research"
          }
        ],
        "concepts": ["PEFT", "LoRA", "Catastrophic Forgetting", "Adapters"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "PEFT Config",
            "code": "from peft import get_peft_model, LoraConfig\n# Basic LoRA config usage"
          }
        ],
        "exercise": {
          "description": "Read the PEFT repo README and identify which LLMs they support and example commands to apply LoRA.",
          "expectedOutput": "List of supported models and basic command."
        }
      },
      {
        "id": "4.2",
        "title": "Practical LoRA: Hands-on Notebook",
        "durationMinutes": 90,
        "difficulty": "Intermediate",
        "description": "Execute a LoRA fine-tuning run. We use a Google Colab notebook to train a small model using Hugging Face's PEFT library.",
        "objectives": [
          "Apply LoRA to a small HF causal model",
          "Train on a tiny dataset and evaluate",
          "Save and load LoRA adapters"
        ],
        "resources": [
          {
            "type": "video",
            "title": "LoRA / PEFT tutorial",
            "url": "https://www.youtube.com/watch?v=iOdFUJiB0Zc",
            "author": "Sam Witteveen"
          },
          {
            "type": "repo",
            "title": "Example LoRA fine-tuning notebook",
            "url": "https://github.com/huggingface/peft/tree/main/examples",
            "author": "Hugging Face"
          }
        ],
        "concepts": ["Training Loop", "Adapters", "Inference", "Saving Models"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "LoRA Training Snippet",
            "code": "from transformers import AutoModelForCausalLM, AutoTokenizer\nfrom peft import LoraConfig, get_peft_model\n# load model, apply lora config, tune with Trainer"
          }
        ],
        "exercise": {
          "description": "Run the example notebook in Colab (use small model) and report training loss progression for 1-2 epochs.",
          "expectedOutput": "Training logs showing decreasing loss."
        }
      },
      {
        "id": "4.3",
        "title": "QLoRA: 4-bit Fine-Tuning for Big Models",
        "durationMinutes": 75,
        "difficulty": "Advanced",
        "description": "Fine-tune 7B+ parameter models on consumer GPUs using QLoRA. We use 4-bit quantization and paged optimizers to reduce memory footprint.",
        "objectives": [
          "Understand quantization and QLoRA method",
          "Run a QLoRA flow on a 13B-ish model in Colab or small GPU node",
          "Evaluate memory vs accuracy tradeoffs"
        ],
        "resources": [
          {
            "type": "article",
            "title": "QLoRA tutorial / resources",
            "url": "https://github.com/tloen/qlora-notebooks",
            "author": "Tim Dettmers / Community"
          },
          {
            "type": "video",
            "title": "QLoRA walkthrough",
            "url": "https://www.youtube.com/watch?v=TPcXVJ1VSRI",
            "author": "AIAnytime"
          }
        ],
        "concepts": ["Quantization", "4-bit", "Double Quantization", "Paged Optimizers"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "QLoRA Steps (Pseudo)",
            "code": "# 1) load model with bitsandbytes\n# 2) apply 4-bit quantization\n# 3) attach LoRA adapter and train"
          }
        ],
        "exercise": {
          "description": "Try a QLoRA notebook and report the peak GPU memory usage. (Note: if no GPU, skip and read the tutorial carefully.)",
          "expectedOutput": "Memory usage stats (e.g. ~10GB for 7B model)."
        }
      }
    ]
  },
  {
    "id": "mod-5",
    "title": "Module 5: Evaluation, Optimization & Quantization",
    "description": "How to test, evaluate, quantize, and optimize models for inference: evaluation metrics, calibration, and smaller model options.",
    "lessons": [
      {
        "id": "5.1",
        "title": "Evaluation Metrics & Data Splits for LLMs",
        "durationMinutes": 40,
        "difficulty": "Intermediate",
        "description": "How do we know the model is getting better? We cover metrics like Perplexity, BLEU, ROUGE, and the emerging 'LLM-as-a-Judge' paradigm.",
        "objectives": [
          "Design evaluation datasets for RAG and fine-tuning",
          "Use metrics: EM, F1, BLEU (where relevant), and human eval proxies",
          "Run automated tests for hallucinations"
        ],
        "resources": [
          {
            "type": "article",
            "title": "Hugging Face - Evaluate library & metrics",
            "url": "https://huggingface.co/docs/evaluate/",
            "author": "Hugging Face"
          }
        ],
        "concepts": ["Evaluation", "Metrics", "Hallucination", "Test Splits"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "Using Evaluate Library",
            "code": "from evaluate import load\nmetric = load('accuracy')\nprint(metric.compute(predictions=[1,0], references=[1,0]))"
          }
        ],
        "exercise": {
          "description": "Prepare a 50-sample test set for your project and design metrics to detect hallucinations.",
          "expectedOutput": "A CSV/JSON evaluation dataset."
        }
      },
      {
        "id": "5.2",
        "title": "Quantization: bitsandbytes, GPTQ, and LLM inference tricks",
        "durationMinutes": 60,
        "difficulty": "Advanced",
        "description": "Make models smaller and faster. We cover 8-bit and 4-bit inference using `bitsandbytes` and `GPTQ`.",
        "objectives": [
          "Understand 8-bit/4-bit quantization methods",
          "Use bitsandbytes for 8-bit training and inference",
          "Learn about GPTQ and CPU inference speedups"
        ],
        "resources": [
          {
            "type": "repo",
            "title": "bitsandbytes",
            "url": "https://github.com/TimDettmers/bitsandbytes",
            "author": "Tim Dettmers"
          },
          {
            "type": "repo",
            "title": "GPTQ and quantization tools",
            "url": "https://github.com/qwopqwop200/GPTQ-for-LLaMa",
            "author": "Community"
          }
        ],
        "concepts": ["GPTQ", "bitsandbytes", "Inference Latency", "FP16 vs INT8"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "Loading in 8-bit",
            "code": "from transformers import AutoModelForCausalLM\nmodel = AutoModelForCausalLM.from_pretrained('model', load_in_8bit=True, device_map='auto')"
          }
        ],
        "exercise": {
          "description": "Quantize a small model to 8-bit and measure inference latency vs float32.",
          "expectedOutput": "Latency comparison table."
        }
      }
    ]
  },
  {
    "id": "mod-6",
    "title": "Module 6: RAG + Fine-Tuned Model — Integrations & Best Practices",
    "description": "Combine retrieval and fine-tuning: how to use a fine-tuned model as the generator in a RAG loop, prompt templates, and evaluation.",
    "lessons": [
      {
        "id": "6.1",
        "title": "Prompting with Retrieved Context & Prompt Templates",
        "durationMinutes": 45,
        "difficulty": "Intermediate",
        "description": "Structuring the prompt is key to RAG. We learn how to dynamically inject retrieved context into a PromptTemplate.",
        "objectives": [
          "Design prompt templates that include retrieved context",
          "Implement safety & fallback behaviors in prompts",
          "Understand token budgets and context windows"
        ],
        "resources": [
          {
            "type": "article",
            "title": "LangChain prompt templates docs",
            "url": "https://python.langchain.com/en/latest/modules/prompts.html",
            "author": "LangChain"
          }
        ],
        "concepts": ["Prompt Template", "Context Injection", "Safety", "Token Budget"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "LangChain PromptTemplate",
            "code": "from langchain import PromptTemplate\nprompt = PromptTemplate(input_variables=['context','question'], template='Context:\\n{context}\\nQ:{question}')"
          }
        ],
        "exercise": {
          "description": "Design two prompt templates: one conservative (safe answers) and one exploratory (creative answers). Test on same retrieval results and compare output.",
          "expectedOutput": "Comparison of outputs for the same query."
        }
      },
      {
        "id": "6.2",
        "title": "Using a LoRA fine-tuned generator in RAG",
        "durationMinutes": 75,
        "difficulty": "Advanced",
        "description": "The ultimate combo: RAG + LoRA. We load a LoRA adapter specifically trained for a domain and use it to generate answers from retrieved context.",
        "objectives": [
          "Load a base model and LoRA adapter, then use it in a RAG generation loop",
          "Compare results vs base model",
          "Understand adapter storage and sharing"
        ],
        "resources": [
          {
            "type": "repo",
            "title": "Example: saving and loading LoRA adapters",
            "url": "https://github.com/huggingface/peft",
            "author": "Hugging Face"
          }
        ],
        "concepts": ["Adapter Loading", "PeftModel", "Domain Adaptation"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "Loading Adapter",
            "code": "from transformers import AutoModelForCausalLM\nfrom peft import PeftModel\nbase = AutoModelForCausalLM.from_pretrained('...')\nadapter = PeftModel.from_pretrained(base, 'path_to_adapter')"
          }
        ],
        "exercise": {
          "description": "Fine-tune a tiny adapter and compare retrieval-grounded answers between base and adapter-enhanced model.",
          "expectedOutput": "A/B test results of base vs adapter."
        }
      }
    ]
  },
  {
    "id": "mod-7",
    "title": "Module 7: Deployment & Packaging",
    "description": "Practical deployment: FastAPI + Docker, Hugging Face Inference, small self-hosted options (ollama/llama.cpp/vLLM) and serving RAG APIs.",
    "lessons": [
      {
        "id": "7.1",
        "title": "Serving LLMs: FastAPI + Docker + simple REST",
        "durationMinutes": 80,
        "difficulty": "Intermediate",
        "description": "Wrap your model in a REST API using FastAPI and containerize it with Docker for deployment anywhere.",
        "objectives": [
          "Create a simple FastAPI endpoint that uses a HF model",
          "Containerize the app with Docker",
          "Deploy to a lightweight VPS or test locally"
        ],
        "resources": [
          {
            "type": "tutorial",
            "title": "Deploy a model with FastAPI and Docker",
            "url": "https://www.youtube.com/watch?v=g-gu4BJ6J9o",
            "author": "Tutorial"
          },
          {
            "type": "docs",
            "title": "Hugging Face - Inference API and deployment notes",
            "url": "https://huggingface.co/docs/inference-api",
            "author": "Hugging Face"
          }
        ],
        "concepts": ["FastAPI", "Docker", "REST API", "Containerization"],
        "codeSnippets": [
          {
            "language": "python",
            "title": "FastAPI Skeleton",
            "code": "from fastapi import FastAPI\nfrom transformers import pipeline\napp = FastAPI()\ngen = pipeline('text-generation', model='gpt2')\n@app.post('/generate')\ndef generate(payload: dict):\n    return gen(payload['prompt'], max_length=100)"
          }
        ],
        "exercise": {
          "description": "Create a Dockerfile for the FastAPI app and run it locally. Then call the endpoint with curl.",
          "expectedOutput": "Successful JSON response from localhost."
        }
      },
      {
        "id": "7.2",
        "title": "Low-cost self-hosting options: ollama, llama.cpp, vLLM",
        "durationMinutes": 50,
        "difficulty": "Advanced",
        "description": "Explore high-performance inference servers. We look at `llama.cpp` for CPU/Apple Silicon and `vLLM` for high-throughput GPU serving.",
        "objectives": [
          "Know options for local CPU-based inference (llama.cpp)",
          "Understand ollama and other local inference tooling",
          "Understand vLLM for high-throughput GPU inference"
        ],
        "resources": [
          {
            "type": "repo",
            "title": "llama.cpp",
            "url": "https://github.com/ggerganov/llama.cpp",
            "author": "Georgi Gerganov"
          },
          {
            "type": "repo",
            "title": "vLLM",
            "url": "https://github.com/vllm-project/vllm",
            "author": "vLLM Team"
          },
          {
            "type": "tool",
            "title": "Ollama docs (self-host local inference)",
            "url": "https://ollama.com/docs",
            "author": "Ollama"
          }
        ],
        "concepts": ["GGUF", "vLLM", "Ollama", "Local Inference"],
        "codeSnippets": [
          {
            "language": "bash",
            "title": "llama.cpp Usage",
            "code": "./main -m ./models/ggml-model.bin -p \"Hello\" -n 128"
          }
        ],
        "exercise": {
          "description": "Try loading a small quantized GGML model with llama.cpp and measure latency on your machine.",
          "expectedOutput": "Inference speed (tokens/sec)."
        }
      }
    ]
  },
  {
    "id": "mod-8",
    "title": "Module 8: Final Projects & Autonomous Practice",
    "description": "Three end-to-end projects that combine all prior modules. Each project includes repos, videos, data, and step-by-step instructions.",
    "lessons": [
      {
        "id": "8.1",
        "title": "Project 1 — Basic RAG Chatbot",
        "durationMinutes": 180,
        "difficulty": "Intermediate",
        "description": "Build a document-backed chatbot using SentenceTransformers, FAISS/Chroma, and a HF causal model for generation. Deploy locally via FastAPI.",
        "objectives": [
          "Build a complete RAG pipeline",
          "Index 500+ documents",
          "Deploy an API endpoint"
        ],
        "resources": [
          {
            "type": "video",
            "title": "LangChain RAG tutorial",
            "url": "https://www.youtube.com/watch?v=o126p1QN_RI",
            "author": "LangChain"
          },
          {
            "type": "repo",
            "title": "LangChain Examples (RAG)",
            "url": "https://github.com/langchain-ai/langchain",
            "author": "LangChain"
          }
        ],
        "concepts": ["End-to-End", "RAG", "Project"],
        "codeSnippets": [],
        "exercise": {
          "description": "1) Transcribe or prepare documents\n2) Chunk & embed (SentenceTransformers)\n3) Index with FAISS or Chroma\n4) Build retrieval + prompt assembly\n5) Use HF model as generator\n6) Deploy with FastAPI + Docker",
          "expectedOutput": "Running server with /ask endpoint. Indexed document set (500 docs). Evaluation script for retrieval accuracy."
        }
      },
      {
        "id": "8.2",
        "title": "Project 2 — LoRA Fine-Tune a Model on Domain Data",
        "durationMinutes": 240,
        "difficulty": "Advanced",
        "description": "Fine-tune a base causal model on domain Q/A or instruction dataset using LoRA/PEFT. Evaluate against base model.",
        "objectives": [
          "Curate a domain-specific dataset",
          "Fine-tune using LoRA",
          "Evaluate performance improvement"
        ],
        "resources": [
          {
            "type": "repo",
            "title": "Hugging Face PEFT examples",
            "url": "https://github.com/huggingface/peft",
            "author": "Hugging Face"
          },
          {
            "type": "video",
            "title": "LoRA fine-tuning tutorial",
            "url": "https://www.youtube.com/watch?v=iOdFUJiB0Zc",
            "author": "Sam Witteveen"
          }
        ],
        "concepts": ["Fine-Tuning", "LoRA", "Domain Adaptation"],
        "codeSnippets": [],
        "exercise": {
          "description": "1) Collect/clean dataset (JSONL or text pairs)\n2) Create HF dataset and dataloader\n3) Use Trainer or accelerate to run PEFT loop\n4) Save adapter and run inference with adapter loaded",
          "expectedOutput": "Trained adapter files. Evaluation script comparing base vs fine-tuned. Notebook showing training steps."
        }
      },
      {
        "id": "8.3",
        "title": "Project 3 — Production-ready RAG + Fine-Tuned LLM",
        "durationMinutes": 420,
        "difficulty": "Advanced",
        "description": "Combine Project 1 + 2: use a domain fine-tuned adapter as the generator inside a RAG system, optimize quantization, and deploy with Docker. Include monitoring & simple automated tests for hallucinations.",
        "objectives": [
          "Integrate Fine-Tuned model into RAG",
          "Optimize for production (quantization)",
          "Deploy full stack with monitoring"
        ],
        "resources": [
          {
            "type": "article",
            "title": "Productionizing RAG architectures",
            "url": "https://learnopencv.com/rag-with-llms/",
            "author": "LearnOpenCV"
          },
          {
            "type": "repo",
            "title": "Example production RAG + LoRA repo",
            "url": "https://github.com/your-org/example-rag-lora",
            "author": "Community Example"
          }
        ],
        "concepts": ["Production", "Integration", "Monitoring"],
        "codeSnippets": [],
        "exercise": {
          "description": "1) Reproduce Project 1 & 2 locally\n2) Measure and reduce latency (quantization)\n3) Build Dockerfile and compose.yml for service + vectorstore (or persist to disk)\n4) Add a simple health-check and smoke-test suite\n5) Document deployment steps (README)",
          "expectedOutput": "Dockerized service. Adapter + index repo with reproducible script. Monitoring sketch (simple logs + test queries)."
        }
      }
    ]
  },
  {
    "id": "mod-9",
    "title": "Module 9: Advanced Topics & Further Reading",
    "description": "Optional deep dives: RLHF basics, multimodal RAG, retrieval for images/video, privacy & data governance, prompt engineering at scale.",
    "lessons": [
      {
        "id": "9.1",
        "title": "RLHF: Concepts and High-level Flow",
        "durationMinutes": 60,
        "difficulty": "Advanced",
        "description": "Introduction to Reinforcement Learning from Human Feedback (RLHF). Understand the Reward Model, PPO, and how it shapes model behavior.",
        "objectives": [
          "Understand RLHF conceptually (reward model, policy, human-in-the-loop)",
          "Know open-source projects and how RLHF fits with fine-tuning"
        ],
        "resources": [
          {
            "type": "article",
            "title": "High-level RLHF overview",
            "url": "https://openai.com/research/rlhf",
            "author": "OpenAI"
          }
        ],
        "concepts": ["RLHF", "PPO", "Reward Model"],
        "codeSnippets": [],
        "exercise": {
          "description": "Sketch an RLHF pipeline for a specific use-case and list required resources.",
          "expectedOutput": "Pipeline diagram or description."
        }
      },
      {
        "id": "9.2",
        "title": "Multimodal Retrieval & RAG for Video",
        "durationMinutes": 70,
        "difficulty": "Advanced",
        "description": "Moving beyond text. How to index and retrieve video and audio content using multimodal embeddings.",
        "objectives": [
          "Index video transcripts + images; use multimodal embeddings",
          "Build RAG for video corpora"
        ],
        "resources": [
          {
            "type": "article",
            "title": "YouTube RAG Chat blog",
            "url": "https://medium.com/@komai.fares.ww/building-youtube-rag-chat-talk-to-any-video-with-local-llms-00db6fa96e6c",
            "author": "Medium"
          }
        ],
        "concepts": ["Multimodal", "Video RAG"],
        "codeSnippets": [],
        "exercise": {
          "description": "Index a set of conference videos and build a search + question answering flow.",
          "expectedOutput": "Searchable video index."
        }
      }
    ]
  }
];