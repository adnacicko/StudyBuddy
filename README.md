# STUDY BUDDY: A HYBRID RULE-BASED NLP PIPELINE FOR AUTOMATIC QUIZ QUESTION GENERATION FROM LECTURE SLIDES

A natural language processing pipeline that extracts clean sentences from a PDF document, analyses the text, and automatically generates question-answer pairs using spaCy dependency parsing and NER.

> **Dataset note:** The source PDF used to produce the sample outputs (`Introduction.pdf`, a Machine Learning introductory text) is also available in this repository alongside the notebook.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Quick Start](#quick-start)
6. [Pipeline Walkthrough](#pipeline-walkthrough)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Sample Output](#sample-output)
9. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project demonstrates an end-to-end pipeline for:

1. **Extracting** raw text from a multi-page PDF.
2. **Cleaning & segmenting** the text into high-quality, self-contained sentences.
3. **Analysing** the corpus (sentence length, word frequency, bigrams, etc.).
4. **Generating** factoid question-answer pairs from every sentence using:
   - Syntactic dependency analysis (subject / object detection)
   - Named-entity recognition (ORG, PRODUCT, EVENT)
   - Noun-chunk extraction
   - Semantic question templates (*What is …?*, *How does … work?*, *What is … used for?*, …)
5. **Evaluating** the output with automatic metrics (coverage, concept precision, grammatical quality).

The notebook was developed and tested on **Google Colab** but is compatible with any Python 3.8+ environment that can install the listed dependencies.

---

## Features

| Feature | Details |
|---|---|
| PDF text extraction | `pdfplumber` with per-page character-count logging |
| Multi-stage text cleaning | Whitespace normalisation, bullet-symbol removal, sentence capitalisation, length filtering |
| SpaCy sentence tokenisation | `en_core_web_sm` model |
| Heading–content merging | Automatically joins short heading fragments with the following content sentence |
| Corpus statistics | Sentence / word / bigram frequency analysis with visualisations |
| Concept extraction | Dependency-parsed objects/subjects + NER + noun chunks with a scoring heuristic |
| Question generation | Six semantic templates selected by lemma matching |
| Answer extraction | Dependency-tree verb-phrase extraction with string-split fallback |
| Evaluation | Coverage rate, concept precision, LanguageTool grammatical quality score |
| Failure analysis | Automatic flagging of empty / very short answers |

---

## Repository Structure

```
.
├── Project.ipynb          # Main notebook (commented version)
├── Introduction.pdf       # Dataset – source PDF used to generate sample outputs
└── README.md              # This file
```

---

## Requirements

Install all dependencies by running the first cell of the notebook, or manually:

```bash
pip install PyPDF2 pdfplumber nltk spacy language-tool-python \
            transformers torch sentence-transformers keybert
python -m spacy download en_core_web_sm
```

NLTK data packages required (downloaded automatically inside the notebook):

- `punkt` / `punkt_tab`
- `stopwords`
- `averaged_perceptron_tagger`

> **LanguageTool** (grammatical quality metric) requires a Java runtime. It is downloaded automatically by `language-tool-python` on first use. If Java is unavailable, the metric is silently skipped.

---

## Quick Start

1. **Clone** this repository.
2. Open `Project.ipynb` in **Google Colab** (recommended) or a local Jupyter environment.
3. Upload your PDF, or use the supplied `Introduction.pdf`.
4. Update the `pdf_path` variable in **Section 6** (cell `nlp = spacy.load(…)`) to point to your file:
   ```python
   pdf_path = "/content/YourDocument.pdf"   # Colab
   # pdf_path = "./YourDocument.pdf"         # local
   ```
5. **Run all cells** (`Runtime → Run all` in Colab).

The notebook will print extracted sentences, generated QA pairs, and evaluation metrics, and display several charts.

---

## Pipeline Walkthrough

### Section 1 – Dependencies & Imports
Installs all libraries and imports standard + third-party modules.

### Section 2 – PDF Text Extraction
`extract_text_from_pdf(pdf_path)` opens the PDF with `pdfplumber` and concatenates the text of every page.

### Section 3 – Text Cleaning
Four layered cleaning functions:
- **`clean_for_spacy`** – pre-processing before spaCy (whitespace, bullet symbols).
- **`clean_sentence_after_tokenization`** – post-tokenisation normalisation (lower-case, strip symbols).
- **`clean_sentence_formatting`** – formatting pass (capitalisation, punctuation, length normalisation).
- **`final_clean_sentences`** – final filter: 6–30 words, ends with `.` or `!`.

### Section 4 – Sentence Quality Filters
Boolean predicates: `is_meaningful_sentence`, `is_heading_sentence`, `is_definition_heading`, `is_good_quality`.

### Section 5 – Sentence Merging
Short heading fragments are merged with their following content sentence using `merge_related_sentences` (unconditional) and `improved_merge_sentences` (relatedness-aware).

### Section 6 – Main Processing Pipeline
Orchestrates all previous utilities: extract → pre-clean → spaCy tokenise → post-clean → filter → word-freq → merge (2 passes) → final filter.

### Sections 7–8 – Corpus Statistics & Visualisations
Sentence length, word length, word frequency, bigram frequency – all with charts.

### Sections 9–10 – QA Generation
- **`advanced_concept_extraction`** – spaCy dependency + NER + noun chunks with scoring.
- **`semantic_question_generation`** – template selection by lemma matching.
- **`extract_semantic_answer`** – dependency verb-phrase extraction with fallback.
- **`nlp_qa_generation`** – orchestrator; returns a list of QA pair dicts.

### Sections 11–15 – Evaluation & Analysis
Coverage rate, concept precision, grammatical quality, question-type pie chart, answer-length histogram, failure analysis, question-vs-answer scatter plot.

---

## Evaluation Metrics

| Metric | Definition | Expected value |
|---|---|---|
| **Coverage rate** | QA pairs generated / input sentences | ≥ 95 % |
| **Concept precision** | Questions containing the concept string / total QA pairs | ≈ 100 % |
| **Grammatical quality** | LanguageTool error-based score (0–1) | ≥ 0.90 |

Sample results on `Introduction.pdf` (11 pages, Machine Learning overview):

| Metric | Value |
|---|---|
| Input sentences after cleaning | 123 |
| QA pairs generated | 120 |
| Coverage rate | 97.6 % |
| Concept precision | 100.0 % |
| Grammatical quality | 0.95 / 1.0 |

---

## Sample Output

```
Q: What is Machine learning?
A: ml) is increasingly relied upon because it can solve complex problems that...
Concept: Machine learning  |  Source sentence: Machine learning (ml) is increasingly...

Q: What is reinforcement learning?
A: an agent interacts with an environment and learns by receiving rewards or...
Concept: reinforcement learning  |  Source sentence: In reinforcement learning, an agent...

Q: What is unsupervised learning used for?
A: helps us discover patterns in data without using labeled outcomes
Concept: unsupervised learning  |  Source sentence: Unsupervised learning examples...
```

---

## Limitations & Future Work

- **Answer quality** – The dependency-based extractor sometimes returns partial phrases or falls back to the raw string after the concept. A span-extraction QA model (e.g. BERT-based) would improve precision.
- **Question diversity** – Only six templates are used. Expanding to *why*, *when*, *who*, and *which* questions would broaden coverage.
- **Language** – The pipeline is English-only; replacing `en_core_web_sm` with a multilingual model would enable other languages.
- **PDF layout** – Tables and multi-column layouts can produce garbled extraction; adding a layout-aware PDF parser (e.g. `pymupdf`) would help.
- **Heading detection** – The keyword-based heuristic for heading detection is fragile; a classifier trained on heading patterns would be more robust.
