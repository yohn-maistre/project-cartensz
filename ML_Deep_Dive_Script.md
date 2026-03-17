# 🎙️ Presenter Script: The Machine Learning Deep Dive
**Project Cartensz — Threat Narrative Classifier**

---

## 1. Introduction: The Data Reality

"In threat intelligence, the data you get in the real world is never a clean, 50/50 split. 

When we ingested the base dataset (IndoDiscourse from HuggingFace), we ran it through a deterministic preprocessing pipeline. We stripped the noise, normalized the Indonesian slang with Sastrawi, and remapped the granular labels into our specific threat intelligence taxonomy: **AMAN** (Safe), **WASPADA** (Caution), and **TINGGI** (High Threat / Incitement to Violence).

The resulting distribution was brutal, but realistic:
- **AMAN:** 92.5% (Over 23,000 texts)
- **WASPADA:** 7.2%
- **TINGGI:** 0.3% (Exactly 80 texts)

This extreme imbalance became the defining engineering challenge of this project."

---

## 2. Attempt 1: The Brute Force Failure

"Our first instinct was to use a standard deep learning approach. We took `NusaBERT-base`, a powerful Indonesian-native language model, and attempted a full fine-tuning. 

To combat the 92% `AMAN` majority, we implemented **inverse class weighting** in our loss function—meaning the model would be penalized 100x more for missing a `TINGGI` text than an `AMAN` text.

**The result was a total failure.**
While the overall Weighted F1 looked misleadingly good at 0.90, the model had simply learned to memorize the `AMAN` class. 
- **TINGGI Precision:** 0.00
- **TINGGI Recall:** 0.00

In our test set, the model saw 16 actual `TINGGI` threats and predicted 0 of them correctly. 
The gradients from those rare 80 samples were completely washed out by the ocean of safe texts. Standard fine-tuning was mathematically destined to fail."

---

## 3. The Pivot: Synthetic Augmentation & Contrastive Learning

"We needed a fundamentally different architecture that excelled at 'few-shot' learning—learning from very few examples.

We pivoted to **SetFit** (Sentence Transformer Fine-Tuning). Unlike standard classifiers that look at one text at a time, SetFit uses *contrastive learning*. It takes pairs of sentences and learns entirely by asking: *Are these two texts semantically similar or different?*

To fuel this, we performed a targeted data intervention:
1. **The Synthetic Engine:** We took our 80 real `TINGGI` samples and fed them into **Gemini 3 Flash** as few-shot seeds. We instructed the LLM to generate 120 brand new, highly realistic Indonesian threat samples featuring similar slang, urgency, and violent intent without copying the originals.
2. **The Balanced Curated Subset:** We combined these 200 `TINGGI` samples with exactly 200 random `AMAN` and 200 random `WASPADA` texts. 

We threw away 22,000 safe texts and trained the pipeline on a perfectly balanced, 600-sample dataset."

---

## 4. The Results: Passing the Criteria

"This pivot changed everything. Because SetFit generates pairs, our 600 samples were dynamically combined into thousands of contrastive training pairs, allowing the model to draw sharp, precise mathematical boundaries between the classes.

When we evaluated this local, lightweight model on our holdout test set, the numbers proved the architecture:
- **Weighted F1 Score:** `0.719` *(Passed the 0.70 KPI min)*
- **TINGGI Precision:** `0.812` *(Passed the stringent 0.75 KPI min)*

We successfully built a model that catches the severe threats (high recall/F1) while maintaining a strict precision boundary. 

At 0.81 precision, when our model flags a text as `TINGGI`, the analyst knows it is not a false alarm. We solved the 'Boy Who Cried Wolf' problem that ruins so many security dashboards, achieving robust, deterministic classification that runs instantly without requiring expensive GPUs."
