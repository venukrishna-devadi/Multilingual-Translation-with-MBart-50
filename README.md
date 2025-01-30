# Multilingual-Translation-with-MBart-50
The project aimed to enable seamless translation from English to Indian languages using the multilingual capabilities of MBart-50. ver 1 million bilingual sentence pairs were preprocessed with language-specific tokens like [en_XX] for English, [te_IN] for Telugu, and [hi_IN] for Hindi, ensuring accurate context-aware translations.

## 📖 Background & Motivation  
With **LLMs evolving rapidly**, understanding how these models work is crucial for staying relevant in AI. Instead of just using pretrained models, this project **dives into the mechanics of multilingual transformers**, training MBart-50 on a **custom dataset** to improve translation accuracy.  

## 🛠️ Features  
- ✅ **Fine-tuned MBart-50** for multilingual translation  
- ✅ **Handles English, Telugu, and Hindi** with language-specific tokens  
- ✅ **Efficient preprocessing** with custom tokenization  
- ✅ **Inference setup** for real-time multilingual translation  
- ✅ **Tracked training metrics (BLEU, perplexity, loss)**  

---

## 📂 Dataset  
The dataset consists of **bilingual sentence pairs** for English-Telugu and English-Hindi translations, formatted with MBart-50 language tokens:  

- `[en_XX]` → English  
- `[te_IN]` → Telugu  
- `[hi_IN]` → Hindi  
- `[eos]` → End of Sentence  

### **Data Split:**  
- **Training Data:** 786,432 pairs  
- **Validation Data:** 98,304 pairs  
- **Test Data:** 98,305 pairs  

---

## 🔧 Model Training  
The MBart-50 model was fine-tuned with the following **hyperparameters**:  

- **Batch Size:** 32  
- **Learning Rate:** 3e-5  
- **Max Sequence Length:** 200  
- **Epochs:** 3  

### **Training Metrics:**  
During training, the model demonstrated **consistent loss reduction**, ensuring stable convergence:  

- **Training Loss:** 1.3037 → 0.1868  
- **Validation BLEU Score:** 26.68% (Epoch 2) → 26.60% (Epoch 3)  
- **Validation Perplexity:** 1.3058 → 1.3047  

---

## 📊 Key Insights  
- 📌 The model **learned robust multilingual translation patterns**  
- 📌 **BLEU scores confirmed** high-quality translation generation  
- 📌 **Low perplexity values** indicated fluent and natural translations  
- 📌 **Vocabulary analysis** improved tokenization efficiency  

## 🚀 Inference  
To generate translations, provide an English sentence with the target language token:  


from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained("path_to_your_trained_model")

sentence = "This is a multilingual translation test."
inputs = tokenizer(f"[en_XX] {sentence} [eos]", return_tensors="pt")

# Generate translation
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang))
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Translated Sentence:", translation)

📌 Next Steps
	•	🔍 Experimenting with low-resource language translations
	•	🔍 Optimizing for faster inference on low-end hardware
	•	🔍 Extending the dataset for better generalization

 💡 Why Understanding LLMs Matters

In the current AI race, tracking every new model and research paper is overwhelming. However, understanding the fundamentals of LLMs helps us adapt to evolving AI trends. This project was an opportunity to go beyond fine-tuning and explore the inner workings of multilingual transformers.

If you’re interested in NLP, LLMs, or machine translation, let’s connect and collaborate! 🚀

📜 References
	•	MBart-50 Paper: https://arxiv.org/abs/2001.08210
	•	Hugging Face MBart-50: https://huggingface.co/facebook/mbart-large-50
