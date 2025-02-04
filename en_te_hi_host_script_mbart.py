import streamlit as st
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load model & tokenizer
save_directory = '/Users/venu/Documents/Productivity/Pytorch Tutorials/Attention is All You Need Paper Replication/mbart50_translation_en_te_hi_model_lit'
model = MBartForConditionalGeneration.from_pretrained(save_directory)
tokenizer = MBart50TokenizerFast.from_pretrained(save_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Translation function
def translate_text(input_text, target_lang):
    tokenizer.src_lang = "en_XX"
    model_inputs = tokenizer(f"en_XX {input_text}", return_tensors="pt").to(device)
    
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    
    outputs = model.generate(
        **model_inputs, forced_bos_token_id=forced_bos_token_id, max_length=200
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("Multilingual Translation Model")
st.write("Translate English sentences into Telugu or Hindi using MBart-50.")

input_text = st.text_area("Enter an English sentence:")
target_lang = st.selectbox("Select Target Language", ["te_IN", "hi_IN"])

if st.button("Translate"):
    if input_text:
        translation = translate_text(input_text, target_lang)
        st.write("### Translation:")
        st.success(translation)
    else:
        st.warning("Please enter a sentence to translate.")
