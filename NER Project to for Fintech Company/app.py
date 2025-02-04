import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load your NER model and tokenizer
@st.cache_resource
def load_ner_model():
    model_name = "your-ner-model-name"  # Replace with your model path or HuggingFace model name
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    return ner_pipeline

ner_pipeline = load_ner_model()

# Streamlit UI
st.title("Named Entity Recognition (NER) Demo")
st.subheader("Extract entities from text like Loan Types, Names, Dates, etc.")

user_input = st.text_area("Enter the text you want to analyze", height=200)

if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Processing..."):
            results = ner_pipeline(user_input)
        
        st.success("Entities Extracted Successfully!")
        st.write("### Extracted Entities:")
        for entity in results:
            st.markdown(
                f"**Entity:** `{entity['word']}`\n"
                f"- **Type:** `{entity['entity_group']}`\n"
                f"- **Score:** `{entity['score']:.2f}`"
            )
    else:
        st.warning("Please enter some text to analyze!")

st.sidebar.markdown("### About the App")
st.sidebar.info(
    "This app demonstrates a Named Entity Recognition (NER) project built with HuggingFace. "
    "It extracts entities from user-provided text using a fine-tuned Transformer model."
)
