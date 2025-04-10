import streamlit as st
import torch 
import numpy as np
from main import calculate_perplexity, load_classifier
def initialize_session_state():
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
def run_app():
    initialize_session_state()
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f4f8;
            color: #333;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .sidebar h1 {
            color: #0072B1;
        }
        .sidebar h2 {
            color: #0056A1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.sidebar:
        st.title("Teplur an 🔍AI Text Detector")
        st.subheader("Overview")
        st.write("""
        This application assists you in:
        - Detecting if the text is AI-generated or human-written.
        - Understanding the perplexity score of the text.
        """)
    st.title("📄 AI Text Detection")
    st.subheader("Analyze Your Text for AI Generation")
    user_input = st.text_area(
        "Input Text",
        placeholder="Paste your text here...",
        help="Provide the text you want to analyze"
    )
    if st.button("Analyze Text", disabled=st.session_state.is_processing):
        if not user_input:
            st.warning("Please enter some text to analyze.")
            return
        st.session_state.is_processing = True
        try:
            with st.spinner("📊 Analyzing your text..."):
                ppl = calculate_perplexity(user_input)
                if np.isnan(ppl) or np.isinf(ppl):
                    st.error("Invalid perplexity score. Try with different text.")
                    return
                clf = load_classifier()
                log_ppl = np.log1p(ppl)
                prediction = clf.predict([[log_ppl]])[0]
                proba = clf.predict_proba([[log_ppl]])[0]
                threshold = 0.5
                st.success("✨ Analysis Completed!")
                st.subheader("Results:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Perplexity Score", f"{ppl:.2f}")
                with col2:
                    result = "AI-Generated" if proba[1] > threshold else "Human-Written"
                    st.metric("Prediction", result)
                st.progress(proba[1])
                st.caption(f"AI generation confidence: {proba[1]*100:.1f}%")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.session_state.is_processing = False
if __name__ == "__main__":
    run_app()