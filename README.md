
# 🧠 Teplur an – AI Text Detector

A web-based application to distinguish between AI-generated and human-written text using GPT-2 perplexity scores and logistic regression. Built with **Streamlit**, **Hugging Face Transformers**, and **scikit-learn**.

---

## 🚀 Features

- Detects if a text is **AI-generated** or **human-written**
- Computes and displays **perplexity scores**
- Intuitive **web interface** using Streamlit
- Simple **logistic regression classifier** trained on custom datasets

---

## 🛠️ Tech Stack

- Python 3
- [Streamlit](https://streamlit.io/)
- [Transformers (GPT-2)](https://huggingface.co/transformers/)
- scikit-learn
- PyTorch
- pandas, numpy

---

## 📁 Project Structure

```
├── app.py                      # Streamlit web interface
├── main.py                     # Core logic: training & prediction
├── DATASET_AD_AI_Updated.csv   # AI-generated text dataset
├── DATASET-AD - HUMAN.csv      # Human-written text dataset
├── ai_detector_clf.pkl         # (Generated) Trained model
└── README.md                   # Project documentation
```

---

## ⚙️ Setup & Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ai-text-detector.git
   cd ai-text-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python main.py
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📊 How It Works

- GPT-2 calculates **perplexity** of input text — a measure of how “surprising” the text is.
- Text with **lower perplexity** is often AI-generated.
- A logistic regression model uses perplexity (log-transformed) to predict origin.

---

## 📈 Example Output

- **Perplexity Score**: 25.84
- **Prediction**: AI-Generated
- **Confidence**: 92.3%
- <p align="center">
  <img src="https://github.com/Kreezon/Teplur/blob/main/Evaluation%20matrix.jpg" width="500"/>

   
  <img src="https://github.com/Kreezon/Teplur/blob/main/Sample%20test.jpg" width="500"/>
</p>


---

## 📌 Notes

- The classifier is trained using the column `IEEE` from both CSV files.
- The model (`ai_detector_clf.pkl`) is saved after training for fast loading in the Streamlit app.
- Ensure GPU is available for faster processing of GPT-2, though CPU fallback is implemented.

---

## 📄 License

MIT License
