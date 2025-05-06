import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import openai
from openai import OpenAI
import io

# Streamlit UI Setup
st.set_page_config(page_title="Financial Risk Analyzer MVP", layout="wide")
st.title("📊 FlipRisk")

# API key input (optional: you can hardcode below if you prefer)
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key above to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Extract text from image
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    return pytesseract.image_to_string(image)

# Ask GPT-4 for relevance and analysis
def analyze_document(text):
    prompt = f"""
You are a world-class financial risk analysis expert. Given the following document, do the following:

1. **Implicit Risk Detection**:
   - **Sentiment Analysis**: Determine the overall sentiment of the document (e.g., optimistic, cautious, neutral), and identify any hidden fears or aspirations.
   - **Behavioral Indicators**: Identify any implicit clues indicating underlying stress, such as attempts to downplay issues or overly optimistic projections.
   - **Unspoken Risks**: Extract risks or concerns that are not explicitly mentioned but can be inferred from the tone or context (e.g., liquidity problems, exposure to market downturns).

2. **Document Summary Report**:
   - Provide a detailed **executive summary** of the document, focusing on key findings related to risks, financial health, and performance.
   - **Highlight Critical Information**: Summarize both explicit and implicit key points like financial ratios, risks, market outlook, and strategic plans.
   - **Hidden Signals**: Identify statements or terms that suggest risks or changes not fully disclosed, such as mentioning "liquidity" without providing full details, or talking about "expansion" without detailing funding sources.

3. **Detailed Financial Analysis**:
   - Extract and present financial ratios and metrics (e.g., liquidity, profitability, solvency) and **analyze trends** over time, even if implied. For example, if revenues have been "steady" but not growing, this could indicate stagnation.
   - **Cash Flow Indicators**: Analyze whether the reported figures point to potential **cash flow problems** despite no explicit mention.
   
4. **Risk Level Breakdown**:
   - **Risk Impact**: Estimate the severity of risks based on the language and numbers provided. For example, "We are facing challenges in the credit market" implicitly suggests potential solvency or liquidity issues.
   - **Likelihood of Risks**: Based on the document's context, estimate the likelihood of different risks (market, operational, liquidity).
   - **Risk Mitigation Signals**: Identify implicit statements of action or **lack of action** on risk mitigation. Are they addressing risks, or are they downplaying them?

5. **Comparative Analysis**:
   - **Industry Comparison**: Even if the document doesn’t explicitly mention competitors or benchmarks, analyze the content against **general industry norms** (e.g., typical debt ratios, profitability margins, etc.). Is the company underperforming, overperforming, or in line with expectations?

6. **Hidden Red Flags**:
   - **Key Red Flags**: Identify terms that suggest hidden dangers, like repeated references to "uncertain," "potential," or "future challenges." These can signal unresolved issues or undetermined risks.
   - **Avoidance Language**: Detect instances of language that avoids directly addressing key concerns, such as vague references to "risk management" or "future plans."

7. **Opportunities & Actionable Insights**:
   - **Strategic Opportunities**: Identify business or market opportunities based on the document (e.g., mentions of expansion plans, new market entry, or investment).
   - **Operational Improvements**: Suggest possible **operational improvements** that can increase efficiency or reduce costs based on the document’s content.

8. **Overall Risk Rating**:
   - Provide a **risk score** for the document’s entity on a scale from 1 to 10, considering both explicit financial data and implicit signals. Give a detailed justification for this score.

Document Text:
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial risk analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# Upload section
uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_type = uploaded_file.type
    st.info(f"Processing: {uploaded_file.name}")

    with st.spinner("Extracting text..."):
        if "pdf" in file_type:
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif any(ext in file_type for ext in ["png", "jpeg", "jpg"]):
            extracted_text = extract_text_from_image(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    st.subheader("Extracted Text Preview")
    st.text_area("Text", extracted_text[:3000], height=200)

    with st.spinner("Analyzing with GPT-4..."):
        result = analyze_document(extracted_text)

    st.subheader("🧠 GPT-4 Risk Analysis")
    st.markdown(result)
else:
    st.write("👈 Upload a document to get started.")
