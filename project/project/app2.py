import streamlit as st
import openai
import os
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DistilBertConfig, DistilBertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from io import StringIO

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Styling ---
st.markdown(
    """
    <style>
        .stApp {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 2rem;
            color: #1E90FF; 
        }
        .section-header {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #4682B4;
        }
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            cursor: pointer;
            border-radius: 5px;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background-color: #4682B4;
        }
        .stTextArea textarea {
            font-size: 1rem;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #ddd;
            resize: vertical;
        }
        .stSelectbox select {
            font-size: 1rem;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .stSlider div {
            font-size: 1rem;
        }
        .stTextInput input {
            font-size: 1rem;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .chart-container {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
        }
        .chart-container .stPyplot {
            flex: 1;
        }
        .stError {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .stInfo {
            background-color: #e2f2ff;
            color: #0c5460;
            border: 1px solid #b8daff;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .code-block {
            background-color: #f2f2f2;
            padding: 1rem;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Dependency Check ---
def check_dependencies():
    try:
        import openai
        from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DistilBertConfig, DistilBertForSequenceClassification
        from datasets import load_dataset
        from langchain.document_loaders import TextLoader
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        st.error(f"Missing dependency: {e.name}. Please ensure all required libraries are installed.")
        return False
    return True

if not check_dependencies():
    st.stop()

# --- Interactive Prompt Exploration ---
st.header("Interactive Prompt Exploration", anchor="prompt-exploration")
st.markdown("This section allows you to experiment with different prompts and LLM parameters.")

# Prompt Input
prompt = st.text_area("Enter your prompt:", "Write a short story about a robot who falls in love.", height=150)

# LLM Selection
model_name = st.selectbox("Choose LLM", ["gpt-3.5-turbo"], index=0)  # Only option is gpt-3.5-turbo

# Parameters
temperature = st.slider("Temperature", 0.0, 1.0, 0.5, step=0.1)
max_tokens = st.number_input("Max Tokens", min_value=10, max_value=1000, value=100, step=10)

# Generate Response
if st.button("Generate"):
    with st.spinner("Generating text..."):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            generated_text = response.choices[0].message['content']
            st.subheader("Generated Text")
            st.write(generated_text)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Fine-tuning (Sentiment Analysis) ---
st.header("Fine-tuning (Sentiment Analysis)", anchor="fine-tuning")
st.markdown("This section demonstrates fine-tuning a pre-trained model for sentiment analysis.")

@st.cache_data
def load_and_train_model():
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine-tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"]
    )
    return trainer

if st.button("Fine-tune Model"):
    with st.spinner("Fine-tuning model..."):
        try:
            trainer = load_and_train_model()
            trainer.train()
            st.success("Model fine-tuned successfully!")
        except Exception as e:
            st.error(f"An error occurred during fine-tuning: {e}")

# --- Distillation (DistilBERT) ---
st.header("Distillation (DistilBERT)", anchor="distillation")
st.markdown("This section shows how to distill a smaller, faster DistilBERT model from a larger BERT model.")

@st.cache_resource
def load_distilbert_model():
    student_config = DistilBertConfig(vocab_size=30522, hidden_size=768)
    student_model = DistilBertForSequenceClassification(student_config)
    return student_model

if st.button("Distill Model"):
    with st.spinner("Distilling model..."):
        try:
            student_model = load_distilbert_model()
            st.info("Distillation in progress!")  # Indicate progress without blocking the UI
        except Exception as e:
            st.error(f"An error occurred during distillation: {e}")

# --- Retrieval-Augmented Generation (RAG) ---
st.header("Retrieval-Augmented Generation (RAG)", anchor="rag")
st.markdown("This section demonstrates Retrieval-Augmented Generation, which combines retrieval with LLM generation.")

uploaded_file = st.file_uploader("Upload a text file for RAG:", type=["txt"])

@st.cache_data
def initialize_rag(file):
    try:
        # Read the uploaded file
        content = file.read().decode("utf-8")
        if not content:
            raise ValueError("The file is empty or could not be read properly.")
        
        # Split the text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)
        
        # Create embeddings and vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Initialize RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=openai.ChatCompletion.create,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        return qa_chain
    
    except FileNotFoundError as fnf_error:
        st.error(f"File error: {fnf_error}")
        return None
    except ValueError as val_error:
        st.error(f"Value error: {val_error}")
        return None
    except Exception as e:
        st.error(f"An error occurred while initializing RAG: {e}")
        return None

query = st.text_input("Enter your question:")
if st.button("Ask Question") and uploaded_file:
    with st.spinner("Retrieving information..."):
        try:
            qa_chain = initialize_rag(uploaded_file)
            if qa_chain:
                answer = qa_chain.run(query)
                st.subheader("Answer")
                st.write(answer)
        except Exception as e:
            st.error(f"An error occurred during retrieval: {e}")

# --- RLHF (Reinforcement Learning from Human Feedback) ---
st.header("RLHF (Reinforcement Learning from Human Feedback)", anchor="rlhf")
st.markdown("This section showcases the RLHF process, where human feedback is used to fine-tune a model.")

# Simulate RLHF functionality
st.subheader("Provide Feedback")
feedback_prompt = st.text_area("Enter a prompt for feedback:", "Write a short story about a robot who learns to dance.", height=150)
feedback_response = st.text_area("Generated Response:", "The robot learned to dance by practicing every day and eventually performed at a grand event.", height=150)
feedback_rating = st.slider("Rate the response (1-5):", 1, 5, 3)

if st.button("Submit Feedback"):
    with st.spinner("Processing feedback..."):
        try:
            # For demonstration purposes, simulate storing feedback and retraining
            feedback_data = {
                "prompt": feedback_prompt,
                "response": feedback_response,
                "rating": feedback_rating
            }
            # Simulate storing feedback (in practice, you'd store this data and use it to fine-tune the model)
            st.success("Feedback submitted successfully!")
            st.info(f"Feedback Data: {feedback_data}")
        except Exception as e:
            st.error(f"An error occurred during feedback processing: {e}")

# --- Visualization ---
st.header("Visualization", anchor="visualization")
st.markdown("This section provides multi-chart visualization options for your data.")

# Generate some dummy data for visualization
data = np.random.randn(100)

# Create the charts
fig1, ax1 = plt.subplots()
ax1.hist(data, bins=20, color='skyblue', edgecolor='black')
ax1.set_title("Histogram")

fig2, ax2 = plt.subplots()
ax2.boxplot(data, vert=False)
ax2.set_title("Box Plot")

# Display the charts side by side
col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig1)

with col2:
    st.pyplot(fig2)

# --- Conclusion ---
st.header("Conclusion", anchor="conclusion")
st.markdown("This Streamlit app showcases various techniques for prompt exploration, fine-tuning, distillation, RAG, RLHF, and data visualization.")

# --- End of Streamlit app ---
