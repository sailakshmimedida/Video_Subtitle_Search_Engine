import streamlit as st
import soundfile as sf
from transformers import pipeline
import sentence_transformers
import chromadb

# Load the Hugging Face speech-to-text model
speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Load the subtitle search model (Assuming embeddings are stored in ChromaDB)
chroma_client = chromadb.PersistentClient(path="./dataset")  
collection = chroma_client.get_collection("subtitles")

# Function to convert audio to text
def convert_audio_to_text(audio_file):
    audio_data, samplerate = sf.read(audio_file)
    result = speech_to_text(audio_data)
    return result["text"]

# Function to search for relevant subtitles
def search_subtitles(query):
    results = collection.query(query_texts=[query], n_results=5)
    return results["documents"][0] if results["documents"] else ["No matching subtitles found."]

# Streamlit UI
st.title("🎬 Video Subtitle Search Engine")

st.write("Upload an audio file to search for matching subtitles.")

# File uploader for audio
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("Converting speech to text... 🎙️"):
        query_text = convert_audio_to_text(uploaded_file)
        st.write("**Detected Text:**", query_text)

    with st.spinner("Searching subtitles... 🔎"):
        subtitles = search_subtitles(query_text)
        st.write("**Matching Subtitles:**")
        for subtitle in subtitles:
            st.write("- ", subtitle)
