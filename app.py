import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

st.set_page_config(page_title="Analise Medica de Comentarios", layout="centered")

tfidf = joblib.load("tfidf_vectorizer.pkl")
modelo = joblib.load("linear_svc_model.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def traduzir_para_ingles(texto):
    try:
        return GoogleTranslator(source="pt", target="en").translate(texto)
    except:
        return texto

def preprocess_text(texto):
    texto = texto.lower()
    texto = re.sub(f"[{string.punctuation}]", "", texto)
    tokens = texto.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

analisador = SentimentIntensityAnalyzer()

def prever_sentimento(texto_pt):
    texto_en = traduzir_para_ingles(texto_pt)
    score = analisador.polarity_scores(texto_en)["compound"]

    if score >= 0.05:
        return "Positivo"
    elif score <= -0.05:
        return "Negativo"
    else:
        return "Neutro"

categoria_para_especialidade = {
    "Knee pain": "Ortopedia",
    "Joint pain": "Ortopedia",
    "Eye Infection": "Oftalmologia",
    "Skin issue": "Dermatologia",
    "Stomach ache": "Gastroenterologia",
    "Heart hurts": "Cardiologia",
    "Hard to breath": "Pneumologia",
    "Open wound": "Cirurgia Geral"
}

def prever_categoria_medica(texto_pt):
    texto_en = traduzir_para_ingles(texto_pt)
    texto_proc = preprocess_text(texto_en)
    vetor = tfidf.transform([texto_proc])
    return modelo.predict(vetor)[0]

st.title("Analise Inteligente de Comentarios Medicos")

st.write(
    "Digite o comentario do paciente para classificar a categoria medica, "
    "especialidade e sentimento."
)

comentario = st.text_area("Comentario do paciente", height=150)

if st.button("Analisar comentario"):
    if comentario.strip() == "":
        st.warning("Digite um comentario para analise.")
    else:
        categoria = prever_categoria_medica(comentario)
        especialidade = categoria_para_especialidade.get(
            categoria, "Clinica Geral"
        )
        sentimento = prever_sentimento(comentario)

        st.subheader("Resultado")
        st.write(f"Categoria medica: {categoria}")
        st.write(f"Especialidade: {especialidade}")
        st.write(f"Sentimento: {sentimento}")
