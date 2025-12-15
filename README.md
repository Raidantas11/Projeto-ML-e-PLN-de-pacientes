Projeto que usa Machine Learning e PLN para analisar comentários de pacientes da área de saúde. 

- Classificar automaticamente a a partir do comentário do paciente  
- Associar a categoria à especialidade médica correspondente
- Realizar análise de sentimentos para identificar comentários negativos  
- Disponibilizar a análise por meio de uma aplicação web com Streamlit

Usado
- Python
- PLN (NLTK, TF-IDF)
- Machine Learning (LinearSVC – SVM)
- Análise de Sentimentos (VADER)
- Streamlit (deploy)
- Joblib

- execução:
python -m streamlit run app.py

NT: Use comentários em inglês. Modelo foi treinado usando base de dados em inglês e mantive sem tradução por enquanto para poder entregar.
