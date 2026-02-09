# BioBot: asistente inteligente para biopesticidas

Chatbot local y API para recomendar biopesticidas a partir de una descripción (ES/EN).
Usa embeddings multilingües (sin APIs de pago).

## Pasos rápidos en Windows (cmd)

1. Instala Python 3.10+ y Git.
2. Abre `cmd` y ve a tu carpeta de usuario: `cd %USERPROFILE%`
3. Descomprime este proyecto en `BioBot_app` (o copia la carpeta). Entra: `cd BioBot_app`
4. Crea venv: `python -m venv .venv` y actívalo: `.\.venv\Scripts\activate`
5. Instala dependencias: `pip install -r requirements.txt`
6. Coloca tu dataset en `data/full_ds.csv`
7. Construye índice: `python build_index.py`
8. API: `uvicorn app:app --reload --port 8000`
9. UI local: `streamlit run streamlit_app.py`
10. (Opcional) WhatsApp/Twilio: expón con `ngrok http 8000` y usa el endpoint `/whatsapp` como webhook entrante.
