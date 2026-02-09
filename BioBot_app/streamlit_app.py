import streamlit as st
from recommender import Recommender

import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="BioBot: asistente inteligente para biopesticidas (Versión 1.1)",
    layout="wide"
)

# Título con un emoji grande y de color, y texto normal
st.markdown("""
    <h1 style='text-align: center;'>
        <span style='font-size: 80px; color: orange;'>🤖</span> 
            BioBot: asistente inteligente para biopesticidas (Versión 1.1)
    </h1>
""", unsafe_allow_html=True)

#st.set_page_config(page_title="BioBot: asistente inteligente para biopesticidas (Versión 1)", layout="wide")
#st.title("🤖")
#st.title("\nBioBot: asistente inteligente para biopesticidas.\n(Versión 1)")

@st.cache_resource
def load_rec():
    return Recommender()

rec = load_rec()

st.write("Describe tu problema/plaga/cultivo (ES o EN). Ej.: *mildiu velloso en vid, temporada húmeda*")
q = st.text_area("Descripción del problema:", height=120)
k = st.slider("Número de recomendaciones", 1, 10, 3)

if st.button("Buscar"):
    if not q.strip():
        st.warning("Por favor, escribe una descripción.")
    else:
        hits = rec.search(q, k=k)
        for hit in hits:
            st.markdown(rec.format_spanish(hit))
            st.markdown("---")
