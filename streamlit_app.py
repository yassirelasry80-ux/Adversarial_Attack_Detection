import streamlit as st
import os
from PIL import Image
from inference import InferenceSystem

# Configuration de la page
st.set_page_config(
    page_title="Détection d'Attaques Adversariales",
    page_icon="🛡️",
    layout="centered"
)

# Titre et Header
st.title("🛡️ Détection d'Attaques Adversariales")
st.markdown("### Analyse de Radiographies Thoraciques")
st.markdown("---")

# Initialisation du système d'inférence (mis en cache pour la performance)
@st.cache_resource
def load_inference_system():
    return InferenceSystem()

try:
    inference = load_inference_system()
    st.success("Système d'inférence chargé avec succès!")
except Exception as e:
    st.error(f"Erreur lors du chargement du système: {e}")
    st.stop()

# Upload de l'image
uploaded_file = st.file_uploader("Choisissez une image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)
    
    # Bouton d'analyse
    if st.button("Lancer l'Analyse 🔍", type="primary"):
        with st.spinner("Analyse en cours..."):
            # Sauvegarde temporaire pour le système d'inférence existant
            temp_path = "temp_upload.jpg"
            image.save(temp_path)
            
            # Prédiction
            try:
                result = inference.predict_single_image(temp_path)
                
                # Nettoyage
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                st.markdown("---")
                st.header("Résultats")
                
                # Création de deux colonnes pour les résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Classification")
                    prediction = result['prediction']
                    conf = result['confidence'] * 100
                    
                    if prediction == "PNEUMONIA":
                        st.error(f"🚨 **{prediction}**")
                    else:
                        st.success(f"✅ **{prediction}**")
                    
                    st.metric("Confiance", f"{conf:.2f}%")
                    
                    # Détail des probabilités
                    st.markdown("**Probabilités détaillées:**")
                    for cls, prob in result['all_probabilities'].items():
                        st.progress(int(prob * 100), text=f"{cls}: {prob*100:.2f}%")

                with col2:
                    st.subheader("Sécurité")
                    is_adv = result['is_adversarial']
                    adv_conf = result['adversarial_confidence'] * 100
                    
                    if is_adv:
                        st.error("⚠️ **ATTAQUE DÉTECTÉE**")
                        st.markdown("Cette image semble avoir été manipulée.")
                    else:
                        st.success("🛡️ **Image Saine**")
                        st.markdown("Aucune modification malveillante détectée.")
                        
                    st.metric("Score de détection", f"{adv_conf:.2f}%")

            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# Footer
st.markdown("---")
st.caption("Projet de Fouille de Données - Attaques Adversariales & Apprentissage Fédéré")
