import streamlit as st
import urllib.parse

st.set_page_config(page_title="Symptom Checker", layout="centered")
st.title("🩺 Smart Symptom Checker")
st.write("Describe your symptoms to get a possible diagnosis, suggested remedies, and nearby help.")

# Predict disease from symptoms (rule-based)
def predict_disease_from_symptoms(symptoms_text):
    symptoms_text = symptoms_text.lower()

    if "chest pain" in symptoms_text or "shortness of breath" in symptoms_text:
        return "Heart Disease"
    elif "frequent urination" in symptoms_text or "thirst" in symptoms_text or "blurred vision" in symptoms_text:
        return "Diabetes"
    elif "lump" in symptoms_text or "breast pain" in symptoms_text:
        return "Breast Cancer"
    else:
        return "Unknown"

# Show nearby services using Google Maps
def show_nearby_search(keyword="pharmacy"):
    location_search_url = f"https://www.google.com/maps/search/{urllib.parse.quote(keyword)}+near+me"
    st.markdown(f"[🔍 Search for nearby {keyword.title()}](%s)" % location_search_url)

# Main symptom input
st.subheader("📝 Describe Your Symptoms")
symptom_input = st.text_area("Enter your symptoms (e.g. frequent urination, fatigue, blurred vision)")

if st.button("Analyze Symptoms"):
    predicted_disease = predict_disease_from_symptoms(symptom_input)
    st.info(f"🧠 Possible Condition: **{predicted_disease}**")

    if predicted_disease == "Diabetes":
        st.write("💊 **Suggested Medicines:** Metformin, Glimepiride")
        st.write("👩‍👧‍👦 **Mom’s Remedy:** Fenugreek water, cinnamon tea")
    elif predicted_disease == "Heart Disease":
        st.write("💊 **Suggested Medicines:** Aspirin, Beta-blockers")
        st.write("👩‍👧‍👦 **Mom’s Remedy:** Garlic in warm water, turmeric milk")
    elif predicted_disease == "Breast Cancer":
        st.write("📌 **Suggested Action:** Please consult an oncologist.")
        st.write("👩‍👧‍👦 **Mom’s Advice:** Warm compress for pain — but don’t delay professional care.")
    else:
        st.warning("❓ Couldn't confidently match your symptoms to a known condition.")

    st.subheader("📍 Find Nearby Help")
    show_nearby_search("medical shop")
    show_nearby_search("path lab")
    show_nearby_search("doctor clinic")
