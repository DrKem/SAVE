import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd

# Set page title and layout
st.set_page_config(page_title="Endangered Species Analysis", layout="wide")

# Custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://via.placeholder.com/600x400");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-color: rgba(255, 255, 255, 0.9);
        background-blend-mode: overlay;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("Smart Analysis for Vulnerable and Endangered Species")

# File upload section
uploaded_file = st.file_uploader(
    "Upload an image to analyze endangered animals",
    type=["jpg", "jpeg", "png"],
    help="Upload an image of an endangered species for analysis.",
)

# Display uploaded image
if uploaded_file is not None:
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Simulate upload progress
    with st.spinner("Uploading and analyzing..."):
        # Simulate a delay for upload progress
        import time

        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)  # Simulate upload delay
            progress_bar.progress(percent_complete + 1)

        # Send the image to the backend for analysis
        backend_url = "http://localhost:8000/analyze-species"
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("Analysis complete!")
        else:
            st.error("Failed to analyze the image. Please try again.")
            st.stop()

    # Display analysis results
    st.subheader("Analysis Results")

    # Basic info
    if "identification" in result:
        species_name = result["identification"].get("species", "Unknown")
        confidence = result["identification"].get("confidence", "N/A")
        st.markdown(f"**Species:** {species_name}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    # Species data
    if "species_data" in result:
        species_data = result["species_data"]
        st.subheader("Species Information")

        # Basic info
        if "basic_info" in species_data:
            basic_info = species_data["basic_info"]
            st.markdown(f"**Class:** {basic_info.get('class', 'Unknown')}")
            st.markdown(f"**Weight:** {basic_info.get('weight', 'Unknown')}")
            st.markdown(f"**Height:** {basic_info.get('height', 'Unknown')}")
            st.markdown(f"**Habitats:** {basic_info.get('habitats', 'Unknown')}")
            st.markdown(f"**Status:** {basic_info.get('status', 'Unknown')}")
            st.markdown(f"**Countries:** {', '.join(basic_info.get('countries', []))}")

        # Population trend
        if "population_trend" in species_data:
            population_trend = species_data["population_trend"]
            st.subheader("Population Trend")
            st.markdown(f"**Current Population:** {population_trend.get('current_population', 'Unknown')}")
            st.markdown(f"**Decline Rate:** {population_trend.get('decline_rate', 'Unknown')}%")

            # Plot population trend
            if "years" in population_trend and "population" in population_trend:
                df = pd.DataFrame({
                    "Year": population_trend["years"],
                    "Population": population_trend["population"],
                })
                st.line_chart(df.set_index("Year"))

        # Threats
        if "threats" in species_data:
            st.subheader("Threats")
            threats = species_data["threats"]
            if threats:
                st.write("Major threats to this species:")
                for threat in threats:
                    st.markdown(f"- {threat}")
            else:
                st.write("No threat data available.")

        # Remediation measures
        if "remediation_measures" in species_data:
            remediation = species_data["remediation_measures"]
            st.subheader("Conservation Measures")
            if remediation.get("measures"):
                st.write("Conservation measures in place:")
                for measure in remediation["measures"]:
                    st.markdown(f"- {measure}")
                if remediation.get("effect"):
                    st.markdown(f"**Effectiveness:** {remediation['effect']}")
            else:
                st.write("No conservation measures data available.")