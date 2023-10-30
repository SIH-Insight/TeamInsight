import streamlit as st
import time
from PIL import Image
import io
import math
from src.crime import crime
from src.crowd import crowd_management as cm
from src.garbage import garbage 
# Define a custom CSS class for bold and italic text
st.set_page_config(page_title='Smart CCTV', page_icon=':camera:')
header_style = """
    <style>
        .bold-header {
            font-weight: bold;
            font-size: 2.75em;
        }
        .italic-text {
            font-style: italic;
            font-size: 1.5em;
        }
        .sidebar-radio label {
            font-size: 1.5em !important;
        }
        italic-new{
            font-style: italic;
            font-size: 3.0em;
        }
        .bold-2 {
            font-weight: bold;
            font-size: 2.0em;
        }
        .bold-1 {
            font-weight: bold;
            font-size: 1.0em;
        }
    </style>
"""

# Add the custom CSS to the page
st.markdown(header_style, unsafe_allow_html=True)   
# Display the header with the custom style using Markdown
st.markdown("<p class='bold-header'>Smart CCTV Integration Using AI/ML</p>", unsafe_allow_html=True)

# Add a newline
st.text(" ")


# List of actions in the sidebar
st.sidebar.markdown("<p class='bold-2'>What do you need?</p>",unsafe_allow_html=True)
selected_action = st.sidebar.radio("",["Work Monitoring", "Crime Detection", "Crowd Management","Garbage Detection"], key="action")

# Apply custom CSS for the sidebar radio options
sidebar_style = """
<style>
div[data-baseweb="radio"] label {
    font-size: 20px !important;
}
</style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)

# Handle actions based on the selected option
if selected_action == "Work Monitoring":


    st.sidebar.markdown("<p class='italic-text'>Monitor work environments for productivity and safety.</p>", unsafe_allow_html=True)
    
    # Display video upload page
    st.markdown("## Monitor work 24/7:")
    
    # Add a file uploader widget
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

elif selected_action == "Crime Detection":
    st.sidebar.markdown("<p class='italic-text'>Proactively detect criminal activity and enhance public safety.</p>", unsafe_allow_html=True)
    st.markdown("## Detect crimes:")
    
    # Add a file uploader widget
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with st.spinner('Loading....'):
            video_bytes = uploaded_file.read()
            file_content=r'C:\Users\Girish\.vscode\programs\new_sih\src\crime'
            predicted_video = crime.predict_frames(video_bytes, 16)
            a,b=crime.predict_video(video_bytes, 16)
            st.write("Predicted: ",a)
            st.write("Confidence: ",b)
            time.sleep(5)
        st.success('Done!')
        

elif selected_action == "Crowd Management":
    st.sidebar.markdown("<p class='italic-text'>Efficiently manage crowds at public events and crowded areas.</p>", unsafe_allow_html=True)
    st.markdown("## Manage crowd:")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with st.spinner('Loading....'):
            video_bytes = uploaded_file.read()
            df=cm.crowd_management(video_bytes)
            df['human_count'] = df['human_count'].apply(lambda x: math.ceil(x))
            # Create a Streamlit app
            st.title('Human Count over Time')
            st.line_chart(df.set_index('timestamp'))
            
            time.sleep(5)
        st.success('Done!')

elif selected_action == "Garbage Detection":
    st.sidebar.markdown("<p class='italic-text'>Detect and manage garbage or litter in public areas.</p>", unsafe_allow_html=True)
    st.markdown("## Detect Garbage:")
    
    # Add a file uploader widget
    uploaded_file = st.file_uploader("Upload a video file for garbage detection", type=["jpeg", "png", "jpg"])
    
    if uploaded_file is not None:
        with st.spinner('Loading....'):
            image = Image.open(io.BytesIO(uploaded_file.read()))
            prediction=garbage.classify_garbage(image)
            st.write(f"Prediction : {prediction}")
            time.sleep(5)
        st.success('Done!') 
        



st.markdown("<h3 style='text-align: center;'>Empowering Smarter Surveillance</h3>", unsafe_allow_html=True)

st.text(" ")
st.text(" ")
st.text(" ")
# Real-world insights section with a fancy underline
st.markdown("### 🌐 Explore Real-World Insights 🌐 :")
st.markdown("<p class='italic-text'>Our platform seamlessly combines the power of Artificial Intelligence and Machine Learning to transform your surveillance footage into real-time intelligence. 🌍</p>", unsafe_allow_html=True)

st.text(" ")

# Key features with an emoji
st.markdown("### 🚀 Key Features 🚀 :")
st.markdown("<p class='italic-text'>Monitor work environments with unparalleled efficiency, proactively detect criminal activity, and manage crowds effortlessly. AI is here to redefine security and productivity. 🔍</p>", unsafe_allow_html=True)

st.text(" ")

# Under the hood section with an emoji
st.markdown("### 🕵️‍♀️ Under the Hood 🕵️‍♂️ :")
st.markdown("<p class='italic-text'>Powered by cutting-edge technologies, we're at the forefront of the AI and ML revolution. From object recognition to anomaly detection, we've got it all. 💡</p>", unsafe_allow_html=True)

st.text(" ")

# User-friendly section with an emoji
st.markdown("### 👩‍💻 User-Friendly 👨‍💻 :")
st.markdown("<p class='italic-text'>No tech wizardry required! With a few clicks, you can transform your surveillance data into actionable insights. 🤖</p>", unsafe_allow_html=True)

st.text(" ")

# Footer with another call to action
st.subheader("Ready to experience the future of surveillance? Dive in and explore!! 🚀.Get Started Now! 👇")

# Add a newline in the sidebar
st.sidebar.text(" ")



