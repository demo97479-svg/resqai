import os
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


st.set_page_config(
    page_title="AI Emergency Call Intelligence",
    page_icon="🚨",
    layout="wide"
)
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(0, 0, 0, 0.65),
                    rgba(0, 0, 0, 0.65)
                ),
                url("data:image/jpeg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )




set_background("background.jpeg")

st.title("🚨 ResQAI — Real-Time Voice-Based Intelligent Emergency Response Agent ")


load_dotenv(override=True)

api_key = st.text_input(
    "🔑 OpenAI API Key",
    type="password",
    placeholder="Enter your OpenAI API key"
)

if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

client = OpenAI(api_key=api_key)


# MODIFIED: Added Streamlit Session State
# WHY:
# Streamlit reruns the complete script whenever a button is
# clicked. Without session_state, previous transcript and
# extracted information would be lost.


if "conversation_transcript" not in st.session_state:
    st.session_state.conversation_transcript = ""

if "incident_info" not in st.session_state:
    st.session_state.incident_info = ""



# STATE


class EmergencyState(TypedDict):

    transcript: str
    incident_info: str
    missing_question: str
    summary: str



# RECORD AUDIO


def record(duration):

    sample_rate = 16000

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(
        "question.wav",
        sample_rate,
        audio
    )



# TRANSCRIBE AUDIO


def transcribe_audio():

    with open("question.wav", "rb") as audio_file:

        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    return result.text



# EXTRACT INFORMATION


def extract_information(previous_info, transcript):

    prompt = f"""
You are an Emergency Call Intelligence Assistant.

You are maintaining information about ONE ongoing emergency incident.

Previously extracted information:

{previous_info}


Complete conversation transcript:

{transcript}


Extract and UPDATE the following information:

1. Emergency Type
2. Location
3. People Involved
4. Injuries
5. Hazards
6. Severity


Rules:

- Keep useful information from previous information.
- Use the complete conversation to understand the incident.
- Add new information from the latest caller statements.
- If new information is available, update the field.
- Do not delete previous information unless the caller clearly corrects it.
- Do not invent information.
- If information is unknown, write "Not available".
- Severity must be Low, Medium, High, or Critical.


Return ONLY this format:

Emergency Type: <value>
Location: <value>
People Involved: <value>
Injuries: <value>
Hazards: <value>
Severity: <value>
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text.strip()



# CHECK MISSING INFORMATION


def check_missing_information(incident_info):

    if "Emergency Type: Not available" in incident_info:

        return "What type of emergency is happening?"

    elif "Location: Not available" in incident_info:

        return "What is the exact location of the emergency?"

    elif "People Involved: Not available" in incident_info:

        return "How many people are involved?"

    elif "Injuries: Not available" in incident_info:

        return "Is anyone injured?"

    elif "Hazards: Not available" in incident_info:

        return "Are there any current hazards?"

    elif "Severity: Not available" in incident_info:

        return "How serious is the situation?"

    else:

        return ""



# GENERATE SUMMARY


def generate_summary(transcript, incident_info):

    prompt = f"""
You are an Emergency Response Assistant.

Based ONLY on the complete conversation transcript and
extracted information, create a concise emergency incident summary.

Complete Conversation:

{transcript}


Extracted Information:

{incident_info}


Create the summary in this format in english language only:

EMERGENCY SUMMARY

Emergency Type:
Location:
People Involved:
Injuries:
Hazards:
Severity:

Recommended Immediate Action:
<short action based ONLY on the information provided>


Rules:

- Use the complete conversation.
- Do not invent information.
- Do not claim that emergency services have been contacted.
- Do not make medical diagnoses.
- Keep the response concise and professional.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text.strip()



# LANGGRAPH NODES


def update_information_node(state):

    transcript = state["transcript"]

    previous_info = state["incident_info"]

    updated_info = extract_information(
        previous_info,
        transcript
    )

    return {
        "incident_info": updated_info
    }


def check_missing_node(state):

    incident_info = state["incident_info"]

    question = check_missing_information(
        incident_info
    )

    return {
        "missing_question": question
    }


def follow_up_node(state):

    return {
        "missing_question": state["missing_question"]
    }


def summary_node(state):

    final_summary = generate_summary(
        state["transcript"],
        state["incident_info"]
    )

    return {
        "summary": final_summary
    }



# CONDITIONAL ROUTING


def route_after_check(state):

    if state["missing_question"]:

        return "follow_up"

    else:

        return "summary"



# CREATE LANGGRAPH WORKFLOW


workflow = StateGraph(EmergencyState)


# Add nodes

workflow.add_node("update_information",update_information_node)
workflow.add_node("check_missing",check_missing_node)
workflow.add_node("follow_up",follow_up_node)
workflow.add_node("summary",summary_node)
workflow.add_edge(START,"update_information")


# Update information → Check missing

workflow.add_edge(
    "update_information",
    "check_missing"
)


# Conditional routing

workflow.add_conditional_edges(
    "check_missing",
    route_after_check,
    {
        "follow_up": "follow_up",
        "summary": "summary"
    }
)


# Follow-up → End

workflow.add_edge(
    "follow_up",
    END
)


# Summary → End

workflow.add_edge(
    "summary",
    END
)


# Compile graph

app = workflow.compile()



# STREAMLIT UI


st.divider()


duration = st.slider(
    "Recording Duration (seconds)",
    min_value=3,
    max_value=10,
    value=5
)


if st.button(
    "🎤 Record Emergency Call",
    use_container_width=True
):

   
    # RECORD
   

    with st.spinner(
        "🎙️ Recording... Please speak."
    ):

        record(duration)

    st.success(
        "✅ Recording completed!"
    )


   
    # TRANSCRIPTION
   

    with st.spinner(
        "📝 Converting speech to text..."
    ):

        transcript = transcribe_audio()


    st.subheader("📝 Latest Transcript")

    st.write(transcript)


   
    # MODIFIED: APPEND NEW TRANSCRIPT TO PREVIOUS TRANSCRIPT
    # WHY:
    # Every Streamlit button click reruns the application.
    # We therefore store the complete conversation in
    # st.session_state and append each new caller response.
   

    if st.session_state.conversation_transcript:

        st.session_state.conversation_transcript += (
            "\n\nCaller:\n" + transcript
        )

    else:

        st.session_state.conversation_transcript = (
            "Caller:\n" + transcript
        )


   
    # MODIFIED: USE PREVIOUS INCIDENT INFORMATION
    # WHY:
    # Previously incident_info was reset to "" on every
    # recording. Now we pass the previously extracted
    # information so LangGraph can update it instead of
    # starting from scratch.
   

    initial_state = {

        # MODIFIED:
        # Pass complete conversation instead of only the
        # latest transcript.
        "transcript": st.session_state.conversation_transcript,

        # MODIFIED:
        # Preserve previously extracted incident information.
        "incident_info": st.session_state.incident_info,

        "missing_question": "",

        "summary": ""
    }


   
    # LANGGRAPH PROCESSING
   

    with st.spinner(
        "🤖 LangGraph is processing..."
    ):

        result = app.invoke(
            initial_state
        )


   
    # MODIFIED: SAVE UPDATED INCIDENT INFORMATION
    #
    # WHY:
    # The next follow-up must know what was already extracted.
   

    st.session_state.incident_info = result["incident_info"]


   
    # DISPLAY EXTRACTED INFORMATION
   

    st.subheader(
        "🚨 Extracted Incident Information"
    )

    st.write(
        result["incident_info"]
    )


   
    # DISPLAY FOLLOW-UP QUESTION
   

    if result["missing_question"]:

        st.warning(
            "⚠️ Some important information is missing."
        )

        st.subheader(
            "❓ Follow-up Question"
        )

        st.info(
            result["missing_question"]
        )


   
    # DISPLAY FINAL SUMMARY
   

    else:

        st.success(
            "✅ All required information is available."
        )

        st.subheader(
            "📋 Emergency Summary"
        )

        st.write(
            result["summary"]
        )



# MODIFIED: OPTIONAL DEBUG / CONVERSATION HISTORY DISPLAY
#
# WHY:
# This lets you see that previous transcripts are actually
# being preserved and appended after every recording.


with st.expander("🗣️ Complete Conversation History"):

    if st.session_state.conversation_transcript:

        st.write(
            st.session_state.conversation_transcript
        )

    else:

        st.write(
            "No conversation recorded yet."
        )