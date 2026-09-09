import os
import base64
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END



# PAGE CONFIGURATION


st.set_page_config(
    page_title="ResQAI - Emergency Response Agent",
    page_icon="🚨",
    layout="wide"
)



# BACKGROUND IMAGE


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



# TITLE


st.title(
    "🚨 ResQAI — Real-Time Voice-Based Intelligent Emergency Response Agent"
)

st.write(
    "AI agent for collecting, understanding, and structuring "
    "emergency call information."
)



# OPENAI API KEY


load_dotenv(override=True)

api_key = st.text_input(
    "🔑 OpenAI API Key",
    type="password",
    placeholder="Enter your OpenAI API key"
)

if not api_key:

    st.warning(
        "Please enter your OpenAI API key to continue."
    )

    st.stop()


client = OpenAI(
    api_key=api_key
)



# SESSION STATE


if "conversation_transcript" not in st.session_state:

    st.session_state.conversation_transcript = ""


if "incident_info" not in st.session_state:

    st.session_state.incident_info = ""



# LANGGRAPH STATE


class EmergencyState(TypedDict):

    transcript: str
    incident_info: str
    missing_question: str
    summary: str



# TRANSCRIBE AUDIO


def transcribe_audio(audio_file):

    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    return result.text



# EXTRACT EMERGENCY INFORMATION


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



# GENERATE EMERGENCY SUMMARY


def generate_summary(transcript, incident_info):

    prompt = f"""
You are an Emergency Response Assistant.

Based ONLY on the complete conversation transcript and
extracted information, create a concise emergency incident summary.

Complete Conversation:

{transcript}


Extracted Information:

{incident_info}


Create the summary in this format in English language only:

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

workflow.add_node(
    "update_information",
    update_information_node
)

workflow.add_node(
    "check_missing",
    check_missing_node
)

workflow.add_node(
    "follow_up",
    follow_up_node
)

workflow.add_node(
    "summary",
    summary_node
)


# START → Update Information

workflow.add_edge(
    START,
    "update_information"
)


# Update Information → Check Missing

workflow.add_edge(
    "update_information",
    "check_missing"
)


# Conditional Routing

workflow.add_conditional_edges(
    "check_missing",
    route_after_check,
    {
        "follow_up": "follow_up",
        "summary": "summary"
    }
)


# Follow-up → END

workflow.add_edge(
    "follow_up",
    END
)


# Summary → END

workflow.add_edge(
    "summary",
    END
)


# Compile

app = workflow.compile()



# STREAMLIT UI


st.divider()

st.subheader("🎤 Voice Emergency Call")

st.write(
    "Click the microphone button and describe the emergency."
)


# Browser-based microphone
audio_value = st.audio_input(
    "🎤 Record Emergency Call"
)



# PROCESS AUDIO


if audio_value is not None:

    st.success(
        "✅ Recording completed!"
    )



    # TRANSCRIPTION


    with st.spinner(
        "📝 Converting speech to text..."
    ):

        transcript = transcribe_audio(
            audio_value
        )


    st.subheader(
        "📝 Latest Transcript"
    )

    st.write(
        transcript
    )



    # SAVE COMPLETE CONVERSATION


    if st.session_state.conversation_transcript:

        st.session_state.conversation_transcript += (
            "\n\nCaller:\n" + transcript
        )

    else:

        st.session_state.conversation_transcript = (
            "Caller:\n" + transcript
        )



    # INITIAL LANGGRAPH STATE


    initial_state = {

        "transcript":
            st.session_state.conversation_transcript,

        "incident_info":
            st.session_state.incident_info,

        "missing_question": "",

        "summary": ""
    }



    # LANGGRAPH PROCESSING


    with st.spinner(
        "🤖 ResQAI Agent is processing..."
    ):

        result = app.invoke(
            initial_state
        )



    # SAVE INCIDENT INFORMATION


    st.session_state.incident_info = (
        result["incident_info"]
    )



    # DISPLAY INCIDENT INFORMATION


    st.subheader(
        "🚨 Extracted Incident Information"
    )

    st.write(
        result["incident_info"]
    )



    # FOLLOW-UP QUESTION


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



    # FINAL SUMMARY


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



# COMPLETE CONVERSATION HISTORY


with st.expander(
    "🗣️ Complete Conversation History"
):

    if st.session_state.conversation_transcript:

        st.write(
            st.session_state.conversation_transcript
        )

    else:

        st.write(
            "No conversation recorded yet."
        )
