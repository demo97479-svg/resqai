# 🚨 ResQAI — Real-Time Voice-Based Intelligent Emergency Response Agent

ResQAI is a **voice-based intelligent emergency response agent** built with **Python, Streamlit, OpenAI, and LangGraph**.

The system accepts emergency information through voice input, converts speech into text, extracts critical incident information, identifies missing information, asks follow-up questions, maintains conversation state, and generates a concise emergency incident summary.

> **Note:** The current implementation processes short voice recordings immediately after recording. Therefore, "real-time" refers to near-real-time interactive processing rather than continuous live audio streaming.

---

## 🎯 Problem Statement

During emergency situations, callers may provide incomplete or unstructured information such as:

* What happened?
* Where did it happen?
* How many people are involved?
* Is anyone injured?
* Are there any hazards?
* How serious is the situation?

Important information can easily be missed during a stressful emergency call.

**ResQAI** addresses this problem by using a voice-based AI agent to:

1. Capture the caller's voice.
2. Convert speech into text.
3. Extract important emergency information.
4. Identify missing information.
5. Ask intelligent follow-up questions.
6. Maintain information across multiple responses.
7. Generate a structured emergency summary.

---

## ✨ Key Features

### 🎤 Voice-Based Input

The user can record emergency information directly through a microphone.

### 📝 Speech-to-Text

Recorded audio is converted into text using OpenAI Whisper.

### 🧠 AI Information Extraction

The AI extracts:

* Emergency Type
* Location
* People Involved
* Injuries
* Hazards
* Severity

### 🤖 Agentic Workflow with LangGraph

LangGraph manages the application's state and workflow.

The agent decides whether:

* More information is required → Ask a follow-up question
* All required information is available → Generate the emergency summary

### 🔄 Stateful Conversation

The application remembers previous caller responses using Streamlit Session State.

New information is added to previously extracted information rather than starting from scratch.

### ❓ Intelligent Follow-Up Questions

If important information is missing, the system generates a targeted question.

Example:

> "What is the exact location of the emergency?"

### 📋 Emergency Summary

Once all required information is available, the system generates a concise structured emergency summary.

### 🖼️ Custom Background

The Streamlit interface supports a local background image with a dark transparent overlay for improved readability.

---

## 🏗️ System Architecture

```text
                    🎤 Voice Input
                         │
                         ▼
                ┌─────────────────┐
                │ Audio Recording │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Speech-to-Text  │
                │ OpenAI Whisper  │
                └────────┬────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │   LangGraph    │
                 │ Agent Workflow │
                 └───────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Extract Information  │
              └──────────┬───────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Check Missing   │
                │ Information     │
                └────────┬────────┘
                         │
                ┌────────┴─────────┐
                │                  │
              Missing            Complete
                │                  │
                ▼                  ▼
       ┌────────────────┐  ┌────────────────┐
       │ Follow-up      │  │ Emergency      │
       │ Question       │  │ Summary        │
       └───────┬────────┘  └────────────────┘
               │
               ▼
          🎤 New Voice Input
```

---

## 🔄 LangGraph Agent Workflow

The application uses a state graph:

```text
START
  │
  ▼
update_information
  │
  ▼
check_missing
  │
  ├────────────── Missing ──────────────► follow_up
  │                                        │
  │                                        ▼
  │                                       END
  │
  └────────────── Complete ─────────────► summary
                                           │
                                           ▼
                                          END
```

### LangGraph Nodes

#### 1. `update_information`

Extracts and updates emergency information using the complete conversation.

#### 2. `check_missing`

Checks whether any required emergency information is unavailable.

#### 3. `follow_up`

Returns a question requesting the missing information.

#### 4. `summary`

Generates the final emergency incident summary.

### Conditional Routing

The workflow dynamically chooses the next step:

```python
if state["missing_question"]:
    return "follow_up"
else:
    return "summary"
```

This makes the application an **agentic, stateful workflow** rather than a simple linear LLM chain.

---

## 🧠 Emergency State

LangGraph maintains the following state:

```python
class EmergencyState(TypedDict):

    transcript: str
    incident_info: str
    missing_question: str
    summary: str
```

### State Fields

| Field              | Purpose                                      |
| ------------------ | -------------------------------------------- |
| `transcript`       | Complete conversation                        |
| `incident_info`    | Extracted emergency information              |
| `missing_question` | Follow-up question if information is missing |
| `summary`          | Final emergency summary                      |

---

## 📊 Information Extracted

The agent maintains six important fields:

```text
Emergency Type
Location
People Involved
Injuries
Hazards
Severity
```

Severity is classified as:

```text
Low
Medium
High
Critical
```

The system does not intentionally invent missing information. Unknown values are represented as:

```text
Not available
```

---

## 🛠️ Technologies Used

| Technology          | Purpose                                  |
| ------------------- | ---------------------------------------- |
| Python              | Core programming language                |
| Streamlit           | Web application interface                |
| OpenAI Whisper      | Speech-to-text                           |
| OpenAI GPT-4.1-mini | Information extraction and summarization |
| LangGraph           | Agentic workflow and state management    |
| SoundDevice         | Microphone/audio recording               |
| SciPy               | WAV file creation                        |        
| Base64              | Local background image embedding         |

---

## 📁 Project Structure

```text
ResQAI/
│
├── app.py
├── requirements.txt
├── README.md
├── background.jpg
└── question.wav
```

### File Description

**`app.py`**
Main Streamlit application containing the voice recording, AI processing, LangGraph workflow, and UI.

**`requirements.txt`**
Python dependencies required to run the project.

**`background.jpg`**
Background image used by the Streamlit application.



**`question.wav`**
Temporary audio file generated during voice recording.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ResQAI.git
cd ResQAI
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

add your openai key in the browser itself.

## 📦 requirements.txt

```text
streamlit
sounddevice
scipy
openai
langgraph
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🚨 Example Interaction

### Caller

> "There has been a car accident near the main road. Two people are involved and one person is injured."

### AI Extracts

```text
Emergency Type: Car Accident
Location: Main Road
People Involved: 2
Injuries: 1 person injured
Hazards: Not available
Severity: Not available
```

### Agent Follow-Up

```text
Are there any current hazards?
```

The caller can provide another voice response.

The system updates the existing information and checks again for missing information.

---

## 🔄 Stateful Conversation

Streamlit reruns the Python script whenever the user interacts with the application.

To preserve information between interactions, ResQAI uses:

```python
st.session_state
```

Two important session variables are maintained:

```python
st.session_state.conversation_transcript
st.session_state.incident_info
```

This allows the system to maintain the same emergency incident across multiple voice inputs.

---

## 🤖 Why LangGraph?

A simple LLM call would look like:

```text
Input → LLM → Output
```

ResQAI requires a more dynamic workflow:

```text
Input
 ↓
Extract information
 ↓
Check missing information
 ↓
Decision
 ├── Missing → Follow-up
 └── Complete → Summary
```

LangGraph is useful because it provides:

* Stateful workflows
* Conditional routing
* Multiple nodes
* Workflow orchestration
* Iterative processing
* Agentic decision flow

Therefore, LangGraph acts as the **orchestration layer for the emergency response agent**.

---

## 🎓 Learning Outcomes

This project demonstrates practical concepts in:

* Generative AI
* Voice AI
* Speech-to-Text
* Prompt Engineering
* LLM Application Development
* LangGraph
* Agentic AI
* Stateful AI Applications
* Conditional Workflows
* Streamlit
* Session State
* AI-powered Information Extraction
* AI-powered Summarization


## 🚀 Future Improvements

Possible future enhancements include:

* 🔊 Text-to-Speech responses
* 🎙️ Continuous voice streaming
* 📞 Real-time voice conversation
* 🗺️ GPS/location integration
* 🚑 Emergency service integration
* 📱 Mobile application
* 🧑‍🚒 Dispatcher dashboard
* 🌐 Multi-language emergency support
* 🗃️ Database-based incident storage
* 🔐 Authentication and role-based access
* 📊 Emergency analytics dashboard
* 🧠 More advanced agent/tool integration
* 📡 Real-time communication with emergency response teams


## 🔒 Safety Considerations

ResQAI is designed as an AI information-collection and summarization prototype.

The application intentionally avoids:

* Claiming that emergency services were contacted.
* Providing medical diagnoses.
* Inventing emergency information.
* Assuming information that was not provided.

For real-world emergency deployment, human oversight and validated emergency-response procedures would be essential.



## ⭐ Project Highlights


🎤 Voice-Based AI
🧠 Generative AI
🤖 Agentic Workflow
🔄 Stateful Conversation
🕸️ LangGraph
📝 Speech-to-Text
🚨 Emergency Intelligence
📋 AI Summarization
🌐 Streamlit

