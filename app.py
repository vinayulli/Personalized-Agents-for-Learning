import streamlit as st
import asyncio
import uuid
import os
import json
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from agent import run_agent, generate_analysis
from vector_store import ingest_text

# Handling imports for file processing with error feedback
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx2txt
except ImportError:
    docx2txt = None

# Page Config
st.set_page_config(page_title="Personalized Learning Agent", layout="wide")

st.title("Personalized Learning Companion 🎓")

# Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

def process_uploaded_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            if pypdf:
                pdf_reader = pypdf.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                st.error("PDF processing unavailable (missing pypdf).")
                return None
        elif uploaded_file.name.endswith(".docx"):
            if docx2txt:
                text = docx2txt.process(uploaded_file)
            else:
                st.error("DOCX processing unavailable (missing docx2txt).")
                return None
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format. Please upload .pdf, .docx, or .txt.")
            return None
        return text
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def extract_json_from_response(response_text):
    try:
        # Regex to find JSON block
        match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        return None
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return None

def format_message(text):
    """
    Helper to format message text for Streamlit.
    Replaces LaTeX delimiters \[ \] and \( \) with $$ and $ for correct rendering.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Replace block math \[ ... \] with $$ ... $$
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    
    # Replace inline math \( ... \) with $ ... $
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    
    return text

def create_performance_chart(results_by_category):
    category_data = []
    for category, data in results_by_category.items():
        accuracy = (data["correct"] / data["total"]) * 100
        category_data.append({
            "Category": category, 
            "Correct": data["correct"],
            "Total": data["total"],
            "Accuracy": accuracy
        })
    
    if not category_data:
        return None

    df_cat = pd.DataFrame(category_data)
    
    fig = go.Figure()
    
    # Correct bar
    fig.add_trace(go.Bar(
        x=df_cat['Category'],
        y=df_cat['Correct'],
        name='Correct',
        marker_color='#2E8B57', # SeaGreen
        text=[f"{v} ({p:.0f}%)" for v, p in zip(df_cat['Correct'], df_cat['Accuracy'])],
        textposition='auto',
    ))
    
    # Incorrect bar
    fig.add_trace(go.Bar(
        x=df_cat['Category'],
        y=df_cat['Total'] - df_cat['Correct'],
        name='Incorrect',
        marker_color='#CD5C5C', # IndianRed
        textposition='auto',
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Performance by Category',
        xaxis_title='Category',
        yaxis_title='Questions',
        legend_title='Result',
        height=400
    )
    return fig

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.write(f"User ID: {st.session_state.user_id}")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.quiz_data = None
        st.session_state.quiz_answers = {}
        st.rerun()
    
    st.divider()
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        if st.button("Process & Upload"):
            with st.spinner("Processing document..."):
                text = process_uploaded_file(uploaded_file)
                if text:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                        
                    try:
                        num_chunks = ingest_text(
                            text, 
                            metadata={"source": uploaded_file.name},
                            status_callback=update_progress
                        )
                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"Successfully added {uploaded_file.name} ({num_chunks} chunks) to Knowledge Base!")
                    except Exception as e:
                        st.error(f"Failed to ingest document: {e}")

# Main Chat Logic
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # If the message contains quiz results, display the chart FIRST
        if "quiz_results" in message:
            st.markdown(f"### Assessment Score: {message['quiz_results']['score']} / {message['quiz_results']['total']}")
            fig = create_performance_chart(message['quiz_results']['category_analysis'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Then display the text content (Analysis) with formatting
        st.markdown(format_message(message["content"]))

# Quiz Display Logic
if st.session_state.quiz_data:
    st.divider()
    st.subheader("📝 Knowledge Assessment")
    
    with st.form("quiz_form"):
        for i, q in enumerate(st.session_state.quiz_data['quiz']):
            category = q.get('category', 'General')
            # Format question text too just in case
            question_text = format_message(q['question'])
            st.markdown(f"**{i+1}. [{category}] {question_text}**")
            st.session_state.quiz_answers[i] = st.radio(
                "Choose an answer:", 
                q['options'], 
                key=f"q_{i}", 
                index=None,
                label_visibility="collapsed"
            )
            st.write("---")
        
        submit_quiz = st.form_submit_button("Submit Assessment")
        
        if submit_quiz:
            # Calculate Results
            score = 0
            total = len(st.session_state.quiz_data['quiz'])
            results_by_category = {}
            
            for i, q in enumerate(st.session_state.quiz_data['quiz']):
                user_answer = st.session_state.quiz_answers.get(i)
                correct_answer = q['correct_answer']
                category = q.get('category', 'General')
                
                if category not in results_by_category:
                    results_by_category[category] = {"correct": 0, "total": 0}
                results_by_category[category]["total"] += 1
                
                if user_answer == correct_answer:
                    score += 1
                    results_by_category[category]["correct"] += 1

            # Generate Analysis
            with st.spinner("Analyzing results..."):
                analysis_result = asyncio.run(generate_analysis(
                    st.session_state.user_id, 
                    {
                        "score": score, 
                        "total": total, 
                        "category_analysis": results_by_category
                    }
                ))
                
                # Add to history with special 'quiz_results' data
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"**Assessment Analysis:**\n\n{analysis_result}",
                    "quiz_results": {
                        "score": score,
                        "total": total,
                        "category_analysis": results_by_category
                    }
                })
            
            # Close Quiz Mode
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = {}
            st.rerun()

# User Input (only if not taking a quiz)
if not st.session_state.quiz_data:
    if prompt := st.chat_input("Ask me anything or request an assessment..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# Handle Assistant Response Generation
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                prompt = st.session_state.messages[-1]["content"]
                chat_history_to_pass = st.session_state.messages[:-1]
                
                response = asyncio.run(run_agent(
                    st.session_state.user_id, 
                    prompt, 
                    chat_history=chat_history_to_pass
                ))
                
                quiz_json = extract_json_from_response(response)
                if quiz_json:
                    st.session_state.quiz_data = quiz_json
                    st.session_state.quiz_answers = {}
                    st.rerun()
                else:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")
