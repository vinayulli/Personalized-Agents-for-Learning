import os
import traceback
import json
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools import get_tools
from memory import get_memories, add_memory
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

tools = get_tools()

# Define the system prompt template
SYSTEM_PROMPT_TEMPLATE = """You are a Personalized Learning Agent. Your goal is to help the user understand topics, conduct assessments, and suggest improvements.

You have access to the following tools:
1. Knowledge Base (Qdrant): Use this to find specific information about the topic in the provided materials.
2. Web Search (Tavily): Use this to find supplementary information or recent developments if the knowledge base is insufficient.

Personalization Context (User Memories):
{memories}

Instructions:
- **PERSONALIZATION**: Always address the user by name if it is known or available in the context/memories.
- Answer the user's questions using the tools provided.
- If the user asks to learn about a topic, explain it clearly.
- **FORMATTING**:
  - Use Markdown for all text.
  - **MATH**: For mathematical formulas, ALWAYS use Streamlit-compatible LaTeX.
    - Inline math: $ E = mc^2 $ (single dollar signs)
    - Block math: $$ E = mc^2 $$ (double dollar signs)
    - DO NOT use `\[` or `\(` syntax.

- **ASSESSMENT GENERATION**:
  - If the user asks for an assessment, quiz, or test:
    - Generate 10 Multiple Choice Questions (MCQs) by default, unless the user specifies a different number.
    - **CRITICAL**: Group the questions into 2-3 distinct conceptual categories (e.g., "Basic Concepts", "Advanced Application", "History", "Syntax", etc.).
    - You MUST return the response in a strictly valid JSON block wrapped in ```json and ```.
    - Do not include conversational text outside the JSON block when generating a quiz.
    - The JSON format must be:
      {{
        "quiz": [
          {{
            "question": "Question text (can include math in $...$)",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "correct_answer": "The correct option text",
            "category": "Category Name (e.g., Basics, Advanced, etc.)"
          }},
          ...
        ]
      }}

- For non-assessment queries:
  - Provide feedback based on their answers.
  - Suggest relevant links or resources for further improvement.
  - Be supportive and act as a companion.
  - You are multi-lingual: Detect the user's language and respond in the same language.

Current User Input: {input}
"""

def get_agent_executor():
    # We use the prebuilt ReAct agent for simplicity and effectiveness
    return create_react_agent(llm, tools)

async def run_agent(user_id: str, user_input: str, chat_history: list = None):
    print(f"DEBUG: Starting agent run for user {user_id}")
    
    try:
        # 1. Fetch Memories
        print("DEBUG: Fetching memories...")
        memories_list = get_memories(user_id=user_id, query=user_input)
        
        memories_text = ""
        if memories_list:
            if isinstance(memories_list, list) and len(memories_list) > 0:
                if isinstance(memories_list[0], dict):
                    memories_text = "\n".join([m.get('memory', str(m)) for m in memories_list])
                else:
                    memories_text = "\n".join([str(m) for m in memories_list])
            else:
                 memories_text = str(memories_list)
        
        print(f"DEBUG: Memories fetched. Context length: {len(memories_text)}")
    except Exception as e:
        print(f"ERROR: Failed to fetch memories: {e}")
        memories_text = "Memory system unavailable."

    # 2. Prepare System Message (Injecting Context)
    formatted_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        memories=memories_text,
        input=user_input
    )
    
    system_message = SystemMessage(content=formatted_system_prompt)
    
    # 3. Prepare Chat History
    history_messages = []
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_messages.append(AIMessage(content=msg["content"]))
    
    agent = get_agent_executor()
    
    messages_payload = [system_message] + history_messages + [HumanMessage(content=user_input)]
    
    inputs = {"messages": messages_payload}
    
    # 4. Run Agent
    print("DEBUG: Invoking agent with history...")
    try:
        result = await agent.ainvoke(inputs)
    except Exception as e:
        print(f"ERROR: Agent invocation failed: {e}")
        traceback.print_exc()
        raise e
    
    # Extract the final response
    final_response = result['messages'][-1].content
    print("DEBUG: Agent finished.")
    
    # 5. Save interaction to Memory
    print("DEBUG: Saving to memory...")
    try:
        if "```json" in final_response:
             add_memory(user_id, f"User asked: {user_input}\nAgent generated a quiz.")
        else:
             add_memory(user_id, f"User asked: {user_input}\nAgent answered: {final_response}")
    except Exception as e:
        print(f"ERROR: Failed to save memory: {e}")
    
    return final_response

async def generate_analysis(user_id: str, quiz_results: dict):
    """
    Generates insights based on quiz results.
    """
    score = quiz_results['score']
    total = quiz_results['total']
    category_analysis = quiz_results['category_analysis']
    
    # Fetch memories
    memories_list = get_memories(user_id=user_id, query="learning style strengths weaknesses")
    memories_text = ""
    if memories_list:
        if isinstance(memories_list, list) and len(memories_list) > 0:
            if isinstance(memories_list[0], dict):
                memories_text = "\n".join([m.get('memory', str(m)) for m in memories_list])
            else:
                memories_text = "\n".join([str(m) for m in memories_list])
        else:
             memories_text = str(memories_list)

    prompt = f"""
    The user just completed a quiz.
    Score: {score}/{total}
    
    Category Breakdown (Correct/Total per category):
    {json.dumps(category_analysis, indent=2)}
    
    User Context/Memories:
    {memories_text}
    
    Please provide a personalized analysis using the available tools (Tavily Search) to find REAL URLs for resources.
    
    Instructions:
    1. Address the user by name if known.
    2. Praise strengths naturally (do not use "Encouragement" as a heading).
    3. Identify areas for improvement.
    4. **CRITICAL**: Use the search tool to find 2-3 ACTUAL, clickable URLs for resources (tutorials, documentation, articles) specific to the weak areas.
    5. Return the response as clean markdown. Do not use generic headings like "Encouragement".
    """
    
    # Use the AGENT EXECUTOR here instead of raw LLM so it can use TOOLS (Tavily)
    # We create a temporary run for analysis
    agent = get_agent_executor()
    
    result = await agent.ainvoke({"messages": [
        SystemMessage(content="You are a helpful learning assistant who generates post-quiz analysis and finds real learning resources using web search."),
        HumanMessage(content=prompt)
    ]})
    
    return result['messages'][-1].content
