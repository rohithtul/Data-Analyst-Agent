import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import contextlib
import io
import os
import logging

# --- Agent and LangChain Imports ---
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from google.api_core.exceptions import NotFound
import json
import urllib.request
import urllib.parse

# --- 1. CUSTOM TOOL DEFINITION (The Streamlit-Aware Execution Engine) ---

def execute_code(code: str) -> dict:
    """
    Executes Python code and captures the output. It is specifically designed
    to handle Matplotlib plots for display in Streamlit.
    """
    global_vars = {
        "pd": pd,
        "plt": plt,
        "df": st.session_state['df'],
        "st": st
    }
    
    buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, global_vars)
        
        text_output = buffer.getvalue()
        
        fig = plt.gcf()
        if len(fig.get_axes()) > 0:
            st.pyplot(fig)
            plt.clf()
            output = {
                "text": text_output,
                "plot": True
            }
        else:
            output = {
                "text": text_output,
                "plot": False
            }
            
    except Exception as e:
        output = {
            "text": f"Error executing code: {e}",
            "plot": False
        }
    
    return f"Execution Output: {output['text']}"


# --- 2. AGENT INITIALIZATION (The Gemini Brain) ---

def initialize_agent(df: pd.DataFrame):
    
    gemini_key = os.environ.get("GEMINI_API_KEY")

    def select_supported_model(api_key: str):
        """Try to initialize the LLM with a model chosen from an env var or a list of fallbacks.

        Returns (llm_instance, model_name) on success or (None, None) on failure.
        """
        # Allow user to override preferred model via env var GEMINI_MODEL
        env_model = os.environ.get("GEMINI_MODEL")
        candidate_models = []
        if env_model:
            candidate_models.append(env_model)

        # Updated list of candidate models based on your available list.
        # Order is from most to least preferred.
        candidate_models.extend([
            "models/gemini-2.5-pro",
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro-preview-06-05",
            "models/gemini-2.5-flash-preview-05-20",
            "models/gemini-2.0-pro-exp",
            "models/gemini-2.0-flash",
        ])

        # Deduplicate while keeping order
        seen = set()
        candidate_models = [m for m in candidate_models if not (m in seen or seen.add(m))]

        for model_name in candidate_models:
            try:
                # Use the full model name as returned by the API
                llm_try = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    google_api_key=api_key,
                )
                logging.info(f"Initialized Gemini LLM with model: {model_name}")
                return llm_try, model_name
            except NotFound as nf:
                # Model doesn't exist for this API/version — try next fallback
                logging.warning(f"Model not found: {model_name} - {nf}")
                continue
            except Exception as e:
                # For other errors, log and continue trying other models
                logging.warning(f"Error initializing model {model_name}: {e}")
                continue

        return None, None

    llm, chosen_model = select_supported_model(gemini_key)

    if llm is None:
        st.error(
            "Error initializing Gemini: no supported model could be initialized. "
            "Set the GEMINI_MODEL env var to a supported model (or call ListModels to discover available models)."
        )
        st.stop()

    if chosen_model != os.environ.get("GEMINI_MODEL") and os.environ.get("GEMINI_MODEL"):
        st.warning(f"Requested GEMINI_MODEL='{os.environ.get('GEMINI_MODEL')}' was not available; using '{chosen_model}' instead.")
        
    tools = [
        Tool(
            name="python_code_executor",
            func=execute_code,
            description="""
            A Python shell. Use this to execute python commands to answer questions about the dataframe 'df'.
            The dataframe 'df' is already loaded.
            - For any text-based output or analysis (like df.head(), df.describe(), or calculations), you MUST use the print() function.
            - For visualizations, create a plot using Matplotlib and call plt.show() at the end.
            This tool will automatically display the plot in the UI.
            Example for text: print(df.head())
            Example for plot: plt.hist(df['Age']); plt.show()
            """
        )
    ]

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )


def list_available_models(api_key: str):
    """Return a tuple (success, message_or_list).

    success: bool
    message_or_list: list of model names when success=True, or error string when success=False
    """
    # Try using the google.generativeai client if available
    try:
        from google.generativeai import client as genai_client
        try:
            genai_client.configure(api_key=api_key)
            models = genai_client.list_models()
            names = []
            for m in models:
                if isinstance(m, dict):
                    names.append(m.get('name'))
                else:
                    names.append(getattr(m, 'name', str(m)))
            return True, names
        except Exception as e:
            logging.warning(f"google.generativeai client present but failed to list models: {e}")
    except Exception:
        pass

    # REST fallback
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    url = f"{base_url}?key={urllib.parse.quote(api_key)}"
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read().decode('utf-8')
            parsed = json.loads(data)
            if isinstance(parsed, dict) and 'models' in parsed:
                return True, [m.get('name') for m in parsed['models']]
            return True, parsed
    except urllib.error.HTTPError as he:
        body = he.read().decode('utf-8')
        return False, f"HTTP Error: {he.code} {he.reason} - {body}"
    except Exception as e:
        return False, f"Failed to list models: {e}"

# --- 3. STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Agentic Data Analyst", layout="wide")
    st.title("📊 Agentic Data Analyst (Gemini-Powered)")
    st.markdown("Upload a CSV file and converse with your data. The Agent will autonomously write and run Python code to analyze and visualize your data.")
    
    # Sidebar: option to list available models
    with st.sidebar:
        st.header("Model tools")
        if st.button("List available Gemini models"):
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                st.error("GEMINI_API_KEY not set in environment. Set it and try again.")
            else:
                with st.spinner("Querying available models..."):
                    success, result = list_available_models(gemini_key)
                    if success:
                        if isinstance(result, list):
                            st.success(f"Found {len(result)} models")
                            for name in result:
                                st.write(name)
                        else:
                            st.json(result)
                    else:
                        st.error(result)
    
    if 'df' not in st.session_state:
        try:
            st.session_state['df'] = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
            st.info("Loaded default **Titanic** dataset for testing. Upload your own CSV below.")
        except Exception:
            st.error("Could not load default dataset.")
            st.stop()

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
     df = pd.read_csv(uploaded_file)
     st.session_state['df'] = df
     st.session_state.pop('agent_executor', None)
     st.session_state.pop('messages', None)
     st.rerun()

    df = st.session_state['df']
    
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    if 'agent_executor' not in st.session_state or 'messages' not in st.session_state:
        with st.spinner("Initializing Agent with DataFrame..."):
            st.session_state['agent_executor'] = initialize_agent(df)
            st.session_state['messages'] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "plot" in message and message["plot"]:
                st.pyplot()
            st.write(message["content"])

    user_input = st.chat_input("Ask a question about the data (e.g., 'Plot a histogram of Age' or 'What is the average fare?')")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            agent_executor = st.session_state['agent_executor']
            
            with st.spinner("Agent is thinking and executing code..."):
                try:
                    response = agent_executor.invoke({"input": user_input})
                except NotFound as nf:
                    logging.error(f"Model not found during invocation: {nf}")
                    st.error(
                        "The model was not found when trying to process your request. "
                        "This can happen if the selected model is unavailable for your account or the API version. "
                        "Try setting the GEMINI_MODEL environment variable to a supported model or contact your admin."
                    )
                    st.session_state.messages.append({"role": "assistant", "content": "Error: model not found during invocation."})
                    response = {"output": "Error: model not found during invocation."}
                except Exception as e:
                    logging.exception("Error invoking agent")
                    st.error(f"An error occurred while the agent was processing your request: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error invoking agent: {e}"})
                    response = {"output": f"Error invoking agent: {e}"}

                st.write(response['output'])
                
                st.session_state.messages.append({"role": "assistant", "content": response['output']})

if __name__ == "__main__":
    if 'GEMINI_API_KEY' not in os.environ:
        st.error("🚨 **CRITICAL ERROR:** GEMINI_API_KEY environment variable not found!")
    else:
        main()