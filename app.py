import os
import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Environment Variables
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# GROQ API KEY
GROQ_API_KEY =st.secrets["GROQ_API_KEY"] 

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="AKS University AI Assistant",
    page_icon="🎓",
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.78), rgba(15, 23, 42, 0.88)), 
                    url('https://www.aksuniversity.ac.in/sites/default/files/2024-04/aks_a_block.jpeg') no-repeat center center fixed;
        background-size: cover;
    }

    .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #ffffff !important;
        text-align: center;
        font-weight: 700;
        text-shadow: 0px 3px 6px rgba(0,0,0,0.7);
    }

    .stChatMessage {
        border-radius: 16px;
        padding: 12px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        flex-direction: row-reverse !important;
        text-align: right !important;
        background-color: #e0f2fe !important;
        color: #0c4a6e !important;
        margin-left: 10% !important;
        max-width: 90% !important;
        border-bottom-right-radius: 2px !important;
    }

    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1e293b !important;
        margin-right: auto !important;
        max-width: 100% !important;
        border-bottom-left-radius: 2px !important;
    }

    .stChatInputContainer {
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 AKS University AI Assistant")
st.markdown("<p style='text-align: center; color: #e2e8f0; font-weight: 500;'>Satna, Madhya Pradesh | Official AI Helpdesk</p>", unsafe_allow_html=True)

# Helper Function: Browser Headers ke saath Web Page Fetch karna
def fetch_url_text(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extra scripts aur styles ko remove karna
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            text = soup.get_text(separator=' ')
            # Clean extra whitespaces
            clean_text = ' '.join(text.split())
            return clean_text
    except Exception as e:
        pass
    return ""

# --- STEP 1: Robust URL Scraper & VectorDB ---
@st.cache_resource
def load_data_from_urls():
    urls = [
        "https://aksuniversity.ac.in/",
        "https://aksuniversity.ac.in/index.php/about-us",
        "https://aksuniversity.ac.in/Chancellor-Message",
        "https://aksuniversity.ac.in/index.php/Officers-of-University",
        "https://aksuniversity.ac.in/Chairman-Message",
        "https://aksuniversity.ac.in/Vice-Chancellor-Message",
        "https://cs.aksuniversity.ac.in/",
        "https://aksuniversity.ac.in/Search-Program",
        "https://aksuniversity.ac.in/Scholarship",
        "https://aksuniversity.ac.in/latest-fee-structure",
        "https://aksuniversity.ac.in/Awards-and-Tie-Ups",
    ]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "./chroma_db"
    
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = []
    for url in urls:
        text = fetch_url_text(url)
        if text and len(text) > 100:
            docs.append(Document(page_content=text, metadata={"source": url}))
            
    # Backup static document agar kisi karan bas Web Fetch zero aaye
    if not docs:
        docs = [Document(page_content="""
        AKS University, Satna (M.P.) Official Info:
        Chancellor: B.P. Soni, Vice Chancellor: Prof. B.A. Chopade, Pro VC: Prof. R.S. Tripathi.
        Courses Offered: B.Tech, M.Tech, B.Sc Agriculture, MBA, B.Pharm, D.Pharm, BCA, MCA, Law.
        Website: https://aksuniversity.ac.in/ Contact: +91 8889207776
        """)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

with st.spinner("🚀 Website ka live data scrape & database setup ho raha hai..."):
    retriever = load_data_from_urls()

def clean_reasoning_tags(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()

def get_text_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    return str(content)

# --- STEP 2: Groq Live Model ---
def query_groq_live(system_prompt, messages_history, user_msg):
    client = Groq(api_key=GROQ_API_KEY)
    
    priority_models = [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ]
    
    try:
        available_models = [m.id for m in client.models.list().data]
        valid_models = [m for m in priority_models if m in available_models]
        if not valid_models:
            valid_models = [m for m in available_models if "guard" not in m.lower() and "whisper" not in m.lower()]
    except Exception:
        valid_models = priority_models

    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    recent_history = messages_history[-2:] if len(messages_history) > 2 else messages_history
    for msg in recent_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        formatted_messages.append({"role": role, "content": msg.content})
        
    formatted_messages.append({"role": "user", "content": user_msg})

    last_err = None
    for model_id in valid_models:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=formatted_messages,
                temperature=0.1,
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            continue
            
    if last_err is not None:
        raise last_err
    else:
        raise Exception("Groq API connection fail ho gaya.")

# --- STEP 3: Chat UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

if user_input := st.chat_input("AKS University ke bare me koi bhi sawal poochein..."):
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 AKS Database se jawab dhoondha ja raha hai..."):
            try:
                relevant_docs = retriever.invoke(user_input)
                context_text = "\n---\n".join([doc.page_content.strip() for doc in relevant_docs])
                
                system_prompt = (
                    "Aap AKS University, Satna ke official AI assistant hain.\n"
                    "INSTRUCTIONS:\n"
                    "1. User ke sawal ka accurate aur saaf jawab neeche 'Context' mein di gayi information ke aadhar par dein.\n"
                    "2. Apni taraf se koi galat number ya naam guess na karein.\n"
                    "3. Jawab polite, clear aur concise Hindi/Hinglish me dein.\n\n"
                    f"Context:\n{context_text}"
                )
                
                raw_response = query_groq_live(system_prompt, st.session_state.chat_history, user_input)
                extracted_text = get_text_content(raw_response)
                clean_response = clean_reasoning_tags(extracted_text)
                
                st.write(clean_response)
                
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=clean_response))
                
            except Exception as e:
                st.error(f"Error aaya: {str(e)}")