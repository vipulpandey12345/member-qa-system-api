"""
    This Flask app answers questions about member messages using a RAG pipeline that combines vector search with GPT to generate accurate responses. 
    When it starts up, it loads all messages from an API into a ChromaDB vector database, and then a background thread continuously checks for new messages every hour to keep the database fresh. 
    When you ask a question, it first uses GPT to figure out which member you're asking about, then searches the vector database filtering only that person's messages using their user_id to avoid mixing up information between different members. 
    Finally, it feeds the retrieved message chunks (with formatted dates) into GPT along with your question to generate a clear, contextually-grounded answer.


"""
from datetime import datetime
import os
import json
import time
import threading
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


"""
    The following components are used:
    - Flask: to create the web application and handle the requests
    - Chroma: vector database to store the messages
    - OpenAI: to generate the embeddings and answer the questions
    - RecursiveCharacterTextSplitter: to split the messages into chunks
    - Threading: to run the background service continously every hour to fetch new messages
    - Requests: to fetch the messages from the remote API
    - JSON: to parse the responses from the language model
    - Dotenv: to load the environment variables

    known_timestamps is to keep track of messages that have already been added to the vector DB to ensure duplicates don't get embedded.
    user_map defines user names -> user ID mapping and tracks which users exist in-case the question is about someone not listed in message vector DB.
    This was an important way to filter out unknown users or people where there is no relevant-context.

"""



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

API_URL = "https://november7-730026606190.europe-west1.run.app/messages/?skip=0&limit=3349"
VECTOR_DB_DIR = "./chroma_store"
CHECK_INTERVAL = 3600

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

vector_db = Chroma(
    collection_name="member_messages",
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

known_timestamps = set()
user_map = {}

app = Flask(__name__)


def fetch_messages():
    """
        Fetch messages from remote API and returns back as JSON. This helper function is used to bootstrap the vector DB 
        and then pull only new messages on continuous basis every hour via the background thread.
    """
    try:
        resp = requests.get(API_URL)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f" Error fetching API: {e}")
        return {"total": 0, "items": []}


def bootstrap_vector_db():
    """
        Initial load of all messages into the vector DB. It gets the messages by calling fetch_messages().
    """
    global known_timestamps, user_map
    data = fetch_messages()
    items = data.get("items", [])

    docs, metas = [], []
    for m in items:
        user_map[m["user_name"].lower()] = m["user_id"]
        known_timestamps.add(m["timestamp"])

        for chunk in splitter.split_text(m["message"]):
            docs.append(chunk)
            metas.append({
                "user_id": m["user_id"],
                "user_name": m["user_name"],
                "timestamp": m["timestamp"],
                "id": m["id"]
            })

    if docs:
        vector_db.add_texts(docs, metadatas=metas)
        print(f"Bootstrapped {len(docs)} message chunks.")


def update_vector_db():
    """
        Background loop that polls the API for new messages. For every message, it splits the message chunks and records different metadata about the chunk.
        This method runs the background thread every 3600 seconds and writes all data to persistent storage. 
    """
    global known_timestamps, user_map
    while True:
        try:
            data = fetch_messages()
            items = data.get("items", [])
            new_msgs = [m for m in items if m["timestamp"] not in known_timestamps]

            if not new_msgs:
                print("No new messages found.")
            else:
                print(f"Found {len(new_msgs)} new messages, updating...")
                for m in new_msgs:
                    user_map[m["user_name"].lower()] = m["user_id"]
                    known_timestamps.add(m["timestamp"])

                    for chunk in splitter.split_text(m["message"]):
                        vector_db.add_texts(
                            [chunk],
                            metadatas=[{
                                "user_id": m["user_id"],
                                "user_name": m["user_name"],
                                "timestamp": m["timestamp"],
                                "id": m["id"]
                            }]
                        )

                vector_db.persist()
                print(f"Added {len(new_msgs)} new messages.")

        except Exception as e:
            print(f"Update failed: {e}")

        time.sleep(CHECK_INTERVAL)



def identify_user(question):
    """
        This function identifies who the question is about. It uses user_map and cross-checks who the question is about. 
        This is important in ask_question() to filter out irrelevant questions where no information can be given.
        If no user can be identified, null is returned.
    """
    prompt = f"""
    You are an assistant that identifies which member a question refers to.
    Known members: {list(user_map.keys())}
    Question: "{question}"

    Respond ONLY in valid JSON as:
    {{
        "user_name": "Alice"
    }}
    If you are not sure about the user or who the message is about, respond with:
    {{
        "user_name": null
    }}
    """
    try:
        response = llm.invoke(prompt)
        data = json.loads(response.content)
        return data.get("user_name")
    except Exception as e:
        print(f"identify_user() error: {e}")
        return None


@app.route("/ask", methods=["POST"])
def ask_question():
    """
        This method handles the main logic for gathering the relevant information and returning a response back. It first checks for a question and then a user. 
        The user is filtered in the vector DB to get the most sementically-relevant chunks(k=4 but this parameter can be adjusted) along with corresponding meta-data such as dates on when the messages was logged.
        Once all documents are formatted, the query + context are fed into LLM to generate a response and return in desired format.
    """
    data = request.json or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    user_name = identify_user(question)
    if not user_name or user_name.lower() not in user_map:
        return jsonify({"error": "Couldn't determine which member the question refers to."}), 400

    user_id = user_map[user_name.lower()]
    relevant_docs = vector_db.similarity_search(question, k=4, filter={"user_id": user_id})

    if not relevant_docs:
        return jsonify({"answer": f"No messages found for {user_name}."})
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        raw_timestamp = doc.metadata.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(raw_timestamp.replace('+00:00', ''))
            formatted_date = dt.strftime("%B %d, %Y")
            formatted_datetime = dt.strftime("%B %d, %Y at %I:%M %p") 
        except:
            formatted_date = "unknown date"
            formatted_datetime = "unknown date"
        context_parts.append(
            f"Message {i} (Date: {formatted_date}):\n{doc.page_content}"
        )
    
    context = "\n\n".join(context_parts)

    prompt = f"""
    You are answering a question about {user_name}'s messages.
    Each message includes the date it was sent.

    Context:
    {context}

    Question:
    {question}

    Answer clearly and concisely. When mentioning dates, use the format shown in the context.
    """

    try:
        answer = llm.invoke(prompt).content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 500
if __name__ == "__main__":
    """
    Main-method. Vector DB is bootstrapped and threading is initialized to ensure the service is pulling the third-party API every hour.
    """
    bootstrap_vector_db()
    threading.Thread(target=update_vector_db, daemon=True).start()
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
