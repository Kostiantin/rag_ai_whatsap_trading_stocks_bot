from flask import Flask, request, jsonify
from twilio.rest import Client
from get_embedding_function import get_embedding_function
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import threading
import logging

# Load environment variables
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # Twilio Sandbox number

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

logging.basicConfig(level=logging.INFO)

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

app = Flask(__name__)

# Initialize DB & embeddings once
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = OllamaLLM(model="mistral")  # Use Mistral for generation

def query_rag(query_text: str) -> str:
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    return model.invoke(prompt)

def process_and_reply(to_number: str, incoming_msg: str):
    try:
        answer = query_rag(incoming_msg)
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=to_number,
            body=answer
        )
        logging.info(f"Sent answer to {to_number}")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=to_number,
            body=f"Error processing request: {e}"
        )

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")
    logging.info(f"Received message from {from_number}: {incoming_msg}")

    # Respond immediately to Twilio to avoid timeout
    threading.Thread(target=process_and_reply, args=(from_number, incoming_msg)).start()
    return jsonify({"status": "processing"}), 200

if __name__ == "__main__":
    logging.info("Starting WhatsApp RAG bot on port 5001")
    app.run(port=5001, host="0.0.0.0")
