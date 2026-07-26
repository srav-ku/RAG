import gradio as gr
import sys
sys.path.append("/content/RAG")

from app.pipeline.ingest_pipeline import ingest_document
from app.pipeline.query_pipeline import answer_question
from app.storage.db import init_db, list_documents


def handle_upload(file):
    """
    Called when a user uploads a file in the Upload tab.
    Gradio gives us a filepath (str) for the uploaded temp file.
    """
    if file is None:
        return "No file selected."

    with open(file, "rb") as f:
        file_bytes = f.read()

    filename = file.split("/")[-1]  # just the filename, not the full temp path
    result = ingest_document(file_bytes, filename)

    if result["status"] == "processed":
        return f"✅ Successfully processed '{filename}'\nPages: {result['page_count']} | Chunks created: {result['chunk_count']}"
    elif result["status"] == "duplicate":
        return f"⚠️ {result['message']}"
    else:
        return f"❌ {result['message']}"


def handle_chat(message, history):
    """
    Called when a user sends a message in the Chat tab.
    `history` is Gradio's running conversation list - we don't use it in our
    logic yet (each question is answered independently), but Gradio requires
    this function signature for its Chatbot component.
    """
    result = answer_question(message)
    response = result["answer"]
    if result["was_answered"]:
        response += f"\n\n**Sources:**\n{result['sources']}"
    return response


def list_documents_display():
    """Returns a simple text summary of ingested documents, for the Documents tab."""
    docs = list_documents()
    if not docs:
        return "No documents uploaded yet."

    lines = []
    for d in docs:
        lines.append(f"- {d['filename']} | status: {d['status']} | pages: {d['page_count']}")
    return "\n".join(lines)


# Build the app
with gr.Blocks(title="RAG Document Intelligence Platform") as demo:
    gr.Markdown("# RAG Document Intelligence Platform")

    with gr.Tab("Upload"):
        file_input = gr.File(label="Upload a PDF")
        upload_button = gr.Button("Process Document")
        upload_output = gr.Textbox(label="Status", lines=4)
        upload_button.click(fn=handle_upload, inputs=file_input, outputs=upload_output)

    with gr.Tab("Chat"):
        chatbot = gr.ChatInterface(fn=handle_chat)

    with gr.Tab("Documents"):
        refresh_button = gr.Button("Refresh List")
        docs_output = gr.Textbox(label="Ingested Documents", lines=10)
        refresh_button.click(fn=list_documents_display, inputs=None, outputs=docs_output)


if __name__ == "__main__":
    init_db()
    demo.launch(share=True, debug=True)
