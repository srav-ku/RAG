import gradio as gr
import sys
sys.path.append("/content/RAG")

from app.pipeline.ingest_pipeline import ingest_document
from app.pipeline.query_pipeline import answer_question
from app.storage.db import init_db, list_documents


def handle_upload(file):
    if file is None:
        return "No file selected.", gr.update()

    with open(file, "rb") as f:
        file_bytes = f.read()

    filename = file.split("/")[-1]
    result = ingest_document(file_bytes, filename)

    if result["status"] == "processed":
        msg = f"✅ Successfully processed '{filename}'\nPages: {result['page_count']} | Chunks created: {result['chunk_count']}"
    elif result["status"] == "duplicate":
        msg = f"⚠️ {result['message']}"
    else:
        msg = f"❌ {result['message']}"

    # Refresh the document choices dropdown after every upload
    return msg, gr.update(choices=get_doc_choices())


def get_doc_choices():
    """Returns (label, value) pairs for the document selector, using filename
    as the visible label and document_id as the actual value passed to filtering."""
    docs = list_documents()
    return [(d["filename"], d["id"]) for d in docs]


def handle_chat(message, history, selected_doc_ids):
    """
    selected_doc_ids comes from the CheckboxGroup - a list of document_ids
    the user checked. Empty list means "search all documents" (no filter).
    """
    doc_filter = selected_doc_ids if selected_doc_ids else None
    result = answer_question(message, document_ids=doc_filter)
    response = result["answer"]
    if result["was_answered"]:
        response += f"\n\n**Sources:**\n{result['sources']}"
    return response


def list_documents_display():
    docs = list_documents()
    if not docs:
        return "No documents uploaded yet."
    lines = [f"- {d['filename']} | status: {d['status']} | pages: {d['page_count']}" for d in docs]
    return "\n".join(lines)


def refresh_doc_selector():
    """Used by a manual refresh button on the Chat tab, in case a document
    was uploaded in another tab/session without reloading the page."""
    return gr.update(choices=get_doc_choices())


with gr.Blocks(title="RAG Document Intelligence Platform") as demo:
    gr.Markdown("# RAG Document Intelligence Platform")

    with gr.Tab("Upload"):
        file_input = gr.File(label="Upload a PDF")
        upload_button = gr.Button("Process Document")
        upload_output = gr.Textbox(label="Status", lines=4)

    with gr.Tab("Chat"):
        with gr.Row():
            doc_selector = gr.CheckboxGroup(
                choices=get_doc_choices(),
                label="Search only these documents (leave empty to search all)"
            )
            refresh_selector_btn = gr.Button("Refresh document list", scale=0)

        chatbot = gr.ChatInterface(
            fn=handle_chat,
            additional_inputs=[doc_selector]
        )

    with gr.Tab("Documents"):
        refresh_button = gr.Button("Refresh List")
        docs_output = gr.Textbox(label="Ingested Documents", lines=10)
        refresh_button.click(fn=list_documents_display, inputs=None, outputs=docs_output)

    # Wire up events that touch components across tabs
    upload_button.click(fn=handle_upload, inputs=file_input, outputs=[upload_output, doc_selector])
    refresh_selector_btn.click(fn=refresh_doc_selector, inputs=None, outputs=doc_selector)


if __name__ == "__main__":
    init_db()
    demo.launch(share=True, debug=True)
