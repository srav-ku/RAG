import os
import sys
sys.path.append("/content/RAG")

from app.config import CONFIG
if CONFIG.use_drive_model_cache:
    os.makedirs(CONFIG.model_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = CONFIG.model_cache_dir
    print(f"[Cache] Using Drive model cache: {CONFIG.model_cache_dir}")

import gradio as gr
from app.pipeline.ingest_pipeline import ingest_document
from app.pipeline.query_pipeline import answer_question
from app.pipeline.delete_pipeline import delete_document
from app.storage.db import init_db, list_documents


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Override Gradio's own theme variables directly - robust against dark mode,
   avoids the class-specificity war we hit last time. */
:root, .dark {
    --body-background-fill: #FAFAFA !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F5F5F7 !important;
    --border-color-primary: #E5E5E7 !important;
    --body-text-color: #1D1D1F !important;
    --body-text-color-subdued: #6E6E73 !important;
    --button-primary-background-fill: #3B5BFB !important;
    --button-primary-background-fill-hover: #2E48D6 !important;
    --button-primary-text-color: #FFFFFF !important;
    --button-secondary-background-fill: #FFFFFF !important;
    --button-secondary-border-color: #E5E5E7 !important;
    --input-background-fill: #FFFFFF !important;
    --block-background-fill: #FFFFFF !important;
    --block-border-color: #E5E5E7 !important;
    --panel-background-fill: #FFFFFF !important;
    --color-accent: #3B5BFB !important;
    --link-text-color: #3B5BFB !important;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
.gradio-container { background: #FAFAFA !important; max-width: 1280px !important; }

/* Sidebar panel */
#sidebar {
    background: #FFFFFF;
    border-right: 1px solid #E5E5E7;
    padding: 24px 16px;
    min-height: 640px;
}
#sidebar-title { font-size: 20px; font-weight: 700; color: #1D1D1F; letter-spacing: -0.02em; margin-bottom: 4px; }
#sidebar-subtitle { font-size: 13px; color: #6E6E73; margin-bottom: 28px; }
.stat-card { background: #F5F5F7; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }
.stat-number { font-size: 22px; font-weight: 700; color: #1D1D1F; }
.stat-label { font-size: 12px; color: #6E6E73; font-weight: 500; }

/* Panel headers */
.panel-header { font-size: 15px; font-weight: 600; color: #1D1D1F; margin-bottom: 12px; }

/* Status badges rendered inside the dataframe via HTML */
.status-processed { color: #1F9D55; font-weight: 600; }
.status-pending { color: #B7791F; font-weight: 600; }
.status-failed { color: #C53030; font-weight: 600; }

/* Chat area */
.message.user { background: #3B5BFB !important; color: white !important; border-radius: 16px 16px 4px 16px !important; }
.message.bot { background: #FFFFFF !important; border: 1px solid #E5E5E7 !important; border-radius: 16px 16px 16px 4px !important; animation: fadeIn 0.25s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

#disclaimer { font-size: 11px; color: #6E6E73; text-align: center; margin-top: 6px; }

label { color: #6E6E73 !important; font-weight: 500 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.03em; }
"""


def get_doc_choices():
    docs = list_documents()
    return [(d["filename"], d["id"]) for d in docs]


def get_stats_html():
    docs = list_documents()
    total_pages = sum(d["page_count"] or 0 for d in docs)
    return f"""
    <div class='stat-card'><div class='stat-number'>{len(docs)}</div><div class='stat-label'>Documents indexed</div></div>
    <div class='stat-card'><div class='stat-number'>{total_pages}</div><div class='stat-label'>Total pages processed</div></div>
    """


def get_library_rows():
    docs = list_documents()
    if not docs:
        return [["No documents yet", "—", "—"]]
    badge_map = {"processed": "🟢 Processed", "pending": "🟡 Pending", "failed": "🔴 Failed"}
    return [[d["filename"], badge_map.get(d["status"], d["status"]), d["page_count"] or "—"] for d in docs]


def handle_upload(file, progress=gr.Progress()):
    if file is None:
        return "No file selected.", gr.update(), gr.update(), gr.update()

    supported = (".pdf", ".txt", ".md", ".docx")
    if not file.lower().endswith(supported):
        return f"Unsupported file type. Supported: {', '.join(supported)}", gr.update(), gr.update(), gr.update()

    progress(0.1, desc="Reading file...")
    with open(file, "rb") as f:
        file_bytes = f.read()
    filename = file.split("/")[-1]

    progress(0.3, desc="Parsing, chunking, and embedding...")
    result = ingest_document(file_bytes, filename)
    progress(1.0, desc="Done")

    if result["status"] == "processed":
        msg = f"Processed '{filename}' — {result['page_count']} pages, {result['chunk_count']} chunks."
    else:
        msg = result["message"]

    return msg, gr.update(value=get_library_rows()), gr.update(choices=get_doc_choices()), gr.update(value=get_stats_html())


def handle_chat(message, history, selected_doc_ids):
    doc_filter = selected_doc_ids if selected_doc_ids else None
    result = answer_question(message, document_ids=doc_filter)
    response = result["answer"]
    if result["was_answered"]:
        pills = result["sources"].replace("- ", "").replace("\n", " · ")
        response += f"\n\n**Source:** {pills}"
    return response


def handle_delete(doc_id):
    if not doc_id:
        return "Select a document to delete first.", gr.update(), gr.update(), gr.update()
    result = delete_document(doc_id)
    new_choices = get_doc_choices()
    return result["message"], gr.update(value=get_library_rows()), gr.update(choices=new_choices, value=None), gr.update(value=get_stats_html())


with gr.Blocks(title="Document Intelligence", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
    with gr.Row():
        # LEFT PANEL - sidebar
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.HTML("<div id='sidebar-title'>Document Intelligence</div><div id='sidebar-subtitle'>Grounded Q&A over your documents</div>")
            stats_display = gr.HTML(get_stats_html())

            gr.Markdown("**Add a document**")
            file_input = gr.File(label="", show_label=False, height=100)
            upload_button = gr.Button("Process document", variant="primary", size="sm")
            upload_output = gr.Textbox(label="", show_label=False, lines=2)

        # MIDDLE PANEL - knowledge base / library
        with gr.Column(scale=2):
            gr.HTML("<div class='panel-header'>Knowledge Base</div>")
            library_table = gr.Dataframe(
                headers=["Document", "Status", "Pages"],
                value=get_library_rows(),
                interactive=False,
                wrap=True
            )
            with gr.Row():
                delete_selector = gr.Dropdown(choices=get_doc_choices(), label="Remove a document", scale=3)
                delete_button = gr.Button("Delete", variant="stop", size="sm", scale=1)
            delete_output = gr.Textbox(label="", show_label=False, lines=1)

        # RIGHT PANEL - chat
        with gr.Column(scale=2):
            gr.HTML("<div class='panel-header'>Ask</div>")
            doc_selector = gr.CheckboxGroup(
                choices=get_doc_choices(),
                label="Scope (leave empty to search all documents)"
            )
            chatbot = gr.ChatInterface(fn=handle_chat, additional_inputs=[doc_selector])
            gr.HTML("<div id='disclaimer'>Answers are grounded in your uploaded documents and may be incomplete.</div>")

    upload_button.click(
        fn=handle_upload,
        inputs=file_input,
        outputs=[upload_output, library_table, doc_selector, stats_display]
    )
    delete_button.click(
        fn=handle_delete,
        inputs=delete_selector,
        outputs=[delete_output, library_table, delete_selector, stats_display]
    )
    delete_button.click(fn=lambda: gr.update(choices=get_doc_choices()), inputs=None, outputs=doc_selector)


if __name__ == "__main__":
    init_db()
    demo.launch(share=True, debug=True)
