import gradio as gr
from gce.core.cc_surface.api import compute_verdict, RunBundle
from gce.exporters.one_pager import generate_pdf

def compute_fn(json_str):
    bundle = RunBundle.model_validate_json(json_str)
    verdict = compute_verdict(bundle)
    return gr.update(value=verdict.label, variant="primary" if verdict.label == "Constructive" else "warning" if "Independent" else "danger"), verdict.recommendation, "\n".join(verdict.next_tests)

with gr.Blocks() as demo:
    json_input = gr.JSON(label="Paste Run Bundle")
    compute_btn = gr.Button("Compute")
    label_chip = gr.Textbox(label="Verdict Label")
    rec = gr.Textbox(label="Recommendation")
    next_tests = gr.Textbox(label="Next Tests")
    export_json = gr.Button("Export JSON")
    export_pdf = gr.Button("Download PDF")
    compute_btn.click(compute_fn, inputs=json_input, outputs=[label_chip, rec, next_tests])
    export_pdf.click(generate_pdf, inputs=json_input, outputs=gr.File(label="PDF"))

demo.launch()
