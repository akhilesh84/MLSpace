from langchain_openai import AzureChatOpenAI
import os
from pathlib import Path

DEPLOYMENT_NAME="gpt-4o"
os.environ["AZURE_OPENAI_API_KEY"] = "3jboEFeFNNQHjQJJHPptWoQBtXhKcO8Tj6M8RI96F00cVS7DNKMLJQQJ99BHAC5RqLJXJ3w3AAAAACOGWY36"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aif-customerchat-dev-eastus.cognitiveservices.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"

# LangChain LLM wrapper for GPT-4o deployed on Azure AI
def get_llm(provider: str = "azure"):
    if provider == "azure":
        return AzureChatOpenAI(
            deployment_name=DEPLOYMENT_NAME,
            temperature=0,
        )
    # elif provider == "ollama":
    #     from langchain_community.llms import Ollama
    #     return Ollama(model="openbmb/minicpm-o2.6:latest")
    else:
        raise ValueError(f"Unknown provider {provider}")

import argparse
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pdf2image import convert_from_path
import pytesseract

# ---------- Step 1: Define Schema ----------
class Item(BaseModel):
    item_name: str = Field(description="Name of the item")
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    nett_value: Optional[float] = None
    hs_code: Optional[str] = None
    page_number: Optional[int] = None

class Totals(BaseModel):
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    grand_total: Optional[float] = None

class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    purchase_order_number: Optional[str] = None
    currency: Optional[str] = None
    pages: List[int] = []
    items: List[Item] = []
    totals: Totals = Totals()

class InvoiceExtraction(BaseModel):
    invoices: List[Invoice] = []

# ---------- Step 2: LangChain Prompt ----------
parser = PydanticOutputParser(pydantic_object=InvoiceExtraction)

prompt = PromptTemplate(
    template="""
You are an intelligent document parser.
You will receive OCR text extracted from one or more invoice documents.
The document can be in a language other than English.
Invoices may span multiple contiguous pages. When one invoice ends, the next invoice begins.
Extract all data into structured JSON.

Rules:
- Do not skip invoices or items.
- Do not invent data. Use null if unreadable.
- Preserve page references for each invoice and item.
- Return JSON strictly matching the schema.

OCR text:
{ocr_text}

{format_instructions}
""",
    input_variables=["ocr_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# ---------- Step 3: OCR Pipeline ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Convert PDF to text per page using Tesseract"""
    pages = convert_from_path(pdf_path)
    all_text = []
    for i, page in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page)
        all_text.append(f"--- Page {i} ---\n{text}")
    ocr_text = "\n".join(all_text)

    path = Path(pdf_path)
    file_name = path.stem
    output_dir = path.parent
    output_file = output_dir / f"{file_name}_ocr.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(ocr_text)
    return ocr_text


# ---------- Step 4: LLM Wrapper ----------
def extract_invoices_from_text(ocr_text: str):
    #TODO: Implement logic to handle token truncation. Explore streaming.
    
    prompt_text = prompt.format(ocr_text=ocr_text)
    llm = get_llm(provider="azure")
    response = llm.invoke(prompt_text)

    # Try to extract plain text from different possible response shapes
    text = None
    try:
        # LangChain LLMResult style: response.generations -> list[list[Generation]]
        if hasattr(response, "generations"):
            gens = response.generations
            if gens and len(gens) > 0 and len(gens[0]) > 0:
                gen0 = gens[0][0]
                # different langchain versions expose .text or .message.content
                text = getattr(gen0, "text", None) or getattr(getattr(gen0, "message", None), "content", None)
        # Chat-style response: maybe has .content or is a string
        if text is None:
            if isinstance(response, str):
                text = response
            else:
                text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
    except Exception as e:
        text = str(response)

    # Debug: show the model output (first N chars)
    # print("=== LLM raw output (first 2000 chars) ===")
    # print(text[:2000])
    # print("=== end LLM output ===")

    # Now try to parse — wrap to capture and show pydantic errors
    try:
        return parser.parse(text)
    except Exception as e:
        # Print full exception so you can inspect validation errors
        import traceback
        print("Failed parsing LLM output into Pydantic model:")
        traceback.print_exc()
        # Re-raise to preserve original behavior if you want the script to fail
        raise


# ---------- Step 5: Full Workflow ----------
def process_pdf(pdf_path: str):
    print(f"Processing {pdf_path} ...")
    ocr_text = extract_text_from_pdf(pdf_path)
    result = extract_invoices_from_text(ocr_text)
    return result


# ---------- Step 6: CLI Entry ----------
if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(description="Extract structured JSON data from invoices in PDFs")
    parser_cli.add_argument("pdf_path", help="Path to the input PDF file")
    parser_cli.add_argument("--output", "-o", help="Path to save the output JSON", default=None)

    args = parser_cli.parse_args()

    result = process_pdf(args.pdf_path)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"✅ Extraction complete. JSON saved to {args.output}")
    else:
        print(result.model_dump_json(indent=2))

    # print(result)

    # ocr_text = extract_text_from_pdf(args.pdf_path)
    # print(f"Extracted OCR text from {args.pdf_path}:\n{ocr_text}...")
    