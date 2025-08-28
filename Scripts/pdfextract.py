import fitz
import pytesseract
from PIL import Image
import io
from langchain_ollama import ChatOllama  # LangChain Ollama integration
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---- Step 1: PDF → Images per page ----
def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages.append((page_num + 1, img))  # page_number starts at 1
    return pages

# ---- Step 2: OCR with Tesseract ----
def ocr_pages(pages):
    ocr_results = []
    for page_number, img in pages:
        text = pytesseract.image_to_string(img)
        ocr_results.append({"page_number": page_number, "text": text})
    return ocr_results

# ---- Step 3: LLM Extraction via Ollama ----
def extract_data_with_llm(ocr_results):
    schema = """
    {
      "invoices": [
        {
          "invoice_number": "string",
          "customer_name": "string",
          "invoice_date": "string",
          "total_amount": "string",
          "address": "string",
          "start_page": "integer",
          "end_page": "integer",
          "items": [
            {
              "description": "string",
              "quantity": "string",
              "unit_price": "string",
              "total_price": "string",
              "page_number": "integer"
            }
          ]
        }
      ]
    }
    """

    combined_text = "\n".join(
        [f"---PAGE {r['page_number']}---\n{r['text']}" for r in ocr_results]
    )

    prompt_template = ChatPromptTemplate.from_template("""
    You are an information extraction assistant.
    The following text comes from a scanned multi-page document containing one or more invoices.
    Invoices always have their pages in contiguous order.
    A new invoice starts when a new invoice number appears.

    Each page is marked as: ---PAGE X---

    For each invoice:
    - Extract: invoice_number, customer_name, invoice_date, total_amount, address
    - start_page: first page number for the invoice
    - end_page: last page number for the invoice
    - items: each with description, HS code (tariff code) quantity, unit_price, total_price, and page_number

    Return JSON strictly in this schema:
    {schema}

    Document text:
    {doc_text}
    """)

    parser = JsonOutputParser()

    llm = ChatOllama(
        model="openbmb/minicpm-o2.6:latest",
        temperature=0,
        base_url="http://localhost:11434"
    )

    chain = prompt_template | llm | parser

    return chain.invoke({
        "schema": schema,
        "doc_text": combined_text
    })

# ---- Step 4: Putting it together ----
if __name__ == "__main__":
    pdf_path = "sample_invoice.pdf"

    # Convert PDF to images
    pages = pdf_to_images(pdf_path)

    # OCR each page
    ocr_results = ocr_pages(pages)

    # Extract structured data using Ollama model
    extracted_data = extract_data_with_llm(ocr_results)

    print(extracted_data)
