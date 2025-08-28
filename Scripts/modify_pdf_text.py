#!/usr/bin/env python3
"""Modify text inside a PDF file.

Supports two modes:
- Text-layer PDFs: uses PyMuPDF (fitz) to search, redact, and insert replacement text.
- Scanned/image PDFs: uses pdf2image + pytesseract + PIL to locate text boxes and paint replacements.

Usage examples:
  # Replace occurrences of 'Old Company' -> 'New Company'
  python Scripts/modify_pdf_text.py input.pdf output.pdf --replace "Old Company=New Company"

  # Use a JSON map file with replacements {"Old":"New", ...}
  python Scripts/modify_pdf_text.py input.pdf output.pdf --map-file replacements.json

  # Force OCR mode (useful for scanned PDFs)
  python Scripts/modify_pdf_text.py input.pdf output.pdf --ocr --dpi 300

Notes:
- This is best-effort. Complex PDFs (flowed text, columns, variable fonts) may need manual adjustment.
- Install required packages: pip install pymupdf pdf2image pytesseract pillow
  On macOS also: brew install tesseract poppler
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    fitz = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


def parse_replacements(pairs: List[str]) -> Dict[str, str]:
    """Parse list of 'old=new' strings into a dict."""
    result: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Replacement must be in the form old=new, got: '{p}'")
        old, new = p.split("=", 1)
        result[old] = new
    return result


def modify_text_layer_pdf(pdf_path: str, output_path: str, replacements: Dict[str, str], case_sensitive: bool = True) -> None:
    """Modify PDFs that contain a selectable text layer using PyMuPDF.

    Approach:
    - For each replacement, search for the text on each page using page.search_for()
    - For each found rect, add a redact annotation, apply redactions (removes original text visually),
      then insert replacement text into the same rect using insert_textbox.

    This preserves the rest of the page content (images, layout).
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for text-layer editing. Install with 'pip install pymupdf'.")

    doc = fitz.open(pdf_path)
    print(f"Opened PDF with {doc.page_count} pages")

    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        text_page = page.get_textpage()
        page_text = page.get_text("text")

        # For each replacement, find all occurrences
        for old, new in replacements.items():
            search_term = old if case_sensitive else old.lower()

            # We'll iterate text spans to capture font and size information.
            rects = []  # type: List[Tuple[fitz.Rect, dict]]  # store rect + span metadata
            try:
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            match = False
                            if case_sensitive:
                                match = search_term in span_text
                            else:
                                match = search_term in span_text.lower()
                            if match:
                                bbox = fitz.Rect(span.get("bbox", [0, 0, 0, 0]))
                                rects.append((bbox, span))
            except Exception:
                # fallback to word-level search if dict extraction fails
                words = page.get_text("words")  # list of tuples (x0, y0, x1, y1, "word", block_no, line_no, word_no)
                for w in words:
                    wtext = w[4]
                    if case_sensitive:
                        ok = (search_term == wtext)
                    else:
                        ok = (search_term == wtext.lower())
                    if ok:
                        rects.append((fitz.Rect(w[0], w[1], w[2], w[3]), {"text": wtext, "font": None, "size": None}))

            if not rects:
                # Not found on this page
                continue

            # Helper: map PDF/embedded font names to core font aliases used by PyMuPDF
            def map_font_name(pdf_font_name: str | None) -> str:
                if not pdf_font_name:
                    return "helv"
                name = pdf_font_name.lower()
                if "helv" in name or "arial" in name or "sans" in name or "dejavu" in name:
                    return "helv"
                if "times" in name or "timesnewroman" in name or "times-roman" in name:
                    return "times"
                if "courier" in name or "mono" in name:
                    return "cour"
                return "helv"

            # Replace each found span rect preserving font size where possible
            for bbox, span in rects:
                try:
                    # paint white rectangle over original span area
                    page.draw_rect(bbox, color=(1, 1, 1), fill=(1, 1, 1))

                    span_text = span.get("text", "") if isinstance(span, dict) else ""
                    span_size = span.get("size") if isinstance(span, dict) else None
                    span_font = span.get("font") if isinstance(span, dict) else None

                    # compute replacement for the span text (replace all occurrences within span)
                    if case_sensitive:
                        replaced_text = span_text.replace(old, new)
                    else:
                        # perform case-insensitive replacement preserving original casing minimally
                        replaced_text = span_text.replace(span_text, span_text)
                        try:
                            # naive approach: replace using lowercased find
                            idx = span_text.lower().find(search_term)
                            if idx >= 0:
                                replaced_text = span_text[:idx] + new + span_text[idx+len(search_term):]
                        except Exception:
                            replaced_text = span_text

                    # determine fontsize
                    if span_size and isinstance(span_size, (int, float)):
                        fontsize = int(span_size)
                    else:
                        fontsize = max(6, int(bbox.height * 0.8))

                    fontname = map_font_name(span_font)

                    # shrink fontsize if replaced text too long to fit bbox
                    if len(replaced_text) > 0:
                        est_char_width = fontsize * 0.6
                        max_chars = max(1, int(bbox.width / est_char_width))
                        if len(replaced_text) > max_chars:
                            fontsize = max(6, int(fontsize * (max_chars / len(replaced_text))))

                    page.insert_textbox(bbox, replaced_text, fontsize=fontsize, fontname=fontname, align=0, color=(0, 0, 0))
                except Exception as e:
                    print(f"Replacement insert failed for bbox {bbox}: {e}")

    doc.save(output_path)
    doc.close()
    print(f"Saved modified PDF to {output_path}")


def modify_scanned_pdf(pdf_path: str, output_path: str, replacements: Dict[str, str], dpi: int = 300, case_sensitive: bool = True) -> None:
    """Modify scanned PDF by converting pages to images, performing OCR to find words, and painting replacements.

    Approach:
    - convert PDF pages to PIL images using pdf2image
    - run pytesseract.image_to_data to get bounding boxes for words
    - for target words, draw a white rectangle over the original box and write replacement text
    - save all modified images back to a multi-page PDF
    """
    if convert_from_path is None or pytesseract is None or Image is None:
        raise RuntimeError("pdf2image, pytesseract and Pillow are required for OCR-based editing. Install with 'pip install pdf2image pytesseract pillow'.")

    images = convert_from_path(pdf_path, dpi=dpi)
    out_images = []
    print(f"Converted PDF to {len(images)} images at {dpi} DPI")

    for page_index, img in enumerate(images, start=1):
        print(f"Processing page {page_index}/{len(images)}")
        img_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)

        # Use pytesseract to get word-level boxes
        data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)
        n_boxes = len(data["text"])

        # iterate through words and match replacements
        for i in range(n_boxes):
            word = data["text"][i]
            if not word or word.strip() == "":
                continue

            compare_word = word if case_sensitive else word.lower()
            for old, new in replacements.items():
                target = old if case_sensitive else old.lower()
                if compare_word == target:
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    # cover original text with white rectangle
                    draw.rectangle(((x, y), (x + w, y + h)), fill=(255, 255, 255))

                    # choose a font size close to the box height
                    fontsize = max(8, int(h * 0.9))
                    try:
                        fnt = ImageFont.truetype("DejaVuSans.ttf", fontsize)
                    except Exception:
                        fnt = ImageFont.load_default()

                    # write replacement text
                    draw.text((x, y), new, fill=(0, 0, 0), font=fnt)

        out_images.append(img_rgb)

    # Save modified images to PDF
    if not out_images:
        raise RuntimeError("No images generated from PDF; cannot save output")

    out_images[0].save(output_path, save_all=True, append_images=out_images[1:])
    print(f"Saved modified scanned PDF to {output_path}")


def load_map_file(map_file: str) -> Dict[str, str]:
    with open(map_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Map file must contain a JSON object mapping old->new strings")
    return {str(k): str(v) for k, v in data.items()}


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Modify text in a PDF (text-layer or scanned)")
    p.add_argument("input", help="Input PDF path")
    p.add_argument("output", help="Output PDF path")
    p.add_argument("--replace", "-r", action="append", default=[], help="Replacement pair in the form old=new. Can be passed multiple times.")
    p.add_argument("--map-file", "-m", help="JSON file containing a map of replacements {\"old\": \"new\"}")
    p.add_argument("--ocr", action="store_true", help="Force OCR-based editing (useful for scanned PDFs)")
    p.add_argument("--dpi", type=int, default=300, help="DPI for pdf2image when in OCR mode")
    p.add_argument("--case-insensitive", action="store_true", help="Perform case-insensitive matching")

    args = p.parse_args(argv)

    replacements: Dict[str, str] = {}
    if args.map_file:
        replacements.update(load_map_file(args.map_file))
    if args.replace:
        replacements.update(parse_replacements(args.replace))

    if not replacements:
        print("No replacements provided. Use --replace or --map-file to supply replacements.")
        return 2

    case_sensitive = not args.case_insensitive

    # Choose mode
    if args.ocr:
        print("Running in OCR (scanned) mode")
        modify_scanned_pdf(args.input, args.output, replacements, dpi=args.dpi, case_sensitive=case_sensitive)
        return 0

    # Try text-layer mode first; if it fails or finds no occurrences, optionally fallback to OCR
    try:
        if fitz is None:
            raise RuntimeError("PyMuPDF not available")

        # quick check: try opening and searching for the first key
        doc = fitz.open(args.input)
        found_any = False
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            for old in replacements.keys():
                if case_sensitive:
                    hits = page.search_for(old)
                else:
                    # simple case-insensitive check in extracted text
                    page_text = page.get_text("text")
                    hits = [1] if old.lower() in page_text.lower() else []
                if hits:
                    found_any = True
                    break
            if found_any:
                break
        doc.close()

        if found_any:
            print("Found text-layer matches; using PyMuPDF editing flow")
            modify_text_layer_pdf(args.input, args.output, replacements, case_sensitive=case_sensitive)
            return 0
        else:
            print("No text-layer matches found. Consider running with --ocr to edit a scanned PDF.")
            return 3

    except Exception as e:
        print(f"Text-layer editing failed: {e}")
        print("You can retry with --ocr to run OCR-based editing (requires pytesseract and pdf2image)")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
