import os
import re
import json
import html
import hashlib
import textwrap
from pathlib import Path

import fitz
import docx
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = Path("qwen_dataset")
OUTPUT_DIR = Path("qwen_dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
OUTPUT_FILE = OUTPUT_DIR / "qwen2_train_data.jsonl"

PROMPT_TEXT = (
    "Convert the following document to markdown.\n"
    "Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html.\n\n"
    "RULES:\n"
    "  - You must include all information on the page. Do not exclude headers, footers, or subtext.\n"
    "  - Return tables in an HTML format.\n"
    "  - Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.\n"
    "  - Prefer using ☐ and ☑ for check boxes."
)


def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    safe = re.sub(r"\s+", "_", safe)
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'document'}_{digest}"


def html_table_from_rows(rows):
    if not rows:
        return ""
    html_rows = []
    for row in rows:
        cells = [html.escape(cell.strip()) for cell in row]
        html_rows.append("    <tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table>\n" + "\n".join(html_rows) + "\n</table>"


def detect_table_rows(lines):
    rows = []
    delimiter = None
    for line in lines:
        if "\t" in line:
            delimiter = "\t"
            break
        if "|" in line and line.strip().count("|") >= 2:
            delimiter = "|"
            break
        if re.search(r"\s{2,}", line):
            delimiter = None
    if delimiter is None:
        return None

    for line in lines:
        if delimiter == "|":
            cells = [cell.strip() for cell in line.strip().strip("|").split("|") if cell.strip()]
        else:
            cells = [cell.strip() for cell in line.split(delimiter) if cell.strip()]
        if len(cells) > 1:
            rows.append(cells)
    if len(rows) >= 2:
        return rows
    return None


def pdf_page_to_markdown(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return ""

    blocks = []
    current = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    markdown = []
    for block in blocks:
        if len(block) == 1 and block[0].isupper() and len(block[0]) < 120:
            markdown.append(f"## {block[0].strip()}")
            continue

        table = detect_table_rows(block)
        if table:
            markdown.append(html_table_from_rows(table))
            continue

        if all(re.match(r"^[\d\)\.]|^[•\-*+]", b.strip()) for b in block):
            for b in block:
                text_line = re.sub(r"^[\d\)\.\s]+", "", b).strip()
                markdown.append(f"- {text_line}")
            continue

        paragraph = " ".join(line.strip() for line in block)
        paragraph = paragraph.replace("\t", " ")
        markdown.append(paragraph)
    return "\n\n".join(markdown).strip()


def docx_table_to_html(table):
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(cells)
    return html_table_from_rows(rows)


def docx_to_markdown(doc: docx.document.Document) -> str:
    markdown = []
    for element in doc.element.body:
        if element.tag.endswith("}p"):
            paragraph = docx.text.paragraph.Paragraph(element, doc)
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name.lower() if paragraph.style else ""
            if "heading" in style:
                level = 1
                if "2" in style:
                    level = 2
                elif "3" in style:
                    level = 3
                markdown.append("#" * level + " " + text)
            elif paragraph._p.pPr is not None and paragraph._p.pPr.numPr is not None:
                markdown.append(f"- {text}")
            elif text.startswith(("•", "-", "*", "☐", "☑")):
                markdown.append(f"- {text.lstrip('•-* ').strip()}")
            else:
                markdown.append(text)
        elif element.tag.endswith("}tbl"):
            table = docx.table.Table(element, doc)
            markdown.append(docx_table_to_html(table))

    return "\n\n".join(markdown).strip()


def render_text_image(text: str, output_path: Path):
    width = 1400
    margin = 30
    font = ImageFont.load_default()
    lines = []
    wrapper = textwrap.TextWrapper(width=110)
    for paragraph in text.split("\n\n"):
        wrapped = wrapper.wrap(paragraph)
        if wrapped:
            lines.extend(wrapped)
            lines.append("")
    if lines and lines[-1] == "":
        lines.pop()

    temp_image = Image.new("RGB", (width, 100), "white")
    temp_draw = ImageDraw.Draw(temp_image)
    _, _, _, text_height = temp_draw.textbbox((0, 0), "A", font=font)
    line_height = text_height + 4
    height = max(1200, margin * 2 + line_height * len(lines) + 10)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height
    image.save(output_path)


PAGE_MARKER_RE = re.compile(r"^\s*={3,}\s*Page\s*(\d+)\s*={3,}\s*$", re.IGNORECASE)


def split_text_by_page_markers(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    sections = []
    current_lines = []
    current_page_num = None

    for line in lines:
        marker = PAGE_MARKER_RE.match(line)
        if marker:
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_page_num or len(sections) + 1, section_text))
                current_lines = []
            current_page_num = int(marker.group(1))
            continue
        current_lines.append(line)

    final_text = "\n".join(current_lines).strip()
    if final_text:
        sections.append((current_page_num or len(sections) + 1, final_text))

    return sections or [(1, text.strip())]


def process_pdf(file_path: Path, entries):
    doc = fitz.open(file_path)
    stem = sanitize_filename(file_path.stem)
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text")
        sections = split_text_by_page_markers(text)

        image_path = IMAGES_DIR / f"{stem}_page{page_idx+1}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pix.save(str(image_path))

        for section_idx, (section_num, section_text) in enumerate(sections, start=1):
            markdown = pdf_page_to_markdown(section_text)
            page_suffix = f"page{page_idx+1}" if len(sections) == 1 else f"page{page_idx+1}_{section_num}"
            markdown_path = MARKDOWN_DIR / f"{stem}_{page_suffix}.md"
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(markdown, encoding="utf-8")
            entries.append({
                "image": str(image_path.relative_to(OUTPUT_DIR).as_posix()),
                "prompt": PROMPT_TEXT,
                "output": markdown,
            })


def process_docx(file_path: Path, entries):
    document = docx.Document(file_path)
    stem = sanitize_filename(file_path.stem)
    markdown = docx_to_markdown(document)
    image_path = IMAGES_DIR / f"{stem}_page1.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    render_text_image(markdown or file_path.name, image_path)
    markdown_path = MARKDOWN_DIR / f"{stem}_page1.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    entries.append({
        "image": str(image_path.relative_to(OUTPUT_DIR).as_posix()),
        "prompt": PROMPT_TEXT,
        "output": markdown,
    })


def is_excluded_path(path: Path) -> bool:
    if not path.is_file():
        return True
    if path.suffix.lower() != ".pdf":
        return True
    if IMAGES_DIR in path.parents or MARKDOWN_DIR in path.parents:
        return True
    if (DATASET_DIR / "pdfs_for_compare") in path.parents:
        return True
    return False


def get_all_pdf_paths() -> list[Path]:
    return sorted([path for path in DATASET_DIR.rglob("*.pdf") if not is_excluded_path(path)])


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    MARKDOWN_DIR.mkdir(exist_ok=True)
    entries = []

    pdf_paths = get_all_pdf_paths()
    for path in pdf_paths:
        print(f"Processing PDF: {path.relative_to(DATASET_DIR)}")
        process_pdf(path, entries)

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for entry in entries:
            json.dump(entry, fh, ensure_ascii=False)
            fh.write("\n")

    print(f"Saved dataset with {len(entries)} page entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
