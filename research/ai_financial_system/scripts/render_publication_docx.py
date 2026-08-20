from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.document import Document as DocumentObject
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
FIXED_PACKAGE_TIME = (2026, 7, 26, 0, 0, 0)
FIXED_CORE_TIME = dt.datetime(2026, 7, 26, tzinfo=dt.UTC)

NAVY = "17365D"
BLUE = "275D84"
SLATE = "536B7A"
LIGHT_BLUE = "EAF1F7"
LIGHT_GREEN = "E9F4EE"
LIGHT_AMBER = "FBF1E6"
LIGHT_GRAY = "F4F6F8"
GRID = "C9D3DC"
WHITE = "FFFFFF"
TEXT = RGBColor(31, 43, 55)
MUTED = RGBColor(83, 107, 122)


def set_cell_shading(cell: object, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()  # type: ignore[attr-defined]
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_border(
    cell: object,
    *,
    left: tuple[str, str] | None = None,
    all_color: str | None = None,
) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()  # type: ignore[attr-defined]
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge == "left" and left:
            color, size = left
            style = "single"
        elif all_color:
            color, size, style = all_color, "4", "single"
        else:
            continue
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), style)
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_repeat_table_header(row: object) -> None:
    tr_pr = row._tr.get_or_add_trPr()  # type: ignore[attr-defined]
    element = OxmlElement("w:tblHeader")
    element.set(qn("w:val"), "true")
    tr_pr.append(element)


def prevent_row_split(row: object) -> None:
    tr_pr = row._tr.get_or_add_trPr()  # type: ignore[attr-defined]
    element = OxmlElement("w:cantSplit")
    tr_pr.append(element)


def add_page_number(paragraph: object) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for child in (begin, instruction, separate, text, end):
        run._r.append(child)


def add_hyperlink(paragraph: object, text: str, url: str) -> None:
    relationship = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.extend((color, underline))
    run.append(properties)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def configure_document(document: DocumentObject, title: str) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.74)
    section.right_margin = Inches(0.74)
    section.header_distance = Inches(0.28)
    section.footer_distance = Inches(0.3)

    normal = document.styles["Normal"]
    normal.font.name = "Iowan Old Style"
    normal.font.size = Pt(10.2)
    normal.font.color.rgb = TEXT
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Iowan Old Style")
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.18
    normal.paragraph_format.widow_control = True

    for name, size, color, before, after in (
        ("Title", 25, NAVY, 0, 15),
        ("Heading 1", 18, NAVY, 18, 7),
        ("Heading 2", 14, NAVY, 15, 5),
        ("Heading 3", 10.5, BLUE, 11, 4),
    ):
        style = document.styles[name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Aptos Display")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for style_name, font_size, color in (
        ("Research Kicker", 8.3, BLUE),
        ("Research Subtitle", 13.2, SLATE),
        ("Research Meta", 8.5, SLATE),
        ("Research Source", 8.1, SLATE),
        ("Research Tag", 7.4, BLUE),
        ("Research Caption", 8.2, SLATE),
    ):
        if style_name not in document.styles:
            style = document.styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
        else:
            style = document.styles[style_name]
        style.font.name = "Aptos"
        style.font.size = Pt(font_size)
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Aptos")
        style.paragraph_format.space_after = Pt(6)

    document.styles["Research Kicker"].font.bold = True
    document.styles["Research Kicker"].font.all_caps = True
    document.styles["Research Subtitle"].font.name = "Iowan Old Style"
    document.styles["Research Subtitle"].font.italic = True
    document.styles["Research Subtitle"].paragraph_format.space_after = Pt(18)
    document.styles["Research Meta"].paragraph_format.left_indent = Inches(0.05)

    for list_style in ("List Bullet", "List Number"):
        style = document.styles[list_style]
        style.font.name = "Iowan Old Style"
        style.font.size = Pt(10)
        style.paragraph_format.space_after = Pt(3)
        style.paragraph_format.left_indent = Inches(0.28)
        style.paragraph_format.first_line_indent = Inches(-0.16)

    header = section.header.paragraphs[0]
    header.text = title
    header.style = document.styles["Research Source"]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    header.paragraph_format.space_after = Pt(0)
    footer = section.footer.paragraphs[0]
    footer.style = document.styles["Research Source"]
    add_page_number(footer)

    document.core_properties.title = title
    document.core_properties.subject = "Forensic research monograph"
    document.core_properties.author = "Trade-bot AI Financial System Research"
    document.core_properties.keywords = (
        "AI finance; capital; contracts; dependencies; control; risk transmission"
    )
    document.core_properties.created = FIXED_CORE_TIME
    document.core_properties.modified = FIXED_CORE_TIME
    document.core_properties.last_modified_by = "Trade-bot publication pipeline"


def append_inline(paragraph: object, node: Tag | NavigableString) -> None:
    if isinstance(node, NavigableString):
        # HTML source is line-wrapped for review. Those source newlines are not
        # semantic and must not become forced Word line breaks. Explicit <br>
        # elements are handled below and remain the only inline hard breaks.
        text = re.sub(r"\s+", " ", str(node))
        if text:
            paragraph.add_run(text)
        return
    if not isinstance(node, Tag):
        return
    if node.name == "br":
        paragraph.add_run().add_break()
        return
    if node.name == "a" and node.get("href"):
        add_hyperlink(paragraph, node.get_text(" ", strip=False), str(node["href"]))
        return
    if node.name in {"strong", "b"}:
        run = paragraph.add_run(node.get_text(" ", strip=False))
        run.bold = True
        return
    if node.name in {"em", "i"}:
        run = paragraph.add_run(node.get_text(" ", strip=False))
        run.italic = True
        return
    if node.name == "code":
        run = paragraph.add_run(node.get_text("", strip=False))
        run.font.name = "Menlo"
        run.font.size = Pt(8.3)
        run.font.color.rgb = RGBColor.from_string(NAVY)
        return
    if node.name == "span":
        run = paragraph.add_run(node.get_text(" ", strip=False))
        classes = set(node.get("class", []))
        if "source" in classes:
            run.font.name = "Aptos"
            run.font.size = Pt(8.1)
            run.font.color.rgb = MUTED
        elif "tag" in classes:
            run.font.name = "Aptos"
            run.font.size = Pt(7.4)
            run.font.bold = True
            run.font.color.rgb = RGBColor.from_string(BLUE)
        elif classes & {"grade-b", "grade-c", "grade-d"}:
            run.font.name = "Aptos"
            run.font.size = Pt(8.1)
            run.font.bold = True
            if "grade-b" in classes:
                run.font.color.rgb = RGBColor.from_string("2F6B4F")
            elif "grade-c" in classes:
                run.font.color.rgb = RGBColor.from_string("9C5B16")
            else:
                run.font.color.rgb = RGBColor.from_string("9B2C2C")
        return
    for child in node.children:
        append_inline(paragraph, child)


def add_paragraph_from_tag(
    document: DocumentObject,
    tag: Tag,
    *,
    style: str | None = None,
    container: object | None = None,
) -> object:
    parent = container if container is not None else document
    paragraph = parent.add_paragraph(style=style)
    for child in tag.children:
        append_inline(paragraph, child)
    return paragraph


def render_callout(document: DocumentObject, tag: Tag) -> None:
    classes = set(tag.get("class", []))
    if "pull" in classes:
        fill, border = "EAF2F7", BLUE
    elif "warning" in classes:
        fill, border = LIGHT_AMBER, "B36A2E"
    elif "notice" in classes:
        fill, border = LIGHT_GREEN, "3F7D5C"
    else:
        fill, border = LIGHT_BLUE, BLUE
    table = document.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    cell = table.cell(0, 0)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_shading(cell, fill)
    set_cell_border(cell, left=(border, "20"), all_color=fill)
    cell.margin_top = Inches(0.08)
    cell.margin_bottom = Inches(0.08)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    tag_nodes: list[Tag] = []
    for child in tag.children:
        if isinstance(child, Tag) and "tag" in set(child.get("class", [])):
            tag_nodes.append(child)
            continue
        append_inline(paragraph, child)
    if "pull" in classes:
        paragraph.paragraph_format.space_before = Pt(3)
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.keep_together = True
        for run in paragraph.runs:
            run.italic = True
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor.from_string(NAVY)
    if tag_nodes:
        tag_paragraph = cell.add_paragraph(style="Research Tag")
        tag_paragraph.paragraph_format.space_before = Pt(4)
        tag_paragraph.paragraph_format.space_after = Pt(0)
        for index, tag_node in enumerate(tag_nodes):
            if index:
                tag_paragraph.add_run("  ·  ")
            run = tag_paragraph.add_run(tag_node.get_text(" ", strip=True))
            run.bold = True
            run.font.color.rgb = RGBColor.from_string(border)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def table_matrix(table_tag: Tag) -> list[list[Tag]]:
    rows: list[list[Tag]] = []
    for row in table_tag.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if cells:
            rows.append(cells)
    return rows


def render_table(document: DocumentObject, tag: Tag) -> None:
    caption = tag.find("caption", recursive=False)
    if caption is not None:
        paragraph = add_paragraph_from_tag(document, caption, style="Caption")
        paragraph.paragraph_format.keep_with_next = True
    matrix = table_matrix(tag)
    if not matrix:
        return
    width = max(len(row) for row in matrix)
    table = document.add_table(rows=len(matrix), cols=width)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for row_index, source_row in enumerate(matrix):
        output_row = table.rows[row_index]
        prevent_row_split(output_row)
        if row_index == 0 and source_row[0].name == "th":
            set_repeat_table_header(output_row)
        for column_index in range(width):
            cell = output_row.cells[column_index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            source = source_row[column_index] if column_index < len(source_row) else None
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.03
            if source is not None:
                for child in source.children:
                    append_inline(paragraph, child)
            for run in paragraph.runs:
                run.font.name = "Aptos"
                run.font.size = Pt(7.5)
            if row_index == 0 and source is not None and source.name == "th":
                set_cell_shading(cell, NAVY)
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor.from_string(WHITE)
            elif row_index % 2 == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            set_cell_border(cell, all_color=GRID)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def render_list(document: DocumentObject, tag: Tag) -> None:
    style = "List Number" if tag.name == "ol" else "List Bullet"
    for item in tag.find_all("li", recursive=False):
        paragraph = document.add_paragraph(style=style)
        for child in item.children:
            if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                continue
            append_inline(paragraph, child)
        for nested in item.find_all(["ul", "ol"], recursive=False):
            render_list(document, nested)


def render_title_page(document: DocumentObject, section: Tag) -> None:
    for child in section.children:
        if not isinstance(child, Tag):
            continue
        classes = set(child.get("class", []))
        if child.name == "h1":
            paragraph = document.add_paragraph(style="Title")
            for node in child.children:
                append_inline(paragraph, node)
        elif "kicker" in classes:
            add_paragraph_from_tag(document, child, style="Research Kicker")
        elif "subtitle" in classes:
            add_paragraph_from_tag(document, child, style="Research Subtitle")
        elif "meta" in classes:
            paragraph = add_paragraph_from_tag(document, child, style="Research Meta")
            top_border = OxmlElement("w:pBdr")
            border = OxmlElement("w:top")
            border.set(qn("w:val"), "single")
            border.set(qn("w:sz"), "8")
            border.set(qn("w:color"), GRID)
            top_border.append(border)
            paragraph._p.get_or_add_pPr().append(top_border)


def render_body(document: DocumentObject, body: Tag) -> None:
    title_page = body.find("section", class_="title-page", recursive=False)
    if title_page is not None:
        render_title_page(document, title_page)
    pending_page_break = title_page is not None
    for child in body.children:
        if not isinstance(child, Tag) or child is title_page:
            continue
        classes = set(child.get("class", []))
        if "page-break" in classes:
            pending_page_break = True
        paragraph_count = len(document.paragraphs)
        if child.name == "h1":
            add_paragraph_from_tag(document, child, style="Heading 1")
        elif child.name == "h2":
            add_paragraph_from_tag(document, child, style="Heading 2")
        elif child.name in {"h3", "h4"}:
            style = "Heading 3" if child.name == "h3" else "Heading 4"
            paragraph = add_paragraph_from_tag(document, child, style=style)
            for run in paragraph.runs:
                run.font.all_caps = True
        elif child.name == "p":
            paragraph = add_paragraph_from_tag(document, child)
            if "keep-together" in classes:
                paragraph.paragraph_format.keep_together = True
        elif child.name in {"ul", "ol"}:
            render_list(document, child)
        elif child.name == "table":
            render_table(document, child)
        elif child.name == "blockquote":
            paragraph = add_paragraph_from_tag(document, child)
            paragraph.paragraph_format.left_indent = Inches(0.28)
            paragraph.paragraph_format.right_indent = Inches(0.18)
            for run in paragraph.runs:
                run.italic = True
                run.font.color.rgb = MUTED
        elif child.name == "figcaption":
            paragraph = add_paragraph_from_tag(document, child, style="Caption")
            paragraph.paragraph_format.keep_with_next = True
        elif child.name == "figure":
            render_body(document, child)
        elif child.name == "div" and classes & {
            "finding",
            "notice",
            "warning",
            "boundary",
            "inference",
            "scenario",
            "hypothesis",
            "pull",
        }:
            render_callout(document, child)
        elif child.name == "div":
            block_children = [
                node
                for node in child.children
                if isinstance(node, Tag)
                and node.name
                in {
                    "section",
                    "div",
                    "figure",
                    "table",
                    "p",
                    "ul",
                    "ol",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                }
            ]
            if block_children:
                render_body(document, child)
            elif child.get_text(" ", strip=True):
                add_paragraph_from_tag(document, child)
        elif child.name == "hr":
            document.add_paragraph("•  •  •").alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif child.name in {"section", "header", "main", "article", "footer"}:
            render_body(document, child)
        if pending_page_break and len(document.paragraphs) > paragraph_count:
            document.paragraphs[paragraph_count].paragraph_format.page_break_before = True
            pending_page_break = False


def normalize_docx_package(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="docx-normalize-") as temporary:
        normalized = Path(temporary) / path.name
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
            normalized, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            for name in sorted(source.namelist()):
                data = source.read(name)
                info = zipfile.ZipInfo(name, FIXED_PACKAGE_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                target.writestr(info, data)
        shutil.copyfile(normalized, path)


def render(source: Path, output: Path) -> None:
    soup = BeautifulSoup(source.read_text(encoding="utf-8"), "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else source.stem
    if soup.body is None:
        raise ValueError(f"HTML has no body: {source}")
    document = Document()
    configure_document(document, title)
    render_body(document, soup.body)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    normalize_docx_package(output)
    if output.stat().st_size == 0:
        raise RuntimeError(f"empty DOCX: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.chapter) == bool(args.source):
        raise SystemExit("provide exactly one of --chapter or --source")
    if args.chapter:
        chapter = str(args.chapter).zfill(2)
        source = PROJECT_ROOT / f"chapters/chapter_{chapter}/chapter.html"
        output = (
            args.output
            or REPO_ROOT
            / f"reports/ai_financial_system/chapters/chapter_{chapter}"
            / f"AI_Financial_System_Chapter_{chapter}.docx"
        )
    else:
        source = args.source
        if args.output is None:
            raise SystemExit("--output is required with --source")
        output = args.output
    render(source.resolve(), output.resolve())
    print(output.resolve())


if __name__ == "__main__":
    main()
