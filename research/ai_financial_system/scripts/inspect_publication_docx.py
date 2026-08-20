from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"

REQUIRED_ENTRIES = {
    "[Content_Types].xml",
    "_rels/.rels",
    "docProps/core.xml",
    "word/document.xml",
    "word/styles.xml",
    "word/header1.xml",
    "word/footer1.xml",
}
REQUIRED_STYLES = {
    "Title",
    "Heading 1",
    "Heading 2",
    "Heading 3",
    "Research Kicker",
    "Research Subtitle",
    "Research Meta",
    "Research Source",
    "Research Tag",
    "Research Caption",
}
FORBIDDEN_GLYPHS = {"\ufffd", "\u25a1", "\u25af"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w%$]+", text.casefold(), flags=re.UNICODE)


def is_subsequence(needles: list[str], haystack: list[str]) -> bool:
    position = 0
    for token in needles:
        while position < len(haystack) and haystack[position] != token:
            position += 1
        if position == len(haystack):
            return False
        position += 1
    return True


def xml_text(root: ET.Element) -> str:
    return " ".join(node.text or "" for node in root.iter(f"{{{W}}}t"))


def inspect(path: Path, source: Path | None) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    with zipfile.ZipFile(path) as package:
        names = set(package.namelist())
        missing = sorted(REQUIRED_ENTRIES - names)
        check("required_package_entries", not missing, {"missing": missing})

        document = ET.fromstring(package.read("word/document.xml"))
        styles = ET.fromstring(package.read("word/styles.xml"))
        core = ET.fromstring(package.read("docProps/core.xml"))
        footer = ET.fromstring(package.read("word/footer1.xml"))
        relationships = ET.fromstring(
            package.read("word/_rels/document.xml.rels")
        )

    text = xml_text(document)
    style_names = {
        node.get(f"{{{W}}}val", "").casefold()
        for node in styles.iter(f"{{{W}}}name")
        if node.get(f"{{{W}}}val")
    }
    missing_styles = sorted(
        style for style in REQUIRED_STYLES if style.casefold() not in style_names
    )
    check("required_styles", not missing_styles, {"missing": missing_styles})

    page_break_nodes = [
        node
        for node in document.iter(f"{{{W}}}br")
        if node.get(f"{{{W}}}type") == "page"
    ]
    paragraph_breaks = list(document.iter(f"{{{W}}}pageBreakBefore"))
    section_properties = list(document.iter(f"{{{W}}}sectPr"))
    next_page_sections = [
        node
        for node in section_properties
        if (type_node := node.find(f"{{{W}}}type")) is None
        or type_node.get(f"{{{W}}}val", "nextPage") == "nextPage"
    ]
    check(
        "paragraph_pagination",
        not page_break_nodes
        and (bool(paragraph_breaks) or len(next_page_sections) >= 2),
        {
            "explicit_page_break_nodes": len(page_break_nodes),
            "paragraph_page_breaks": len(paragraph_breaks),
            "next_page_sections": len(next_page_sections),
        },
    )

    missing_glyphs = sorted(glyph for glyph in FORBIDDEN_GLYPHS if glyph in text)
    check("no_missing_glyph_markers", not missing_glyphs, {"found": missing_glyphs})

    page_fields = [
        (node.text or "").strip()
        for node in footer.iter(f"{{{W}}}instrText")
        if (node.text or "").strip()
    ]
    check("footer_page_field", "PAGE" in page_fields, {"fields": page_fields})

    title = core.findtext(f"{{{DC}}}title", default="").strip()
    author = core.findtext(f"{{{DC}}}creator", default="").strip()
    check("core_metadata", bool(title and author), {"title": title, "author": author})

    relationship_rows = {}
    broken_relationships = []
    unsafe_external_targets = []
    for node in relationships.iter(f"{{{PR}}}Relationship"):
        relationship_id = node.get("Id", "")
        target = node.get("Target", "")
        target_mode = node.get("TargetMode", "")
        relationship_rows[relationship_id] = {
            "target": target,
            "target_mode": target_mode,
        }
        if target_mode == "External":
            if not re.match(r"^(https?://|mailto:)", target):
                unsafe_external_targets.append(
                    {"relationship_id": relationship_id, "target": target}
                )
            continue
        resolved = posixpath.normpath(posixpath.join("word", target))
        if resolved not in names:
            broken_relationships.append(
                {
                    "relationship_id": relationship_id,
                    "target": target,
                    "resolved": resolved,
                }
            )
    missing_hyperlink_relationships = []
    for hyperlink in document.iter(f"{{{W}}}hyperlink"):
        relationship_id = hyperlink.get(f"{{{R}}}id")
        if relationship_id and relationship_id not in relationship_rows:
            missing_hyperlink_relationships.append(relationship_id)
    check(
        "document_relationships",
        not (
            broken_relationships
            or unsafe_external_targets
            or missing_hyperlink_relationships
        ),
        {
            "relationship_count": len(relationship_rows),
            "broken": broken_relationships,
            "unsafe_external": unsafe_external_targets,
            "missing_hyperlink_relationships": missing_hyperlink_relationships,
        },
    )

    tracked_change_nodes = (
        len(list(document.iter(f"{{{W}}}ins")))
        + len(list(document.iter(f"{{{W}}}del")))
        + len(list(document.iter(f"{{{W}}}moveFrom")))
        + len(list(document.iter(f"{{{W}}}moveTo")))
    )
    comment_parts = sorted(name for name in names if name.startswith("word/comments"))
    check(
        "no_review_artifacts",
        tracked_change_nodes == 0 and not comment_parts,
        {
            "tracked_change_nodes": tracked_change_nodes,
            "comment_parts": comment_parts,
        },
    )

    placeholder_patterns = {
        "TODO": len(re.findall(r"\bTODO\b", text, flags=re.IGNORECASE)),
        "TBD": len(re.findall(r"\bTBD\b", text, flags=re.IGNORECASE)),
        "double_brace": text.count("{{") + text.count("}}"),
        "double_bracket": text.count("[[") + text.count("]]"),
    }
    check(
        "no_unresolved_placeholders",
        not any(placeholder_patterns.values()),
        placeholder_patterns,
    )

    media_parts = sorted(name for name in names if name.startswith("word/media/"))
    media_targets = {
        posixpath.normpath(posixpath.join("word", row["target"]))
        for row in relationship_rows.values()
        if row["target_mode"] != "External"
        and posixpath.normpath(posixpath.join("word", row["target"])).startswith(
            "word/media/"
        )
    }
    check(
        "media_inventory",
        set(media_parts) == media_targets,
        {
            "media_parts": media_parts,
            "relationship_targets": sorted(media_targets),
        },
    )

    heading_paragraphs = []
    paragraphs = list(document.iter(f"{{{W}}}p"))
    for index, paragraph in enumerate(paragraphs):
        style = paragraph.find(f"./{{{W}}}pPr/{{{W}}}pStyle")
        if style is None:
            continue
        style_id = style.get(f"{{{W}}}val", "")
        if style_id.startswith("Heading"):
            heading_paragraphs.append((index, xml_text(paragraph), style_id))
    empty_headings = [
        {"index": index, "style": style_id}
        for index, value, style_id in heading_paragraphs
        if not value.strip()
    ]
    check(
        "nonempty_headings",
        bool(heading_paragraphs) and not empty_headings,
        {"heading_count": len(heading_paragraphs), "empty": empty_headings},
    )

    tables = list(document.iter(f"{{{W}}}tbl"))
    table_count = len(tables)
    repeat_headers = len(list(document.iter(f"{{{W}}}tblHeader")))
    row_guards = len(list(document.iter(f"{{{W}}}cantSplit")))
    layout_table_count = 0
    governed_data_table_count = 0
    ungoverned_data_tables: list[int] = []
    for table_index, table in enumerate(tables, start=1):
        rows = list(table.findall(f"{{{W}}}tr"))
        cell_count = sum(
            len(row.findall(f"{{{W}}}tc"))
            for row in rows
        )
        # The renderer deliberately uses a one-row, one-cell table to create a
        # shaded callout with a left border. That object is page layout, not a
        # data table, and therefore must not be required to repeat a header.
        if len(rows) == 1 and cell_count == 1:
            layout_table_count += 1
            continue
        governed_data_table_count += 1
        has_repeat_header = any(
            node is not None
            for row in rows
            for node in [row.find(f".//{{{W}}}tblHeader")]
        )
        has_row_guard = any(
            node is not None
            for row in rows
            for node in [row.find(f".//{{{W}}}cantSplit")]
        )
        if not (has_repeat_header and has_row_guard):
            ungoverned_data_tables.append(table_index)
    check(
        "table_controls",
        not ungoverned_data_tables,
        {
            "table_count": table_count,
            "layout_table_count": layout_table_count,
            "governed_data_table_count": governed_data_table_count,
            "ungoverned_data_tables": ungoverned_data_tables,
            "repeating_headers": repeat_headers,
            "non_split_rows": row_guards,
        },
    )

    if source is not None:
        soup = BeautifulSoup(source.read_text(encoding="utf-8"), "html.parser")
        source_text = soup.body.get_text(" ", strip=True) if soup.body else ""
        source_words = word_tokens(source_text)
        output_words = word_tokens(text)
        check(
            "source_word_sequence",
            bool(source_words) and is_subsequence(source_words, output_words),
            {
                "source_words": len(source_words),
                "output_words": len(output_words),
            },
        )

    failed = [row["check"] for row in checks if not row["passed"]]
    return {
        "status": "PASS" if not failed else "FAIL",
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "checks": checks,
        "failed_checks": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.docx, args.source)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
