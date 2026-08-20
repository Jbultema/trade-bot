from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "research/ai_financial_system/scripts"


def _load_script(name: str) -> ModuleType:
    path = SCRIPT_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_publication_docx_is_deterministic_and_source_faithful(tmp_path: Path) -> None:
    renderer = _load_script("render_publication_docx")
    inspector = _load_script("inspect_publication_docx")
    source = tmp_path / "source.html"
    source.write_text(
        """
        <html>
          <head><title>Forensic publication specimen</title></head>
          <body>
            <section class="title-page">
              <div class="kicker">Forensic research</div>
              <h1>Test chapter</h1>
              <div class="subtitle">A governed publication specimen.</div>
            </section>
            <header>
              <h1>Research perimeter</h1>
              <p>Header scope survives rendering.</p>
            </header>
            <h2>Observed facts</h2>
            <p>The renderer retains the authoritative words in order.</p>
            <table>
              <caption>Evidence-state matrix</caption>
              <tr><th>Evidence</th><th>State</th></tr>
              <tr><td>Primary filing</td><td>Preserved</td></tr>
            </table>
            <h3>Interpretation</h3>
            <h4>Bounded implication</h4>
            <div class="finding">Preservation does not itself prove a claim.</div>
            <figure aria-label="Dependency sequence">
              <figcaption>Figure one. Contract to cash sequence.</figcaption>
              <div class="inference">
                <span class="grade-c">Grade C inference.</span>
                Delivery can precede productive use.
              </div>
              <table>
                <caption>Nested figure evidence</caption>
                <tr><th>Stage</th><th>Boundary</th></tr>
                <tr><td>Delivery</td><td>Not utilization</td></tr>
              </table>
            </figure>
            <div class="hypothesis">
              <span class="grade-d">Grade D hypothesis.</span>
              Repricing could transmit stress.
            </div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    first = tmp_path / "first.docx"
    second = tmp_path / "second.docx"

    renderer.render(source, first)
    renderer.render(source, second)
    result = inspector.inspect(first, source)

    assert result["status"] == "PASS"
    assert result["failed_checks"] == []
    assert inspector.sha256(first) == inspector.sha256(second)
    from docx import Document

    rendered = Document(first)
    styles = {
        paragraph.text: paragraph.style.name
        for paragraph in rendered.paragraphs
        if paragraph.text
    }
    assert styles["Interpretation"] == "Heading 3"
    assert styles["Bounded implication"] == "Heading 4"
    body_heading = next(
        paragraph
        for paragraph in rendered.paragraphs
        if paragraph.text == "Research perimeter"
    )
    assert body_heading.paragraph_format.page_break_before is True
    with __import__("zipfile").ZipFile(first) as package:
        document_xml = package.read("word/document.xml")
    assert b'<w:br w:type="page"' not in document_xml


def test_docx_inspector_detects_exact_source_word_population(tmp_path: Path) -> None:
    renderer = _load_script("render_publication_docx")
    inspector = _load_script("inspect_publication_docx")
    source = tmp_path / "source.html"
    source.write_text(
        """
        <html><head><title>Word sequence</title></head><body>
          <section class="title-page"><h1>Chapter</h1></section>
          <h2>Observed facts</h2>
          <p>Reported amounts remain distinct from reconstructed scenarios.</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "output.docx"
    renderer.render(source, output)

    result = inspector.inspect(output, source)
    sequence = next(
        row for row in result["checks"] if row["check"] == "source_word_sequence"
    )

    assert result["status"] == "PASS"
    assert sequence["detail"] == {"source_words": 10, "output_words": 10}


def test_docx_inspector_treats_one_cell_callout_as_layout_not_data_table(
    tmp_path: Path,
) -> None:
    renderer = _load_script("render_publication_docx")
    inspector = _load_script("inspect_publication_docx")
    source = tmp_path / "callout.html"
    source.write_text(
        """
        <html><head><title>Callout specimen</title></head><body>
          <section class="title-page">
            <h1>Callout specimen</h1>
            <p class="meta">Governed layout test.</p>
          </section>
          <h2>Observed fact</h2>
          <div class="finding">This shaded box is not a tabular dataset.</div>
        </body></html>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "callout.docx"
    renderer.render(source, output)

    result = inspector.inspect(output, source)
    table_controls = next(
        row for row in result["checks"] if row["check"] == "table_controls"
    )

    assert result["status"] == "PASS"
    assert table_controls["passed"] is True
    assert table_controls["detail"]["layout_table_count"] == 1
    assert table_controls["detail"]["governed_data_table_count"] == 0


def test_docx_inspector_rejects_real_table_without_header_semantics(
    tmp_path: Path,
) -> None:
    renderer = _load_script("render_publication_docx")
    inspector = _load_script("inspect_publication_docx")
    source = tmp_path / "unguarded-table.html"
    source.write_text(
        """
        <html><head><title>Table control specimen</title></head><body>
          <section class="title-page">
            <h1>Table control specimen</h1>
            <p class="meta">Governed negative control.</p>
          </section>
          <h2>Observed rows</h2>
          <table>
            <tr><td>Field</td><td>Value</td></tr>
            <tr><td>Commitment</td><td>Announced only</td></tr>
          </table>
        </body></html>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "unguarded-table.docx"
    renderer.render(source, output)

    result = inspector.inspect(output, source)
    table_controls = next(
        row for row in result["checks"] if row["check"] == "table_controls"
    )

    assert result["status"] == "FAIL"
    assert table_controls["passed"] is False
    assert table_controls["detail"]["layout_table_count"] == 0
    assert table_controls["detail"]["governed_data_table_count"] == 1
    assert table_controls["detail"]["ungoverned_data_tables"] == [1]
