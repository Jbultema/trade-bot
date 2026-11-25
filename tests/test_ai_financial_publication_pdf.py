from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from bs4 import BeautifulSoup
from pypdf import PdfReader

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


def test_portable_publication_font_mapping_keeps_embedded_styles_and_unicode(tmp_path):
    renderer = _load_script("render_publication_pdf")
    fonts = renderer.publication_font_files(tmp_path)
    assert set(fonts) == set(renderer.FONT_FILES)
    assert all(Path(path).parent == tmp_path for path in fonts.values())
    assert fonts["AFSUnicode"].endswith("DejaVuSans.ttf")
    assert fonts["AFSBodyBoldItalic"].endswith("DejaVuSerif-BoldItalic.ttf")


def test_publication_pdf_is_deterministic_searchable_and_embedded(tmp_path: Path) -> None:
    renderer = _load_script("render_publication_pdf")
    auditor = _load_script("audit_publication_pdf")
    source = tmp_path / "source.html"
    source.write_text(
        """
        <html><body>
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
          <p>The renderer retains searchable text and embedded fonts.</p>
          <table>
            <caption>Evidence-state matrix</caption>
            <tr><th>Evidence</th><th>State</th></tr>
            <tr><td>Primary filing</td><td>Preserved</td></tr>
          </table>
          <h3>Interpretation</h3>
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
          <div class="scenario">
            <span class="grade-d">Grade D scenario.</span>
            Repricing could transmit stress.
          </div>
          <div class="step"><strong>Defined input</strong>Observed fact → contract
          ↔ claimant.</div>
        </body></html>
        """,
        encoding="utf-8",
    )
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    kwargs = {
        "title": "Publication renderer test",
        "subtitle": "The Modern AI Financial System",
        "evidence_cutoff": "2026-07-26",
        "include_front_matter": True,
    }
    renderer.render([source], first, **kwargs)
    renderer.render([source], second, **kwargs)

    result = auditor.audit(first, expected_title=kwargs["title"], minimum_pages=3)

    assert result["status"] == "pass"
    assert result["all_fonts_embedded"] is True
    assert result["outline_count"] >= 3
    assert result["blank_pages"] == []
    assert auditor.sha256(first) == auditor.sha256(second)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(first).pages)
    for expected in (
        "Evidence-state matrix",
        "Research perimeter",
        "Header scope survives rendering.",
        "Figure one. Contract to cash sequence.",
        "Grade C inference.",
        "Nested figure evidence",
        "Grade D scenario.",
        "Defined input",
        "Observed fact",
        "contract",
        "claimant.",
    ):
        assert expected in text
    assert "\x00" not in text
    assert "inputObserved" not in text
    assert "contract ↔ claimant" in text


def test_outline_levels_normalize_when_source_starts_at_h2(tmp_path: Path) -> None:
    renderer = _load_script("render_publication_pdf")
    auditor = _load_script("audit_publication_pdf")
    source = tmp_path / "source.html"
    source.write_text(
        """
        <html><body>
          <h2>Observed facts</h2>
          <p>A chapter fragment can begin at heading level two.</p>
          <h3>Evidence detail</h3>
          <p>The outline remains structurally valid.</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "normalized.pdf"
    renderer.render(
        [source],
        output,
        title="Normalized outline",
        subtitle="The Modern AI Financial System",
        evidence_cutoff="2026-07-26",
        include_front_matter=True,
    )

    result = auditor.audit(output, expected_title="Normalized outline", minimum_pages=3)

    assert result["status"] == "pass"
    assert result["outline_titles"] == ["Observed facts", "Evidence detail"]


def test_each_chapter_source_restarts_outline_hierarchy(tmp_path: Path) -> None:
    renderer = _load_script("render_publication_pdf")
    auditor = _load_script("audit_publication_pdf")
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    first.write_text(
        "<html><body><h1>Chapter one</h1><h2>Evidence</h2><p>First.</p></body></html>",
        encoding="utf-8",
    )
    second.write_text(
        "<html><body><h2>Chapter two</h2><h3>Evidence</h3><p>Second.</p></body></html>",
        encoding="utf-8",
    )
    output = tmp_path / "combined.pdf"
    renderer.render(
        [first, second],
        output,
        title="Combined outline",
        subtitle="The Modern AI Financial System",
        evidence_cutoff="2026-07-26",
        include_front_matter=True,
    )

    result = auditor.audit(output, expected_title="Combined outline", minimum_pages=4)

    assert result["status"] == "pass"
    assert result["outline_titles"] == [
        "Chapter one",
        "Evidence",
        "Chapter two",
        "Evidence",
    ]
    outline = PdfReader(output).outline
    assert len([item for item in outline if not isinstance(item, list)]) == 2


def test_unordered_lists_do_not_render_as_repeated_numeric_markers() -> None:
    renderer = _load_script("render_publication_pdf")
    renderer.register_fonts()
    context = renderer.RenderContext(
        styles=renderer.build_styles(),
        source_name="list-regression.html",
    )
    unordered = BeautifulSoup(
        "<ul><li>First finding</li><li>Second finding</li></ul>",
        "html.parser",
    ).ul
    ordered = BeautifulSoup(
        "<ol><li>First step</li><li>Second step</li></ol>",
        "html.parser",
    ).ol
    assert unordered is not None
    assert ordered is not None

    unordered_flowable = renderer.html_list(unordered, context)
    ordered_flowable = renderer.html_list(ordered, context)

    assert unordered_flowable._bulletType == "bullet"
    assert unordered_flowable._start == "bulletchar"
    assert ordered_flowable._bulletType == "1"
    assert ordered_flowable._start == "1"


def test_title_page_starts_fresh_and_stays_together(tmp_path: Path) -> None:
    renderer = _load_script("render_publication_pdf")
    source = tmp_path / "title_page_pagination.html"
    preceding = "".join(
        f"<p>Preceding chapter evidence paragraph {index}. "
        "This text consumes enough space to leave an awkward partial page.</p>"
        for index in range(18)
    )
    source.write_text(
        f"""
        <html><body>
          <h1>Chapter one</h1>
          {preceding}
          <section class="title-page">
            <div class="kicker">Forensic research monograph</div>
            <h1>Chapter two<br/>A title that must remain intact</h1>
            <div class="subtitle">The subtitle and evidence boundary belong to
            the same governed title page.</div>
            <div class="meta">Evidence cutoff: 26 July 2026<br/>
            Scope boundary retained.</div>
          </section>
          <h2>Observed facts</h2>
          <p>The chapter body begins after the title page.</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "title_page_pagination.pdf"
    renderer.render(
        [source],
        output,
        title="Title-page pagination",
        subtitle="The Modern AI Financial System",
        evidence_cutoff="2026-07-26",
        include_front_matter=False,
    )

    pages = [" ".join((page.extract_text() or "").split()) for page in PdfReader(output).pages]
    title_pages = [
        index
        for index, text in enumerate(pages)
        if "A title that must remain intact" in text
    ]
    assert len(title_pages) == 1
    title_text = pages[title_pages[0]]
    assert "Chapter two" in title_text
    assert "same governed title page" in title_text
    assert "Scope boundary retained" in title_text
    assert "Preceding chapter evidence paragraph" not in title_text
    assert "Observed facts" not in title_text


def test_contact_sheet_is_deterministic_and_covers_every_source_page(
    tmp_path: Path,
) -> None:
    builder = _load_script("build_pdf_contact_sheet")
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
        "YAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    images = []
    for page_number in range(1, 6):
        image = tmp_path / f"page_{page_number:04d}.png"
        image.write_bytes(one_pixel_png)
        images.append(image)

    first = tmp_path / "contact_first.pdf"
    second = tmp_path / "contact_second.pdf"
    first_result = builder.build(
        images,
        first,
        title="Every-page inspection",
        columns=2,
        rows=2,
    )
    second_result = builder.build(
        images,
        second,
        title="Every-page inspection",
        columns=2,
        rows=2,
    )

    assert first_result["status"] == "pass"
    assert first_result["source_pages"] == 5
    assert first_result["contact_sheet_pages"] == 2
    assert [row["page_number"] for row in first_result["sources"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert len(PdfReader(first).pages) == 2
    assert first_result["output_sha256"] == second_result["output_sha256"]
