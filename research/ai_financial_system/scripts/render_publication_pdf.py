#!/usr/bin/env python3
"""Render governed AI-financial-system HTML sources as searchable PDFs."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from html import escape
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

rl_config.invariant = 1

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN_X = 0.72 * inch
MARGIN_TOP = 0.72 * inch
MARGIN_BOTTOM = 0.68 * inch
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN_X)

FONT_FILES = {
    "AFSBody": "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "AFSBodyItalic": "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
    "AFSBodyBold": "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "AFSBodyBoldItalic": "/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf",
    "AFSDisplay": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "AFSDisplayItalic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "AFSDisplayBold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "AFSDisplayBoldItalic": "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
    "AFSUnicode": "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
}


@dataclass(frozen=True)
class RenderContext:
    styles: dict[str, ParagraphStyle]
    source_name: str


class OutlineReset(Spacer):
    """Zero-height marker that restarts relative heading levels per source."""

    def __init__(self) -> None:
        super().__init__(1, 0)


def register_fonts() -> None:
    for name, path in FONT_FILES.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"required publication font is missing: {path}")
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(
        "AFSBody",
        normal="AFSBody",
        bold="AFSBodyBold",
        italic="AFSBodyItalic",
        boldItalic="AFSBodyBoldItalic",
    )
    pdfmetrics.registerFontFamily(
        "AFSDisplay",
        normal="AFSDisplay",
        bold="AFSDisplayBold",
        italic="AFSDisplayItalic",
        boldItalic="AFSDisplayBoldItalic",
    )
    # ReportLab otherwise injects an unembedded Helvetica resource into every
    # page stream even when all visible text uses the governed font families.
    rl_config.canvas_basefontname = "AFSBody"


def build_styles(
    *,
    body_font_size: float = 9.4,
    body_leading: float = 13.5,
) -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=sample["BodyText"],
        fontName="AFSBody",
        fontSize=body_font_size,
        leading=body_leading,
        textColor=colors.HexColor("#182230"),
        spaceAfter=7,
        widowOrphanControl=1,
    )
    return {
        "Body": body,
        "Title": ParagraphStyle(
            "Title",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=28,
            leading=31,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=18,
            alignment=TA_LEFT,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=body,
            fontName="AFSBody",
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#486581"),
            spaceAfter=25,
        ),
        "Kicker": ParagraphStyle(
            "Kicker",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#486581"),
            spaceAfter=15,
            uppercase=True,
        ),
        "Meta": ParagraphStyle(
            "Meta",
            parent=body,
            fontName="AFSDisplay",
            fontSize=7.8,
            leading=11,
            textColor=colors.HexColor("#627D98"),
            borderColor=colors.HexColor("#9FB3C8"),
            borderWidth=0.5,
            borderPadding=(9, 0, 0, 0),
            spaceBefore=8,
        ),
        "Heading1": ParagraphStyle(
            "Heading1",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=20,
            leading=23,
            textColor=colors.HexColor("#102A43"),
            spaceBefore=8,
            spaceAfter=12,
            keepWithNext=True,
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=14.5,
            leading=18,
            textColor=colors.HexColor("#163A5F"),
            spaceBefore=16,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "Heading3": ParagraphStyle(
            "Heading3",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#234E70"),
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "Heading4": ParagraphStyle(
            "Heading4",
            parent=body,
            fontName="AFSDisplayBoldItalic",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#334E68"),
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "Source": ParagraphStyle(
            "Source",
            parent=body,
            fontName="AFSDisplay",
            fontSize=7.2,
            leading=10,
            textColor=colors.HexColor("#486581"),
            spaceAfter=5,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=7.8,
            leading=10.2,
            textColor=colors.HexColor("#334E68"),
            spaceBefore=4,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "Callout": ParagraphStyle(
            "Callout",
            parent=body,
            fontName="AFSBody",
            fontSize=8.9,
            leading=12.8,
            textColor=colors.HexColor("#182230"),
            spaceAfter=0,
        ),
        "PullQuote": ParagraphStyle(
            "PullQuote",
            parent=body,
            fontName="AFSBodyItalic",
            fontSize=max(11.2, body_font_size),
            leading=max(16.0, body_leading),
            textColor=colors.HexColor("#163A5F"),
            spaceAfter=0,
        ),
        "Table": ParagraphStyle(
            "Table",
            parent=body,
            fontName="AFSDisplay",
            fontSize=6.8,
            leading=8.6,
            spaceAfter=0,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=6.8,
            leading=8.6,
            textColor=colors.white,
            spaceAfter=0,
        ),
        "TOCTitle": ParagraphStyle(
            "TOCTitle",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=21,
            leading=24,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=20,
        ),
        "CoverTitle": ParagraphStyle(
            "CoverTitle",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=31,
            leading=35,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=16,
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=body,
            fontName="AFSBody",
            fontSize=14,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#486581"),
            spaceAfter=18,
        ),
        "LegendLabel": ParagraphStyle(
            "LegendLabel",
            parent=body,
            fontName="AFSDisplayBold",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#163A5F"),
            spaceAfter=0,
        ),
        "LegendText": ParagraphStyle(
            "LegendText",
            parent=body,
            fontName="AFSDisplay",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#486581"),
            spaceAfter=0,
        ),
    }


def inline_markup(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        value = escape(str(node))
        # Georgia and Arial on macOS do not provide the arrow glyphs used by
        # the dependency diagrams. Preserve those characters with a governed
        # embedded Unicode fallback instead of emitting .notdef squares.
        return (
            value.replace("↔", '<font name="AFSUnicode">↔</font>')
            .replace("→", '<font name="AFSUnicode">→</font>')
            .replace("←", '<font name="AFSUnicode">←</font>')
        )
    content = "".join(inline_markup(child) for child in node.children)
    name = node.name.lower()
    if name in {"strong", "b"}:
        parent = node.parent if isinstance(node.parent, Tag) else None
        parent_classes = set(parent.get("class", [])) if parent is not None else set()
        block_lead = bool(
            parent is not None
            and parent.name == "div"
            and parent_classes
            & {
                "step",
                "flow-card",
                "grid-card",
                "role-card",
                "channel",
            }
        )
        suffix = "<br/>" if block_lead else ""
        return f"<b>{content}</b>{suffix}"
    if name in {"em", "i"}:
        return f"<i>{content}</i>"
    if name == "br":
        return "<br/>"
    if name == "code":
        return f'<font name="AFSDisplay" size="7.5">{content}</font>'
    if name == "a":
        href = escape(node.get("href", ""), quote=True)
        return f'<a href="{href}" color="#245B82">{content}</a>' if href else content
    if name == "span":
        classes = set(node.get("class", []))
        if "source" in classes:
            return f'<font name="AFSDisplay" size="7.2" color="#486581">{content}</font>'
        if "tag" in classes:
            return f'<font name="AFSDisplay" size="6.8" color="#334E68">{content}</font>'
        if classes & {"grade-b", "grade-c", "grade-d"}:
            color = (
                "#2F6B4F"
                if "grade-b" in classes
                else "#9C5B16"
                if "grade-c" in classes
                else "#9B2C2C"
            )
            return (
                f'<font name="AFSDisplayBold" size="7.2" color="{color}">'
                f"{content}</font>"
            )
    return content


def normalized_markup(node: Tag) -> str:
    value = inline_markup(node)
    return re.sub(r"\s+", " ", value).strip()


def callout(node: Tag, context: RenderContext) -> Table:
    classes = set(node.get("class", []))
    if "pull" in classes:
        background = colors.HexColor("#EAF2F7")
        accent = colors.HexColor("#26658E")
        style_name = "PullQuote"
        padding = 12
    elif "warning" in classes:
        background = colors.HexColor("#FFF6E6")
        accent = colors.HexColor("#C05621")
        style_name = "Callout"
        padding = 8
    elif "notice" in classes:
        background = colors.HexColor("#EDF9F0")
        accent = colors.HexColor("#2F855A")
        style_name = "Callout"
        padding = 8
    else:
        background = colors.HexColor("#F2F6FA")
        accent = colors.HexColor("#3E6C8E")
        style_name = "Callout"
        padding = 8
    table = Table(
        [[Paragraph(normalized_markup(node), context.styles[style_name])]],
        colWidths=[CONTENT_WIDTH],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#BCCCDC")),
                ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
                ("FONTNAME", (0, 0), (-1, -1), "AFSBody"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.9),
                ("LEADING", (0, 0), (-1, -1), 12.8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ]
        )
    )
    return table


def html_table(node: Tag, context: RenderContext) -> Table:
    rows: list[list[Paragraph]] = []
    header_rows = 0
    for row in node.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        is_header = all(cell.name == "th" for cell in cells)
        if is_header and not rows:
            header_rows = 1
        style = context.styles["TableHeader"] if is_header else context.styles["Table"]
        rows.append([Paragraph(normalized_markup(cell), style) for cell in cells])
    if not rows:
        return Table([[""]], colWidths=[CONTENT_WIDTH])
    columns = max(len(row) for row in rows)
    for row in rows:
        row.extend(Paragraph("", context.styles["Table"]) for _ in range(columns - len(row)))
    widths = [CONTENT_WIDTH / columns] * columns
    table = Table(rows, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#163A5F"))
                if header_rows
                else ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2EC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "AFSDisplay"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.8),
                ("LEADING", (0, 0), (-1, -1), 8.6),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
            ]
        )
    )
    return table


def html_list(node: Tag, context: RenderContext) -> ListFlowable:
    items: list[ListItem] = []
    for item in node.find_all("li", recursive=False):
        items.append(
            ListItem(
                Paragraph(normalized_markup(item), context.styles["Body"]),
                leftIndent=12,
            )
        )
    options: dict[str, object] = {
        "bulletType": "1" if node.name == "ol" else "bullet",
        "leftIndent": 18,
        "bulletFontName": "AFSDisplay",
        "bulletFontSize": 8,
        "spaceAfter": 7,
    }
    # ReportLab interprets an explicit ``start="1"`` as the literal bullet
    # glyph even when bulletType is ``bullet``.  That made every unordered
    # list render as repeated "1" markers.  Only ordered lists should receive
    # a numeric start value; unordered lists use ReportLab's governed bullet.
    if node.name == "ol":
        options["start"] = "1"
    return ListFlowable(items, **options)


def blocks_from_container(
    container: Tag,
    context: RenderContext,
    *,
    title_page: bool = False,
) -> list[object]:
    story: list[object] = []
    if title_page:
        story.append(Spacer(1, 1.35 * inch))
    for child in container.children:
        if not isinstance(child, Tag):
            continue
        classes = set(child.get("class", []))
        if child.name in {"section", "header", "main", "article", "footer"}:
            is_title = "title-page" in classes
            if is_title:
                if story and not isinstance(story[-1], PageBreak):
                    story.append(PageBreak())
                title_story = blocks_from_container(
                    child, context, title_page=True
                )
                story.append(KeepTogether(title_story))
                story.append(PageBreak())
            else:
                story.extend(
                    blocks_from_container(child, context, title_page=False)
                )
            continue
        if child.name in {"h1", "h2", "h3", "h4"}:
            if "page-break" in classes and story:
                story.append(PageBreak())
            style_name = {
                "h1": "Title" if title_page else "Heading1",
                "h2": "Heading2",
                "h3": "Heading3",
                "h4": "Heading4",
            }[child.name]
            heading = Paragraph(normalized_markup(child), context.styles[style_name])
            heading._afs_outline_text = child.get_text(" ", strip=True)
            story.append(heading)
            continue
        if child.name == "p":
            style = "Source" if "source" in classes else "Body"
            paragraph = Paragraph(normalized_markup(child), context.styles[style])
            if "keep-together" in classes:
                story.append(KeepTogether([paragraph]))
            else:
                story.append(paragraph)
            continue
        if child.name == "div":
            if classes & {
                "finding",
                "notice",
                "warning",
                "boundary",
                "inference",
                "scenario",
                "hypothesis",
                "pull",
            }:
                story.extend([callout(child, context), Spacer(1, 8)])
            elif "kicker" in classes:
                story.append(Paragraph(normalized_markup(child), context.styles["Kicker"]))
            elif "subtitle" in classes:
                story.append(Paragraph(normalized_markup(child), context.styles["Subtitle"]))
            elif "meta" in classes:
                story.append(Paragraph(normalized_markup(child), context.styles["Meta"]))
            else:
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
                    story.extend(
                        blocks_from_container(child, context, title_page=title_page)
                    )
                elif child.get_text(" ", strip=True):
                    story.append(
                        Paragraph(normalized_markup(child), context.styles["Body"])
                    )
            continue
        if child.name == "figure":
            story.extend(blocks_from_container(child, context, title_page=title_page))
            continue
        if child.name == "figcaption":
            story.append(Paragraph(normalized_markup(child), context.styles["Caption"]))
            continue
        if child.name == "table":
            caption = child.find("caption", recursive=False)
            if caption is not None:
                story.append(
                    Paragraph(normalized_markup(caption), context.styles["Caption"])
                )
            story.extend([html_table(child, context), Spacer(1, 7)])
            continue
        if child.name in {"ul", "ol"}:
            story.append(html_list(child, context))
            continue
        if child.name == "hr":
            story.append(Spacer(1, 9))
    return story


class PublicationDocTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename: str,
        *,
        title: str,
        author: str,
        subject: str,
    ) -> None:
        self.publication_title = title
        self.publication_author = author
        self.publication_subject = subject
        self._bookmark_serial = 0
        self._outline_level_offset: int | None = None
        super().__init__(
            filename,
            pagesize=LETTER,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=title,
            author=author,
            subject=subject,
            creator="Trade Bot AI Financial System governed publication renderer",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            PageTemplate(id="publication", frames=[frame], onPage=self._on_page)
        )

    def beforeDocument(self) -> None:
        self._bookmark_serial = 0
        self._outline_level_offset = None

    def _on_page(self, canvas: object, document: object) -> None:
        canvas.saveState()
        canvas.setTitle(self.publication_title)
        canvas.setAuthor(self.publication_author)
        canvas.setSubject(self.publication_subject)
        if self.page > 1:
            canvas.setStrokeColor(colors.HexColor("#BCCCDC"))
            canvas.setLineWidth(0.35)
            canvas.line(MARGIN_X, PAGE_HEIGHT - 0.45 * inch, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 0.45 * inch)
            canvas.setFont("AFSDisplay", 7)
            canvas.setFillColor(colors.HexColor("#627D98"))
            canvas.drawString(MARGIN_X, PAGE_HEIGHT - 0.36 * inch, self.publication_title[:72])
            canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 0.38 * inch, str(self.page))
        canvas.restoreState()

    def afterFlowable(self, flowable: object) -> None:
        if isinstance(flowable, OutlineReset):
            self._outline_level_offset = None
            return
        if not isinstance(flowable, Paragraph):
            return
        level_by_style = {
            "Title": 0,
            "Heading1": 0,
            "Heading2": 1,
            "Heading3": 2,
            "Heading4": 3,
        }
        if flowable.style.name not in level_by_style:
            return
        source_level = level_by_style[flowable.style.name]
        if self._outline_level_offset is None:
            self._outline_level_offset = source_level
        level = max(0, source_level - self._outline_level_offset)
        text = getattr(flowable, "_afs_outline_text", flowable.getPlainText())
        self._bookmark_serial += 1
        key = f"heading-{self._bookmark_serial:04d}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def cover_story(
    title: str,
    subtitle: str,
    evidence_cutoff: str,
    version: str,
    publication_date: str,
    styles: dict[str, ParagraphStyle],
) -> list[object]:
    legend_rows = [
        [
            Paragraph("A — Primary", styles["LegendLabel"]),
            Paragraph(
                "Filings, audited statements, transcripts, bond documents, "
                "contracts, and official announcements.",
                styles["LegendText"],
            ),
        ],
        [
            Paragraph("B — Reporting", styles["LegendLabel"]),
            Paragraph(
                "High-quality financial reporting and rating-agency analysis.",
                styles["LegendText"],
            ),
        ],
        [
            Paragraph("C — Inference", styles["LegendLabel"]),
            Paragraph(
                "A reconstruction supported by multiple sources but not directly disclosed.",
                styles["LegendText"],
            ),
        ],
        [
            Paragraph("D — Hypothesis", styles["LegendLabel"]),
            Paragraph(
                "A clearly labeled proposition requiring additional evidence.",
                styles["LegendText"],
            ),
        ],
    ]
    legend = Table(
        legend_rows,
        colWidths=[1.0 * inch, CONTENT_WIDTH - 1.0 * inch],
        hAlign="CENTER",
    )
    legend.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F6FA")),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCCCDC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E2EC")),
                ("FONTNAME", (0, 0), (-1, -1), "AFSDisplay"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return [
        Spacer(1, 1.15 * inch),
        Paragraph(escape(title), styles["CoverTitle"]),
        Paragraph(escape(subtitle), styles["CoverSubtitle"]),
        Spacer(1, 0.18 * inch),
        Paragraph(
            f"Version: {escape(version)} · Publication date: {escape(publication_date)}<br/>"
            f"Evidence cutoff: {escape(evidence_cutoff)}<br/>"
            "Forensic research monograph · Investigative research · Not investment advice",
            ParagraphStyle(
                "CoverMeta",
                parent=styles["Meta"],
                alignment=TA_CENTER,
                borderWidth=0,
            ),
        ),
        Spacer(1, 0.18 * inch),
        legend,
        PageBreak(),
    ]


def toc_story(styles: dict[str, ParagraphStyle]) -> list[object]:
    toc = TableOfContents()
    toc.tableStyle = TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("FONTNAME", (0, 0), (-1, -1), "AFSDisplay"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEADING", (0, 0), (-1, -1), 13),
        ]
    )
    toc.levelStyles = [
        ParagraphStyle(
            f"TOCLevel{level}",
            parent=styles["Body"],
            fontName="AFSDisplayBold" if level == 0 else "AFSDisplay",
            fontSize=9.5 - min(level, 2) * 0.5,
            leading=13,
            leftIndent=level * 14,
            firstLineIndent=0,
            textColor=colors.HexColor("#163A5F" if level < 2 else "#486581"),
        )
        for level in range(4)
    ]
    return [
        Paragraph("Contents", styles["TOCTitle"]),
        toc,
        PageBreak(),
    ]


def source_story(path: Path, styles: dict[str, ParagraphStyle]) -> list[object]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    if soup.body is None:
        raise ValueError(f"HTML source has no body: {path}")
    context = RenderContext(styles=styles, source_name=path.name)
    return blocks_from_container(soup.body, context)


def render(
    inputs: Iterable[Path],
    output: Path,
    *,
    title: str,
    subtitle: str,
    evidence_cutoff: str,
    include_front_matter: bool,
    version: str = "Research release 1.0",
    publication_date: str = "26 July 2026",
    body_font_size: float = 9.4,
    body_leading: float = 13.5,
) -> None:
    register_fonts()
    styles = build_styles(
        body_font_size=body_font_size,
        body_leading=body_leading,
    )
    story: list[object] = []
    if include_front_matter:
        story.extend(
            cover_story(
                title,
                subtitle,
                evidence_cutoff,
                version,
                publication_date,
                styles,
            )
        )
        story.extend(toc_story(styles))
    for index, source in enumerate(inputs):
        if index and story and not isinstance(story[-1], PageBreak):
            story.append(PageBreak())
        story.append(OutlineReset())
        story.extend(source_story(source, styles))
    output.parent.mkdir(parents=True, exist_ok=True)
    document = PublicationDocTemplate(
        str(output),
        title=title,
        author="AI Financial System Research Project",
        subject=subtitle,
    )
    document.multiBuild(
        story,
        canvasmaker=partial(
            Canvas,
            initialFontName="AFSBody",
            initialFontSize=9.4,
            initialLeading=13.5,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--evidence-cutoff", default="26 July 2026")
    parser.add_argument("--version", default="Research release 1.0")
    parser.add_argument("--publication-date", default="26 July 2026")
    parser.add_argument("--body-font-size", type=float, default=9.4)
    parser.add_argument("--body-leading", type=float, default=13.5)
    parser.add_argument("--front-matter", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [path for path in args.input if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input source(s): {missing}")
    render(
        args.input,
        args.output,
        title=args.title,
        subtitle=args.subtitle,
        evidence_cutoff=args.evidence_cutoff,
        include_front_matter=args.front_matter,
        version=args.version,
        publication_date=args.publication_date,
        body_font_size=args.body_font_size,
        body_leading=args.body_leading,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
