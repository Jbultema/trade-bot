# Final PDF publication standard

The completed investigation is delivered as three separate, publication-quality
PDFs. Chapter DOCX files remain useful intermediate and chapter-by-chapter
deliverables, but they are not substitutes for this final package.

## Required standalone publications

1. **Executive Summary — The Modern AI Financial System**
   - 1–4 substantive pages, excluding cover and source/method note;
   - written for a financially sophisticated reader who may not read the monograph;
   - complete enough to preserve every qualification that changes a conclusion's
     economic meaning.
2. **The Modern AI Financial System — A Forensic Reconstruction**
   - one complete book-length PDF containing Chapters 1–19;
   - includes front matter, methodology, all chapter appendices required to
     understand the analysis, consolidated glossary, source note, and index of
     principal entities, exposures, and reconstructed mechanisms.
3. **The “Burning Up” Essay**
   - one standalone long-form analytical article;
   - visually and editorially distinct from the monograph while using the same
     canonical evidence;
   - includes a short method/source note and observable falsifiers.

Each PDF must have its own source file, build manifest, content hash, and visual
inspection record. The three publications may share styles and canonical facts but
must not depend on one another to be intelligible.

## Editorial design

All three publications must use a restrained institutional-research design:

- readable serif body type and compatible sans-serif display type;
- consistent hierarchy for title, chapter, section, subsection, caption, and note;
- generous margins and line spacing suitable for long-form reading;
- page numbers, running heads, document title, publication date, and version;
- a cover page with title, subtitle, evidence cutoff, and explicit research status;
- a compact evidence-grade legend;
- tables with repeated headers, declared units, source notes, and no clipped cells;
- figures with titles, legends, units, dates, and canonical artifact references;
- callouts used sparingly for principal discoveries, boundaries, and falsifiers;
- color that remains legible in grayscale and does not carry meaning alone;
- no orphan headings, stranded captions, blank accidental pages, split single-row
  tables, or footers overlapping content.

The complete monograph must additionally include:

- clickable table of contents;
- PDF bookmarks for every chapter and major appendix;
- internal links from cross-references to chapters, figures, tables, and appendices;
- consolidated list of figures and tables when their final count warrants it;
- a glossary of accounting, contractual, project-finance, compute, and power terms;
- a final “what would change our mind” monitoring index; and
- navigation that makes each chapter usable independently.

## Evidence presentation

- Grades A, B, C, and D must be visible at the point where the distinction matters.
- Grade C inference and Grade D hypothesis must never inherit the typography of an
  unqualified Grade A fact.
- Every quantified statement must resolve to a canonical claim, exposure, model
  scenario, or explicitly identified analytical calculation.
- Tables must distinguish reported, reconstructed, scenario, approximate, rounded,
  undisclosed, not applicable, and not comparable values.
- Source notes must identify the evidence cutoff and preserve qualifications such as
  primarily attributable, associated with, nonadditive, and unnamed portfolio.
- Footnotes and endnotes must be live text, not raster images.

## Build and acceptance gates

The final PDF package may be generated only after the final canonical database and
all Chapter 1–19 publication manifests agree.

Each PDF must pass:

1. reproducible clean build from versioned source;
2. deterministic content and manifest accounting, allowing only explicitly
   documented renderer metadata differences;
3. page-count, file-size, embedded-font, hyperlink, bookmark, and metadata checks;
4. text extraction confirming that substantive text is searchable and in reading
   order;
5. table and figure inventory reconciliation;
6. citation and canonical-ID coverage audit;
7. automated checks for missing glyphs, overflow indicators, blank pages, duplicate
   pages, broken links, and unresolved placeholders;
8. visual inspection of every page at normal reading scale and representative
   print/grayscale inspection;
9. independent editorial review of the executive summary and essay against the
   accepted monograph; and
10. final primary-agent sign-off that no PDF silently differs from its accepted
    source.

`scripts/inspect_publication_pdf.swift` produces the page-level machine audit and
rendered inspection images. `scripts/validate_publication_package.py` fails closed
unless all three publications, their source trees, manifests, hashes, PDF audits,
and page-by-page visual acceptances form one consistent package.

The visual inspection record must enumerate every page, not merely sample the first
and last pages. A technically valid PDF is not publication-ready if its pagination,
tables, figures, citations, or typographic hierarchy impair interpretation.

## Delivery layout

```text
reports/ai_financial_system/publication/
  executive_summary/
    AI_Financial_System_Executive_Summary.pdf
    source/
    publication_manifest.json
    visual_inspection.csv
    visual_contact_sheet.pdf
    pdfkit_audit.json
    structural_pdf_audit.json
  full_monograph/
    The_Modern_AI_Financial_System_Forensic_Monograph.pdf
    source/
    publication_manifest.json
    visual_inspection.csv
    visual_contact_sheet.pdf
    pdfkit_audit.json
    structural_pdf_audit.json
  burning_up_essay/
    The_AI_Financial_System_Burning_Up_Essay.pdf
    source/
    publication_manifest.json
    visual_inspection.csv
    visual_contact_sheet.pdf
    pdfkit_audit.json
    structural_pdf_audit.json
  final_package_manifest.json
```

The final package manifest records the canonical database hash, evidence cutoff,
source commit or tree hash, renderer and font versions, individual PDF hashes, page
counts, build timestamps, and acceptance-review references.

Each publication manifest additionally records the deterministic twin hash, the
complete source-tree hash, the PDFKit page audit, the low-level structural/font
audit, and the every-page contact sheet. The deterministic twin must be
byte-identical to the delivered PDF; a merely visually similar rebuild does not pass.
