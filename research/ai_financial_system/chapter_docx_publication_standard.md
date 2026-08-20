# Chapter DOCX publication standard

## Purpose

Each accepted chapter is delivered as a professionally formatted, independently
usable DOCX before assembly into the complete monograph. A file that merely opens,
contains the prose, or was converted from HTML is not publication-ready.

The accepted HTML/source package and canonical research artifacts govern content.
The DOCX is a rendered publication derivative and may not silently alter, omit, or
reorder substantive text, evidence grades, tables, figures, qualifiers, source
notes, or canonical identities.

## Required chapter package

```text
reports/ai_financial_system/chapters/chapter_NN/
  AI_Financial_System_Chapter_NN.docx
  source/
  chapter_NN_publication_manifest.json
  chapter_NN_docx_inspection.json
  chapter_NN_visual_inspection.csv
```

The manifest records:

- accepted chapter-packet and review identities;
- canonical database hash and evidence cutoff;
- authoritative source-file hashes;
- renderer source hash, version, and Python/package versions;
- font names and availability state;
- DOCX hash and byte count;
- automated inspection result;
- page count from the accepted visual-review derivative; and
- reviewer and primary-agent acceptance references.

## Editorial and typographic requirements

Every chapter DOCX must have:

- a restrained institutional-research design consistent across Chapters 1–19;
- a proper title page with chapter number, title, subtitle, evidence cutoff,
  research status, and governing evidence-package identity;
- real Word paragraph styles for title, headings, body, source notes, captions,
  tags, lists, and metadata—not visual formatting alone;
- one logically nested heading hierarchy with no empty or orphan headings;
- readable serif body type and compatible sans-serif display type;
- consistent margins, line spacing, widow/orphan control, and keep-with-next
  behavior;
- running head, footer, and dynamic page-number field;
- paragraph-level pagination rather than visible break characters;
- tables with declared units, repeated headers, protected rows, readable column
  widths, and no clipped or split single-row content;
- figures with title, legend, units, date scope, source note, and alt-text or
  machine-readable caption;
- visibly distinct Grade-C inference and Grade-D hypothesis treatment where the
  distinction affects interpretation;
- live text for citations, source notes, footnotes, and endnotes; and
- colors and labels that remain interpretable in grayscale.

Source-code wrapping, indentation, or line breaks must not become forced line breaks
in ordinary prose. Explicit semantic breaks remain permitted.

## Content fidelity gates

1. Every authoritative source word appears in the DOCX in the same order, allowing
   only renderer-inserted navigation, running matter, and controlled separators.
2. Every table and figure in the accepted source reconciles one-for-one to the DOCX
   inventory.
3. Every canonical claim, exposure, model, source, and unknown identity cited in the
   chapter remains present and machine-extractable.
4. No unresolved placeholder, missing-glyph marker, hidden tracked change, comment,
   broken relationship, or stale external-link target remains.
5. The DOCX core metadata identifies the chapter, research project, evidence
   status, and build version.
6. Embedded or linked media resolve from the package; no local absolute development
   path is required to render the accepted file.

## Reproducibility and automated inspection

`scripts/render_publication_docx.py` is the governed renderer.
`scripts/inspect_publication_docx.py` must pass against the accepted source.

Two clean renders from the same accepted source and environment must be byte
identical after controlled package normalization. If a future renderer introduces
irreducible metadata differences, the manifest must identify and exclude only those
specific fields; content-part hashes must still match.

The automated inspection must cover:

- required OPC package parts;
- required Word styles;
- pagination controls;
- missing-glyph markers;
- footer page field and core metadata;
- nonempty heading population;
- table repeat-header and row-split controls;
- source-word-sequence preservation;
- document relationships and hyperlink targets;
- figure/media inventory; and
- absence of tracked changes, comments, and unresolved placeholders.

## Visual acceptance

The DOCX must be rendered through an accepted Word-compatible engine to a temporary
PDF solely for inspection. Every page must be reviewed at normal reading scale.
The visual-inspection record enumerates every page and records:

- title and running-matter correctness;
- heading hierarchy and page starts;
- paragraph flow and whitespace;
- table/figure clipping, splitting, and legibility;
- caption/source-note placement;
- evidence-grade visibility;
- missing glyphs or font substitution;
- accidental blank pages;
- orphan headings or stranded captions; and
- grayscale/print legibility on representative pages.

Quick Look thumbnails are useful during renderer development but do not replace the
final page-by-page review.

## Acceptance boundary

A chapter DOCX may be generated for testing before evidence acceptance, but it must
be labeled nonfinal and cannot replace the delivered file. Final chapter DOCX
replacement is authorized only after:

1. the chapter research packet and fresh semantic review are accepted;
2. the canonical database and chapter source agree;
3. the content-fidelity and automated gates pass;
4. two clean renders reconcile;
5. every rendered page is visually accepted; and
6. the publication manifest is sealed.

Completion of all chapter DOCX files does not by itself authorize the three final
PDFs; those remain governed by `publication_pdf_standard.md`.
