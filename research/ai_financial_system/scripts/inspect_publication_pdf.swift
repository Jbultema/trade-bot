#!/usr/bin/env swift

import AppKit
import Foundation
import PDFKit

struct PageAudit {
    let index: Int
    let label: String
    let widthPoints: Double
    let heightPoints: Double
    let extractedCharacters: Int
    let annotationCount: Int
    let linkAnnotationCount: Int
    let sampledNonWhiteShare: Double
    let renderedFile: String
    let sampledRenderFingerprint: String

    func jsonObject() -> [String: Any] {
        [
            "page_index": index,
            "page_label": label,
            "width_points": widthPoints,
            "height_points": heightPoints,
            "extracted_characters": extractedCharacters,
            "annotation_count": annotationCount,
            "link_annotation_count": linkAnnotationCount,
            "sampled_nonwhite_share": sampledNonWhiteShare,
            "rendered_file": renderedFile,
            "sampled_render_fingerprint": sampledRenderFingerprint,
        ]
    }
}

enum InspectionError: Error, CustomStringConvertible {
    case usage
    case missingInput(String)
    case unreadablePDF(String)
    case renderFailure(Int)
    case outputFailure(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: inspect_publication_pdf.swift INPUT.pdf OUTPUT_DIR AUDIT.json"
        case .missingInput(let path):
            return "input PDF does not exist: \(path)"
        case .unreadablePDF(let path):
            return "PDFKit could not open: \(path)"
        case .renderFailure(let page):
            return "could not render page \(page)"
        case .outputFailure(let path):
            return "could not write output: \(path)"
        }
    }
}

func outlineCount(_ outline: PDFOutline?) -> Int {
    guard let outline else { return 0 }
    var count = 0
    for index in 0..<outline.numberOfChildren {
        if let child = outline.child(at: index) {
            count += 1 + outlineCount(child)
        }
    }
    return count
}

func fnv1a64(_ values: [UInt8]) -> String {
    var hash: UInt64 = 14_695_981_039_346_656_037
    for value in values {
        hash ^= UInt64(value)
        hash &*= 1_099_511_628_211
    }
    return String(format: "%016llx", hash)
}

func renderPage(
    _ page: PDFPage,
    pageIndex: Int,
    outputDirectory: URL,
    scale: CGFloat = 1.5
) throws -> (String, Double, String) {
    let bounds = page.bounds(for: .mediaBox)
    let pixelWidth = max(1, Int(ceil(bounds.width * scale)))
    let pixelHeight = max(1, Int(ceil(bounds.height * scale)))

    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixelWidth,
        pixelsHigh: pixelHeight,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        throw InspectionError.renderFailure(pageIndex + 1)
    }

    NSGraphicsContext.saveGraphicsState()
    guard let graphics = NSGraphicsContext(bitmapImageRep: bitmap) else {
        NSGraphicsContext.restoreGraphicsState()
        throw InspectionError.renderFailure(pageIndex + 1)
    }
    NSGraphicsContext.current = graphics
    let context = graphics.cgContext
    context.setFillColor(NSColor.white.cgColor)
    context.fill(CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
    context.saveGState()
    context.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: context)
    context.restoreGState()
    graphics.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()

    let fileName = String(format: "page_%04d.png", pageIndex + 1)
    let outputURL = outputDirectory.appendingPathComponent(fileName)
    guard
        let png = bitmap.representation(using: .png, properties: [:])
    else {
        throw InspectionError.renderFailure(pageIndex + 1)
    }
    do {
        try png.write(to: outputURL, options: .atomic)
    } catch {
        throw InspectionError.outputFailure(outputURL.path)
    }

    let sampleStride = max(1, min(pixelWidth, pixelHeight) / 120)
    var sampled = 0
    var nonWhite = 0
    var fingerprintBytes: [UInt8] = []
    fingerprintBytes.reserveCapacity(16_384)

    for y in stride(from: 0, to: pixelHeight, by: sampleStride) {
        for x in stride(from: 0, to: pixelWidth, by: sampleStride) {
            guard let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else {
                continue
            }
            let red = UInt8(max(0, min(255, Int(round(color.redComponent * 255)))))
            let green = UInt8(max(0, min(255, Int(round(color.greenComponent * 255)))))
            let blue = UInt8(max(0, min(255, Int(round(color.blueComponent * 255)))))
            sampled += 1
            if red < 248 || green < 248 || blue < 248 {
                nonWhite += 1
            }
            fingerprintBytes.append(red)
            fingerprintBytes.append(green)
            fingerprintBytes.append(blue)
        }
    }

    let share = sampled == 0 ? 0.0 : Double(nonWhite) / Double(sampled)
    return (fileName, share, fnv1a64(fingerprintBytes))
}

func inspect() throws {
    guard CommandLine.arguments.count == 4 else {
        throw InspectionError.usage
    }

    let input = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
    let outputDirectory = URL(fileURLWithPath: CommandLine.arguments[2])
        .standardizedFileURL
    let auditOutput = URL(fileURLWithPath: CommandLine.arguments[3])
        .standardizedFileURL

    guard FileManager.default.fileExists(atPath: input.path) else {
        throw InspectionError.missingInput(input.path)
    }
    guard let document = PDFDocument(url: input) else {
        throw InspectionError.unreadablePDF(input.path)
    }

    try FileManager.default.createDirectory(
        at: outputDirectory,
        withIntermediateDirectories: true
    )
    try FileManager.default.createDirectory(
        at: auditOutput.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )

    var pageAudits: [PageAudit] = []
    var extractedDocumentText = ""
    var fingerprints: [String: [Int]] = [:]

    for index in 0..<document.pageCount {
        guard let page = document.page(at: index) else {
            throw InspectionError.renderFailure(index + 1)
        }
        let bounds = page.bounds(for: .mediaBox)
        let text = page.string ?? ""
        extractedDocumentText.append(text)
        extractedDocumentText.append("\n\n")

        let rendered = try renderPage(
            page,
            pageIndex: index,
            outputDirectory: outputDirectory
        )
        fingerprints[rendered.2, default: []].append(index + 1)
        let annotations = page.annotations
        let links = annotations.filter { annotation in
            annotation.action is PDFActionURL || annotation.action is PDFActionGoTo
        }
        pageAudits.append(
            PageAudit(
                index: index + 1,
                label: page.label ?? String(index + 1),
                widthPoints: Double(bounds.width),
                heightPoints: Double(bounds.height),
                extractedCharacters: text.count,
                annotationCount: annotations.count,
                linkAnnotationCount: links.count,
                sampledNonWhiteShare: rendered.1,
                renderedFile: rendered.0,
                sampledRenderFingerprint: rendered.2
            )
        )
    }

    let duplicateRenderGroups = fingerprints
        .filter { $0.value.count > 1 }
        .map { ["fingerprint": $0.key, "pages": $0.value] as [String: Any] }
        .sorted {
            (($0["pages"] as? [Int])?.first ?? 0)
                < (($1["pages"] as? [Int])?.first ?? 0)
        }

    let blankCandidatePages = pageAudits
        .filter { $0.extractedCharacters == 0 && $0.sampledNonWhiteShare < 0.002 }
        .map(\.index)
    let lowTextPages = pageAudits
        .filter { $0.extractedCharacters < 40 }
        .map(\.index)

    let attributes = document.documentAttributes ?? [:]
    let metadata = Dictionary(
        uniqueKeysWithValues: attributes.map {
            (String(describing: $0.key), String(describing: $0.value))
        }
    )
    let report: [String: Any] = [
        "input_pdf": input.path,
        "page_count": document.pageCount,
        "is_encrypted": document.isEncrypted,
        "is_locked": document.isLocked,
        "allows_copying": document.allowsCopying,
        "allows_printing": document.allowsPrinting,
        "outline_item_count": outlineCount(document.outlineRoot),
        "metadata": metadata,
        "extracted_document_characters": extractedDocumentText.count,
        "blank_candidate_pages": blankCandidatePages,
        "low_text_pages": lowTextPages,
        "duplicate_render_groups": duplicateRenderGroups,
        "pages": pageAudits.map { $0.jsonObject() },
        "font_embedding_check": "requires a separate low-level PDF font audit",
        "visual_acceptance": "pending human page-by-page review",
    ]

    let reportData = try JSONSerialization.data(
        withJSONObject: report,
        options: [.prettyPrinted, .sortedKeys]
    )
    do {
        try reportData.write(to: auditOutput, options: .atomic)
    } catch {
        throw InspectionError.outputFailure(auditOutput.path)
    }

    let textOutput = auditOutput
        .deletingPathExtension()
        .appendingPathExtension("txt")
    do {
        try extractedDocumentText.write(
            to: textOutput,
            atomically: true,
            encoding: .utf8
        )
    } catch {
        throw InspectionError.outputFailure(textOutput.path)
    }

    print(auditOutput.path)
}

do {
    try inspect()
} catch {
    FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
    exit(1)
}
