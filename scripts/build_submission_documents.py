"""Compile and package the checked-in MAP documents; no scientific computations."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "submission_MAP"
QA = PAPER / "document_build"
TECTONIC = os.environ.get("TECTONIC", "tectonic")

def read(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")

def table_bodies(text):
    return re.findall(r"\\begin\{tabular\}[\s\S]*?\\end\{tabular\}", text)

def compile_document(path):
    result = subprocess.run(
        [TECTONIC, "--keep-logs", "--keep-intermediates", path.name],
        cwd=path.parent, check=True, text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(result.stdout, flush=True)
    log = path.with_suffix(".log")
    content = read(log) if log.exists() else result.stdout
    failures = re.findall(
        r"(?:LaTeX Warning:.*(?:undefined|multiply defined)|"
        r"Overfull \\[hv]box.*|There were undefined references)", content
    )
    if failures:
        raise RuntimeError("Document warnings require review: " + repr(failures))
    return {"compiler": "Tectonic 0.17.0", "undefined_references": 0,
            "overfull_boxes": 0}

def callout_order(text, prefix):
    labels = re.findall(r"\\label\{(" + prefix + r":[^}]+)\}", text)
    positions = [text.find("\\ref{" + label + "}") for label in labels]
    if any(x < 0 for x in positions) or positions != sorted(positions):
        raise ValueError("Nonconsecutive " + prefix + " first-reference order")
    return labels

def pdf_checks(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    texts = [p.extract_text() or "" for p in reader.pages]
    if any("??" in t for t in texts):
        raise ValueError("Unresolved PDF reference: " + str(path))
    return {"pages": len(reader.pages), "unresolved_reference_markers": 0}, texts

def previews(path, name):
    import pymupdf
    document = pymupdf.open(path)
    directory = QA / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(document), 16):
        count = min(16, len(document) - start)
        sheet = pymupdf.open()
        canvas = sheet.new_page(width=880, height=((count + 3) // 4) * 310)
        for index in range(count):
            row, col = divmod(index, 4)
            rect = pymupdf.Rect(col * 220 + 6, row * 310 + 18,
                               (col + 1) * 220 - 6, (row + 1) * 310 - 6)
            canvas.show_pdf_page(rect, document, start + index)
            canvas.insert_text((col * 220 + 6, row * 310 + 12),
                               f"{name}: page {start + index + 1}", fontsize=8)
        canvas.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(
            directory / f"{name}-{start + 1:02d}.png")
        sheet.close()
    if name == "main":
        for index in [0, 9, 10, len(document)-3, len(document)-2, len(document)-1]:
            if 0 <= index < len(document):
                document[index].get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(
                    directory / f"main-full-{index + 1:02d}.png")
    document.close()

def make_zip(path, entries):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in entries:
            archive.write(source, name)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError("Corrupt ZIP: " + str(path))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads(read(QA / "expected_inputs.json"))
    text = read(PAPER / "main.tex")
    supplement = read(PAPER / "supplementary.tex")
    actual = Counter(table_bodies(text) + table_bodies(supplement))
    wanted = Counter(expected["source_table_bodies"] +
                     expected["supplementary_table_bodies"])
    if actual != wanted:
        raise ValueError("Document table bodies differ from the reviewed text revision")
    original_numeric = expected.get("original_numeric_result_table_bodies", [])
    if original_numeric and any(actual[body] < count for body, count in Counter(original_numeric).items()):
        raise ValueError("An original numerical result-table body changed")
    for record in expected["scientific_input_git_blobs"]:
        actual_sha = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{record['path']}"],
            cwd=ROOT, text=True).strip()
        if actual_sha != record["sha"]:
            raise ValueError("Scientific input changed: " + record["path"])
    figures = callout_order(text, "fig")
    tables = callout_order(text, "tab")
    abstract = read(OUT / "abstract.txt").strip()
    if len(abstract.split()) != expected["expected_abstract_words"]:
        raise ValueError("Abstract word count mismatch")
    build = {}
    for stem in ["main", "supplementary"]:
        build[stem] = compile_document(PAPER / f"{stem}.tex")
        detail, _ = pdf_checks(PAPER / f"{stem}.pdf")
        build[stem].update(detail)
        previews(PAPER / f"{stem}.pdf", stem)
    build["cover_letter"] = compile_document(OUT / "cover_letter.tex")
    detail, _ = pdf_checks(OUT / "cover_letter.pdf")
    build["cover_letter"].update(detail)
    if detail["pages"] != 1:
        raise ValueError("Cover letter should occupy one page")
    previews(OUT / "cover_letter.pdf", "cover")
    shutil.copyfile(PAPER / "main.pdf", OUT / "manuscript_MAP.pdf")
    shutil.copyfile(PAPER / "supplementary.pdf", OUT / "ESM_1.pdf")
    cover_source = read(OUT / "cover_letter.tex")
    cover_text = cover_source.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    cover_text = re.sub(r"\\url\{([^}]+)\}", r"\1", cover_text)
    cover_text = cover_text.replace("$^\\circ$C", "°C").replace("\\\\", "\n")
    cover_text = re.sub(r"\n{3,}", "\n\n", cover_text).strip()
    (OUT / "cover_letter.txt").write_text(cover_text + "\n", encoding="utf-8")
    sources = []
    for name in ["main.tex", "supplementary.tex", "refs.bib",
                 "sn-jnl.cls", "sn-basic.bst", "main.bbl", "supplementary.bbl"]:
        path = PAPER / name
        if not path.exists():
            raise FileNotFoundError(path)
        sources.append((path, name))
    sources.extend((p, "figures/" + p.name)
                   for p in sorted((PAPER / "figures").glob("Fig*.pdf")))
    if len([p for p, _ in sources if p.parent.name == "figures"]) != 12:
        raise ValueError("Expected twelve submission figure PDFs")
    make_zip(OUT / "manuscript_source.zip", sources)
    report = {
        "new_experiments": False,
        "source_publication_commit": expected["source_publication_commit"],
        "compiled_documents": build,
        "table_bodies_preserved": sum(actual.values()),
        "original_numerical_table_bodies_preserved": len(original_numeric),
        "text_revision_base_commit": expected.get("text_revision_base_commit"),
        "unchanged_scientific_input_blobs": len(expected["scientific_input_git_blobs"]),
        "abstract_words": len(abstract.split()),
        "main_figure_first_reference_order": figures,
        "main_table_first_reference_order": tables,
        "submission_figure_count": 12,
        "final_author_review": "pending",
        "submission_status": "NOT SUBMITTED",
        "scope": "Text revision, document compilation and preserved result-table checks only; no new uncertainty estimates."
    }
    (QA / "validation.json").write_text(json.dumps(report, indent=2) + "\n",
                                      encoding="utf-8")
    package = [(p, "source/" + name) for p, name in sources]
    for name in ["manuscript_MAP.pdf", "ESM_1.pdf", "cover_letter.pdf",
                 "cover_letter.tex", "cover_letter.txt", "abstract.txt",
                 "submission_metadata.json", "SUBMISSION_CHECKLIST.md"]:
        package.append((OUT / name, name))
    package.extend([(PAPER / "REVISION_NOTES.md", "REVISION_NOTES.md"),
                    (QA / "validation.json", "DOCUMENT_VALIDATION.json")])
    make_zip(OUT / "MAP_submission_package.zip", package)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()
