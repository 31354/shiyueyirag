import os


def parse_document(file_path: str) -> str:
    """统一入口：根据扩展名调用对应解析器，返回 Markdown 文本。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext in (".docx", ".doc", ".pptx", ".xlsx", ".html", ".htm"):
        return _parse_with_docling(file_path)
    elif ext in (".md", ".txt"):
        return _parse_text(file_path)
    else:
        raise ValueError(f"不支持的格式: {ext}")


def _parse_pdf(file_path: str) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(file_path)


def _parse_with_docling(file_path: str) -> str:
    from docling.document_converter import DocumentConverter
    result = DocumentConverter().convert(file_path)
    return result.document.export_to_markdown()


def _parse_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    for name in ["sample.pdf", "sample.docx"]:
        path = os.path.join(BASE_DIR, name)
        print("=" * 50)
        print("文件:", name)
        print("=" * 50)
        try:
            print(parse_document(path))
        except Exception as e:
            print(f"[失败] {type(e).__name__}: {e}")