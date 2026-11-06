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
    import re  # 引入正则表达式库

    # 1. 解析 PDF
    text = pymupdf4llm.to_markdown(file_path, pages=list(range(5, 165)))

    # 2. 核心清洗逻辑（忽略大小写，替换所有 kuka）
    text = re.sub(r'(?i)kuka', '工业机器人', text)
    text = text.replace('库卡', '工业机器人')

    # 3. 替换其他专有名词
    text = text.replace('KR C4', '控制系统')
    text = text.replace('smartPAD', '示教器')
    text = text.replace('KRL', '机器人编程语言')

    # 4. 清理无意义的页眉页脚（可选，但建议加上，能减少噪音）
    text = re.sub(r'发布日期:.*?2011.*?\n', '', text)
    text = re.sub(r'<mark>.*?机器人编程 1.*?</mark>', '', text)

    return text

def _parse_with_docling(file_path: str) -> str:
    from docling.document_converter import DocumentConverter
    result = DocumentConverter().convert(file_path)
    return result.document.export_to_markdown()


def _parse_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 修改为你真实的 PDF 路径，注意 Windows 路径的斜杠
    pdf_path = r"D:\shiyueyirag\初级编程.pdf"

    print("=" * 50)
    print("开始解析:", pdf_path)
    print("=" * 50)
    try:
        md_text = parse_document(pdf_path)

        # 【关键】把解析结果存成 .md 文件，方便你查看效果，也方便后续切片
        output_path = r"D:\shiyueyirag\kuka_parsed.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        print(f"解析成功！结果已保存至: {output_path}")
    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}")