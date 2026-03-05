#!/usr/bin/env python3
"""
multimodal.py - Multi-Modal Document Support

Extends ingest.py with support for:
- PDF extraction
- Image OCR (via pytesseract)
- Code file syntax-aware chunking
- Audio transcription (optional, via whisper)

All processing is local - no external APIs.
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()

# Supported extensions by type
TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}


def check_optional_deps() -> dict:
    """Check which optional dependencies are available."""
    deps = {}
    
    try:
        import pypdf
        deps["pdf"] = True
    except ImportError:
        deps["pdf"] = False
    
    try:
        import pytesseract
        from PIL import Image
        deps["ocr"] = True
    except ImportError:
        deps["ocr"] = False
    
    try:
        import whisper
        deps["audio"] = True
    except ImportError:
        deps["audio"] = False
    
    return deps


def extract_pdf_text(file_path: Path) -> Optional[str]:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(str(file_path))
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    except ImportError:
        print(f"⚠️ pypdf not installed. Install with: pip install pypdf")
        return None
    except Exception as e:
        print(f"⚠️ Error extracting PDF {file_path}: {e}")
        return None


def extract_image_text(file_path: Path) -> Optional[str]:
    """Extract text from an image using OCR."""
    try:
        import pytesseract
        from PIL import Image
        
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        
        return text.strip() if text.strip() else None
    
    except ImportError:
        print(f"⚠️ pytesseract/PIL not installed. Install with: pip install pytesseract pillow")
        print("  Also install tesseract: brew install tesseract (macOS)")
        return None
    except Exception as e:
        print(f"⚠️ Error OCR on {file_path}: {e}")
        return None


def chunk_code_file(content: str, file_ext: str) -> list[dict]:
    """
    Chunk a code file with syntax awareness.
    Returns list of chunks with metadata.
    """
    lines = content.split('\n')
    chunks = []
    
    # Language-specific patterns for function/class definitions
    patterns = {
        ".py": {
            "function": r"^(async\s+)?def\s+\w+",
            "class": r"^class\s+\w+",
        },
        ".js": {
            "function": r"^(async\s+)?function\s+\w+|^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",
            "class": r"^class\s+\w+",
        },
        ".ts": {
            "function": r"^(async\s+)?function\s+\w+|^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",
            "class": r"^class\s+\w+|^interface\s+\w+",
        },
        ".go": {
            "function": r"^func\s+",
            "class": r"^type\s+\w+\s+struct",
        },
        ".rs": {
            "function": r"^(pub\s+)?fn\s+",
            "class": r"^(pub\s+)?struct\s+|^(pub\s+)?impl\s+",
        },
        ".java": {
            "function": r"^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(",
            "class": r"^\s*(public|private)?\s*(abstract)?\s*class\s+",
        },
    }
    
    # Get patterns for this extension
    ext_patterns = patterns.get(file_ext, {
        "function": r"^(def|function|func|fn)\s+",
        "class": r"^(class|struct|type)\s+",
    })
    
    # Combine into a definition pattern
    combined = f"({ext_patterns.get('function', '')}|{ext_patterns.get('class', '')})"
    
    current_chunk_lines = []
    current_def = None
    current_start = 0
    
    for i, line in enumerate(lines):
        # Check if this line starts a new definition
        if re.match(combined, line.strip(), re.IGNORECASE):
            # Save previous chunk if exists
            if current_chunk_lines:
                chunks.append({
                    "content": '\n'.join(current_chunk_lines),
                    "definition": current_def,
                    "start_line": current_start,
                    "end_line": i - 1,
                })
            
            current_chunk_lines = [line]
            current_def = line.strip()[:100]  # First 100 chars of definition
            current_start = i
        else:
            current_chunk_lines.append(line)
    
    # Don't forget the last chunk
    if current_chunk_lines:
        chunks.append({
            "content": '\n'.join(current_chunk_lines),
            "definition": current_def,
            "start_line": current_start,
            "end_line": len(lines) - 1,
        })
    
    # If no definitions found, fall back to size-based chunking
    if len(chunks) <= 1 and len(content) > 1500:
        chunks = []
        chunk_size = 1000
        overlap = 200
        
        for start in range(0, len(content), chunk_size - overlap):
            chunk = content[start:start + chunk_size]
            if chunk.strip():
                chunks.append({
                    "content": chunk,
                    "definition": None,
                    "start_line": content[:start].count('\n'),
                    "end_line": content[:start + chunk_size].count('\n'),
                })
    
    return chunks


def process_multimodal_file(file_path: Path) -> Optional[dict]:
    """
    Process a file based on its type.
    Returns dict with content and metadata, or None if unsupported.
    """
    ext = file_path.suffix.lower()
    
    result = {
        "path": file_path,
        "extension": ext,
        "type": "unknown",
        "content": None,
        "chunks": None,
        "metadata": {},
    }
    
    # Text files
    if ext in TEXT_EXTENSIONS:
        try:
            result["content"] = file_path.read_text(encoding="utf-8")
            result["type"] = "text"
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            return None
    
    # Code files
    elif ext in CODE_EXTENSIONS:
        try:
            content = file_path.read_text(encoding="utf-8")
            result["content"] = content
            result["type"] = "code"
            result["chunks"] = chunk_code_file(content, ext)
            result["metadata"]["language"] = ext[1:]  # Remove leading dot
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            return None
    
    # PDF files
    elif ext in PDF_EXTENSIONS:
        content = extract_pdf_text(file_path)
        if content:
            result["content"] = content
            result["type"] = "pdf"
        else:
            return None
    
    # Image files
    elif ext in IMAGE_EXTENSIONS:
        content = extract_image_text(file_path)
        if content:
            result["content"] = content
            result["type"] = "image"
            result["metadata"]["ocr"] = True
        else:
            return None
    
    else:
        return None
    
    return result


def find_multimodal_documents(data_dir: Path) -> list[Path]:
    """Find all supported documents including multi-modal types."""
    all_extensions = TEXT_EXTENSIONS | CODE_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS
    
    documents = []
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in all_extensions:
            documents.append(path)
    
    return sorted(documents)


def get_supported_extensions() -> dict:
    """Get list of supported extensions by type."""
    deps = check_optional_deps()
    
    supported = {
        "text": list(TEXT_EXTENSIONS),
        "code": list(CODE_EXTENSIONS),
    }
    
    if deps["pdf"]:
        supported["pdf"] = list(PDF_EXTENSIONS)
    else:
        supported["pdf_unavailable"] = "Install pypdf: pip install pypdf"
    
    if deps["ocr"]:
        supported["image"] = list(IMAGE_EXTENSIONS)
    else:
        supported["image_unavailable"] = "Install pytesseract and pillow"
    
    return supported


if __name__ == "__main__":
    print("Multi-Modal Document Support")
    print("-" * 40)
    
    # Check dependencies
    deps = check_optional_deps()
    print("\nDependency Status:")
    print(f"  PDF support (pypdf): {'✓' if deps['pdf'] else '✗'}")
    print(f"  OCR support (pytesseract): {'✓' if deps['ocr'] else '✗'}")
    print(f"  Audio support (whisper): {'✓' if deps['audio'] else '✗'}")
    
    # Show supported extensions
    print("\nSupported Extensions:")
    for ext_type, exts in get_supported_extensions().items():
        if isinstance(exts, list):
            print(f"  {ext_type}: {', '.join(exts)}")
        else:
            print(f"  {ext_type}: {exts}")
    
    # Find documents
    if DATA_DIR.exists():
        docs = find_multimodal_documents(DATA_DIR)
        print(f"\nFound {len(docs)} document(s) in {DATA_DIR}")
        for doc in docs[:10]:
            print(f"  - {doc.relative_to(DATA_DIR)}")
        if len(docs) > 10:
            print(f"  ... and {len(docs) - 10} more")
    
    print("\n✓ Multi-modal module ready")
