from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(input_path: Path) -> str:
	"""
	Extract text from a PDF file using pypdf.
	Skips pages that fail to extract clean text.
	"""
	reader = PdfReader(str(input_path))
	chunks: list[str] = []
	for page in reader.pages:
		try:
			txt = page.extract_text() or ""
		except Exception:
			txt = ""
		chunks.append(txt)
	return "\n".join(chunks).strip()


def main() -> None:
	parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
	parser.add_argument("--in", dest="input_path", required=True, help="Path to input PDF")
	parser.add_argument("--out", dest="output_path", required=True, help="Path to output text file")
	args = parser.parse_args()

	in_path = Path(args.input_path)
	out_path = Path(args.output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	text = extract_pdf_text(in_path)
	out_path.write_text(text, encoding="utf-8")
	print(f"Wrote extracted text to: {out_path}")


if __name__ == "__main__":
	main()


