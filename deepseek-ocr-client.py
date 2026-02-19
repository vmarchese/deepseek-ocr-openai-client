import io
import argparse
import base64
from pathlib import Path

from openai import OpenAI
from pdf2image import convert_from_path
from tqdm import tqdm


def process_image(base64_image: str, media_type: str = "image/png", prompt: str = "Convert the document to markdown.") -> str:
    client = OpenAI()

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-OCR",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=2028,
        temperature=0.0,
        extra_body={
            "skip_special_tokens": False,
            "chat_template": "{%- if messages[0]['role'] == 'system' -%}{%- set system_message = messages[0]['content'] -%}{%- set messages = messages[1:] -%}{%- else -%}{% set system_message = '' -%}{%- endif -%}{{ bos_token + system_message }}{%- for message in messages -%}{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{%- endif -%}{{ message['content'] }}{%- endfor -%}",
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],
            },
        },
    )

    return response.choices[0].message.content


def image_to_base64(image_path: Path) -> tuple[str, str]:
    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    base64_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return base64_image, media_type


def pdf_to_base64_pages(pdf_path: Path, first_page: int | None = None, last_page: int | None = None) -> list[tuple[str, str]]:
    kwargs = {}
    if first_page is not None:
        kwargs["first_page"] = first_page
    if last_page is not None:
        kwargs["last_page"] = last_page
    images = convert_from_path(pdf_path, **kwargs)
    pages = []
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        pages.append((b64, "image/png"))
    return pages


def main():
    parser = argparse.ArgumentParser(
        description="OCR images and PDFs using DeepSeek-OCR via the OpenAI-compatible API."
    )
    parser.add_argument(
        "input",
        help="path to an image or PDF file",
    )
    parser.add_argument(
        "-r", "--page-range",
        metavar="FROM-TO",
        help="range of PDF pages to process, e.g. 1-5",
    )
    parser.add_argument(
        "-p", "--prompt",
        default="Convert the document to markdown.",
        help="prompt sent to the model (default: 'Convert the document to markdown.')",
    )
    parser.add_argument(
        "-o", "--output",
        help="output file path (default: <input>.md)",
    )
    args = parser.parse_args()

    file_path = Path(args.input)
    if not file_path.exists():
        parser.error(f"file not found: {file_path}")

    output_path = Path(args.output) if args.output else file_path.with_suffix(".md")
    suffix = file_path.suffix.lower()

    first_page = None
    last_page = None
    if args.page_range:
        try:
            first_page, last_page = (int(x) for x in args.page_range.split("-"))
        except ValueError:
            parser.error(f"invalid page range: '{args.page_range}' (expected FROM-TO, e.g. 1-5)")

    if suffix == ".pdf":
        pages = pdf_to_base64_pages(file_path, first_page, last_page)
        start = first_page or 1
        with open(output_path, "a") as f:
            for i, (b64, media_type) in tqdm(enumerate(pages), total=len(pages), desc="Processing pages", unit="page"):
                result = process_image(b64, media_type, args.prompt)
                f.write(result + "\n\n")
    else:
        b64, media_type = image_to_base64(file_path)
        result = process_image(b64, media_type, args.prompt)
        print(result)
        with open(output_path, "a") as f:
            f.write(result + "\n")


if __name__ == "__main__":
    main()
