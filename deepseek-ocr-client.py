import io
import re
import argparse
import base64
import shutil
from pathlib import Path

from PIL import Image
from openai import OpenAI
from pdf2image import convert_from_path
from tqdm import tqdm


GROUNDING_PATTERN = re.compile(
    r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>'
)


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


def pil_image_to_base64(image: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


def image_to_pil(image_path: Path) -> Image.Image:
    return Image.open(image_path)


def crop_region(image: Image.Image, bbox: list[int]) -> Image.Image:
    w, h = image.size
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    return image.crop((x1, y1, x2, y2))


def parse_grounding_tags(text: str) -> list[dict]:
    results = []
    for match in GROUNDING_PATTERN.finditer(text):
        ref_type = match.group(1)
        bbox = [int(match.group(i)) for i in range(2, 6)]
        results.append({
            'type': ref_type,
            'bbox': bbox,
        })
    return results


def pdf_to_base64_pages(
    pdf_path: Path,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[tuple[str, str, Image.Image]]:
    kwargs = {}
    if first_page is not None:
        kwargs["first_page"] = first_page
    if last_page is not None:
        kwargs["last_page"] = last_page
    images = convert_from_path(pdf_path, **kwargs)
    pages = []
    for image in images:
        b64, media_type = pil_image_to_base64(image)
        pages.append((b64, media_type, image))
    return pages


def process_page_recursive(
    base64_image: str,
    media_type: str,
    pil_image: Image.Image,
    page_num: int,
    temp_dir: Path | None = None,
    image_prompt: str = "Parse the figure.",
) -> str:
    grounding_prompt = "<|grounding|>Convert the document to markdown."
    result = process_image(base64_image, media_type, grounding_prompt)

    lines = result.split('\n')
    output_lines = []
    fig_idx = 0

    for line in lines:
        output_lines.append(line)
        match = GROUNDING_PATTERN.search(line)
        if match and match.group(1) == 'image':
            bbox = [int(match.group(i)) for i in range(2, 6)]
            cropped = crop_region(pil_image, bbox)
            fig_idx += 1

            if temp_dir is not None:
                crop_path = temp_dir / f"page{page_num}_fig{fig_idx}.png"
                cropped.save(crop_path)

            crop_b64, crop_media = pil_image_to_base64(cropped)
            figure_text = process_image(crop_b64, crop_media, image_prompt)
            output_lines.append(figure_text)

    return '\n'.join(output_lines)


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
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="enable recursive scan: detect image regions via grounding and re-scan them",
    )
    parser.add_argument(
        "--image-prompt",
        default="Parse the figure.",
        help="prompt used for recursive image region scans (default: 'Parse the figure.')",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="keep temporary cropped images in a ./temp folder",
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

    temp_dir = None
    if args.recursive:
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)

    if suffix == ".pdf":
        pages = pdf_to_base64_pages(file_path, first_page, last_page)
        start = first_page or 1
        with open(output_path, "a") as f:
            for i, (b64, media_type, pil_image) in tqdm(enumerate(pages), total=len(pages), desc="Processing pages", unit="page"):
                if args.recursive:
                    result = process_page_recursive(b64, media_type, pil_image, start + i, temp_dir, args.image_prompt)
                else:
                    result = process_image(b64, media_type, args.prompt)
                f.write(result + "\n\n")
    else:
        b64, media_type = image_to_base64(file_path)
        if args.recursive:
            pil_image = image_to_pil(file_path)
            result = process_page_recursive(b64, media_type, pil_image, 1, temp_dir, args.image_prompt)
        else:
            result = process_image(b64, media_type, args.prompt)
        print(result)
        with open(output_path, "a") as f:
            f.write(result + "\n")

    if temp_dir and not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
