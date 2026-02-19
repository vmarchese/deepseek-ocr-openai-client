# deepseek-ocr-openai-client

CLI tool for OCR on images and PDFs using [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) via an OpenAI-compatible API.

## Prerequisites

- Python 3.14+
- An OpenAI-compatible API endpoint serving `deepseek-ai/DeepSeek-OCR` (e.g. vLLM)
- `poppler` (required by `pdf2image` for PDF conversion)

Set the endpoint and API key via environment variables:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your-key"
```

## Installation

```bash
pip install -e .
```

## Usage

```
python deepseek-ocr-client.py [options] <image_or_pdf>
```

### Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--page-range FROM-TO` | `-r` | Range of PDF pages to process (e.g. `1-5`) | All pages |
| `--prompt TEXT` | `-p` | Prompt sent to the model | `Convert the document to markdown.` |
| `--output FILE` | `-o` | Output file path | `<input>.md` |
| `--recursive` | | Enable recursive scan: detect image regions via grounding and re-scan them | Off |
| `--image-prompt TEXT` | | Prompt used for recursive image region scans | `Parse the figure.` |
| `--keep-temp` | | Keep temporary cropped images in a `./temp` folder | Off |
| `--help` | `-h` | Show help message and exit | |

### Examples

OCR an image:

```bash
python deepseek-ocr-client.py scan.png
```

Convert an entire PDF to markdown:

```bash
python deepseek-ocr-client.py document.pdf
```

Convert pages 3 through 8 with a custom prompt:

```bash
python deepseek-ocr-client.py \
  -r 3-8 \
  -p "Extract all tables as markdown" \
  -o tables.md \
  document.pdf
```


### DeepSeek OCR Prompts 

See [DeepSeek-OCR on github](https://github.com/deepseek-ai/DeepSeek-OCR)

```
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
```


### Example 
Scanning a page of a PDF document with text and images and converting in markdown with a description of the images in the text can be done with:

```
python deepseek-ocr-client.py \
   -r <page range> \
   --recursive \
   --keep-temp \
   -p "<|grounding|>Convert the document to markdown." \
   --image-prompt "Describe this image in detail." \
   -o <output file> \
   <input file>
```
