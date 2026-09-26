import argparse
from pathlib import Path

from app.core.ingest import extract_pdf, save_chunks
from app.core.generation import caption_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", default="artifacts/index")
    args = parser.parse_args()

    chunks = extract_pdf(args.pdf, args.out)

    # Figure captions are optional because they incur vision-model calls.
    # They can be enabled later without changing the retrieval schema.
    figure_chunks = []
    for c in chunks:
        if c.modality == "figure" and c.image_path:
            caption = caption_figure(c.image_path)
            if caption:
                c.text += "\nVision description:\n" + caption
        figure_chunks.append(c)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_chunks(figure_chunks, str(out / "chunks.json"))

    print(f"Ingested {len(figure_chunks)} chunks")
    print(f"Text:   {sum(c.modality == 'text' for c in figure_chunks)}")
    print(f"Tables: {sum(c.modality == 'table' for c in figure_chunks)}")
    print(f"Figures:{sum(c.modality == 'figure' for c in figure_chunks)}")


if __name__ == "__main__":
    main()
