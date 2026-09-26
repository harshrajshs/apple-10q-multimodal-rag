from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fitz
import pdfplumber

from .models import Chunk


def _id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode()).hexdigest()[:16]


def _normalize(s: str) -> str:
    return " ".join((s or "").replace("\x00", " ").split())


def _clean_table_cell(value: str | None) -> str:
    """Normalize a single PDF-extracted table cell."""
    if value is None:
        return ""

    value = _normalize(str(value))

    # Standalone currency symbols are PDF layout artifacts.
    if value == "$":
        return ""

    return value.strip()


def _extract_table_context(page_text: str) -> tuple[str, list[str]]:
    """
    Extract table title, units and column labels from surrounding
    page text.

    SEC PDFs frequently split dates over multiple lines, e.g.:

        June 25,
        2022

    The order of repeated dates matters because the same date can
    appear under both the three-month and nine-month columns.
    """

    # ---------------------------------------------------------
    # Normalize PDF text
    # ---------------------------------------------------------

    text = page_text.replace("\x00", " ")

    # Join dates split across lines:
    #
    # June 25,
    # 2022
    #
    # -> June 25, 2022

    text = re.sub(
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
        r"\s+(\d{1,2}),\s*(\d{4})",
        r"\1 \2, \3",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize remaining whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # ---------------------------------------------------------
    # Table title
    # ---------------------------------------------------------

    title = ""

    title_patterns = [
        r"CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS",
        r"CONDENSED CONSOLIDATED BALANCE SHEETS",
        r"CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS",
    ]

    for pattern in title_patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            title = _normalize(match.group(0)).title()
            break

    # ---------------------------------------------------------
    # Units
    # ---------------------------------------------------------

    unit = ""

    unit_match = re.search(
        r"\((In millions.*?per share amounts)\)",
        text,
        flags=re.IGNORECASE,
    )

    if unit_match:
        unit = _normalize(unit_match.group(1))

    if title and unit:
        title = f"{title} — {unit}"
    elif unit and not title:
        title = unit

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------

    date_pattern = (
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
        r"\s+\d{1,2},\s+\d{4}"
    )

    # IMPORTANT:
    # Do NOT deduplicate these dates.
    #
    # The filing legitimately contains:
    #
    # June 25, 2022
    # June 26, 2021
    # June 25, 2022
    # June 26, 2021
    #
    # The repeated dates correspond to different reporting periods.

    dates = re.findall(
        date_pattern,
        text,
        flags=re.IGNORECASE,
    )

    dates = [
        _normalize(date)
        for date in dates
    ]

    columns: list[str] = []

    if len(dates) >= 4:

        columns = [
            f"Three Months Ended {dates[0]}",
            f"Three Months Ended {dates[1]}",
            f"Nine Months Ended {dates[2]}",
            f"Nine Months Ended {dates[3]}",
        ]

    return title, columns


def _table_to_text(
    table: list[list[str | None]],
    page_text: str = "",
) -> str:
    """
    Convert a PDF table into a retrieval-friendly semantic representation.

    When reliable column metadata is available from the page text,
    associate each extracted value with its corresponding column.
    Otherwise preserve the values in their original order.
    """

    title, columns = _extract_table_context(page_text)

    lines: list[str] = []

    if title:
        lines.append(f"TABLE: {title}")
        lines.append("")

    if columns:
        lines.append("COLUMNS:")

        for i, column in enumerate(columns, start=1):
            lines.append(
                f"  COLUMN {i}: {column}"
            )

        lines.append("")

    for row in table:

        cleaned = [
            _clean_table_cell(cell)
            for cell in row
        ]

        # Remove empty cells.
        cleaned = [
            cell
            for cell in cleaned
            if cell
        ]

        if not cleaned:
            continue

        label = cleaned[0]
        values = cleaned[1:]

        # Section headings.
        if not values:
            lines.append(
                f"TABLE SECTION: {label}"
            )
            continue

        lines.append(
            f"TABLE ROW: {label}"
        )

        # We only attach column labels when the number of
        # extracted values matches the number of columns.
        if columns and len(values) == len(columns):

            for column, value in zip(
                columns,
                values,
            ):
                lines.append(
                    f"  {column}: {value}"
                )

        else:

            lines.append(
                f"  VALUES: {' | '.join(values)}"
            )

        lines.append("")

    return "\n".join(lines).strip()


def _chunk_text(
    text: str,
    size: int = 1100,
    overlap: int = 180,
):
    words = text.split()

    if not words:
        return []

    out = []
    start = 0

    while start < len(words):

        end = min(
            len(words),
            start + size,
        )

        out.append(
            " ".join(words[start:end])
        )

        if end == len(words):
            break

        start = max(
            0,
            end - overlap,
        )

    return out


def extract_pdf(
    pdf_path: str,
    out_dir: str,
) -> list[Chunk]:

    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    images_dir = out_dir / "figures"
    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_id = hashlib.sha1(
        pdf_path.read_bytes()
    ).hexdigest()[:16]

    chunks: list[Chunk] = []

    # =========================================================
    # TEXT + FIGURES
    # =========================================================

    doc = fitz.open(pdf_path)

    # Keep page text so the exact PyMuPDF text used for normal
    # text chunks can also be used as table context.
    page_texts: dict[int, str] = {}

    for page_idx, page in enumerate(
        doc,
        start=1,
    ):

        page_text = page.get_text("text")

        page_texts[page_idx] = page_text

        normalized_text = _normalize(
            page_text
        )

        # -------------------------
        # Text chunks
        # -------------------------

        for i, part in enumerate(
            _chunk_text(normalized_text)
        ):

            chunks.append(
                Chunk(
                    chunk_id=_id(
                        document_id,
                        str(page_idx),
                        "text",
                        str(i),
                    ),
                    document_id=document_id,
                    page=page_idx,
                    modality="text",
                    text=part,
                    source=f"page:{page_idx}",
                )
            )

        # -------------------------
        # Figures
        # -------------------------

        images = page.get_images(
            full=True
        )

        if images:

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    1.5,
                    1.5,
                ),
                alpha=False,
            )

            img_path = (
                images_dir
                / f"page_{page_idx}.png"
            )

            pix.save(img_path)

            chunks.append(
                Chunk(
                    chunk_id=_id(
                        document_id,
                        str(page_idx),
                        "figure",
                    ),
                    document_id=document_id,
                    page=page_idx,
                    modality="figure",
                    text=(
                        f"Figure/image content on "
                        f"page {page_idx}. "
                        "The page image should be "
                        "interpreted visually when "
                        "answering."
                    ),
                    source=f"page:{page_idx}",
                    image_path=str(img_path),
                )
            )

    doc.close()

    # =========================================================
    # TABLES
    # =========================================================

    with pdfplumber.open(pdf_path) as pdf:

        for page_idx, page in enumerate(
            pdf.pages,
            start=1,
        ):

            tables = (
                page.extract_tables()
                or []
            )

            # IMPORTANT:
            # Use the PyMuPDF page text rather than
            # pdfplumber's page text for table context.
            page_text = page_texts.get(
                page_idx,
                "",
            )

            for table_idx, table in enumerate(
                tables
            ):

                table_text = _table_to_text(
                    table,
                    page_text=page_text,
                )

                if not table_text:
                    continue

                chunks.append(
                    Chunk(
                        chunk_id=_id(
                            document_id,
                            str(page_idx),
                            "table",
                            str(table_idx),
                        ),
                        document_id=document_id,
                        page=page_idx,
                        modality="table",
                        text=table_text,
                        source=(
                            f"page:{page_idx}"
                            f":table:{table_idx}"
                        ),
                    )
                )

    return chunks


def save_chunks(
    chunks: list[Chunk],
    path: str,
):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(path).write_text(
        json.dumps(
            [
                c.to_dict()
                for c in chunks
            ],
            indent=2,
        ),
        encoding="utf-8",
    )