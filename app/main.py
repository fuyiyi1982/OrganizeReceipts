from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = DATA_DIR / "storage"
TMP_DIR = DATA_DIR / "tmp"

for directory in (DATA_DIR, STORAGE_DIR, TMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)


DATE_PATTERN = re.compile(r"(20\d{2})[-年/.]?(\d{1,2})[-月/.]?(\d{1,2})")
AMOUNT_PATTERNS = (
    re.compile(r"<TotalTax-includedAmount>([0-9]+(?:\.[0-9]+)?)</TotalTax-includedAmount>"),
    re.compile(r"<TotaltaxIncludedAmount>([0-9]+(?:\.[0-9]+)?)</TotaltaxIncludedAmount>"),
)


@dataclass
class InvoiceItem:
    pdf: Path
    date_value: date | None
    amount: float


@dataclass
class TripRecord:
    trip_dir: Path
    date_value: date
    invoice_count: int
    amount_total: float
    source_package: str
    source_segment: str


app = FastAPI(title="Toll Invoice Organizer")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def parse_date_from_text(text: str) -> date | None:
    match = DATE_PATTERN.search(text)
    if not match:
        return None
    year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_amount_from_xml(text: str) -> float:
    for pattern in AMOUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return 0.0


def extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def find_segment_dirs(root: Path) -> list[Path]:
    segments: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if (path / "apply.zip").exists() and (path / "detail.zip").exists():
            segments.append(path)
    return sorted(segments)


def collect_invoice_from_folder(folder: Path) -> InvoiceItem | None:
    pdf = next(iter(sorted(folder.glob("*.pdf"))), None)
    xml = next(iter(sorted(folder.glob("*.xml"))), None)
    if not pdf:
        return None

    date_value = None
    amount = 0.0
    if xml:
        raw = xml.read_text(encoding="utf-8", errors="ignore")
        date_value = parse_date_from_text(raw)
        amount = parse_amount_from_xml(raw)

    return InvoiceItem(pdf=pdf, date_value=date_value, amount=amount)


def extract_invoices(apply_zip: Path, working_dir: Path) -> list[InvoiceItem]:
    apply_dir = working_dir / "apply"
    extract_zip(apply_zip, apply_dir)

    inner_zips = sorted(apply_dir.glob("*.zip"))
    invoices: list[InvoiceItem] = []

    if inner_zips:
        for idx, inner in enumerate(inner_zips, start=1):
            inner_dir = working_dir / f"inner_{idx:03d}"
            extract_zip(inner, inner_dir)
            item = collect_invoice_from_folder(inner_dir)
            if item:
                invoices.append(item)
    else:
        item = collect_invoice_from_folder(apply_dir)
        if item:
            invoices.append(item)

    return invoices


def extract_itinerary_pdf(detail_zip: Path, working_dir: Path) -> Path | None:
    detail_dir = working_dir / "detail"
    extract_zip(detail_zip, detail_dir)
    trans_pdf = detail_dir / "trans.pdf"
    if trans_pdf.exists():
        return trans_pdf
    fallback = sorted(detail_dir.glob("*.pdf"))
    return fallback[0] if fallback else None


def get_next_trip_index(day_dir: Path) -> int:
    existing = []
    for directory in day_dir.glob("trip_*"):
        if directory.is_dir():
            match = re.match(r"trip_(\d+)$", directory.name)
            if match:
                existing.append(int(match.group(1)))
    return (max(existing) + 1) if existing else 1


def save_trip(
    trip_date: date,
    itinerary_pdf: Path | None,
    invoices: Iterable[InvoiceItem],
    source_package: str,
    source_segment: str,
) -> TripRecord:
    day_dir = STORAGE_DIR / f"{trip_date.year:04d}" / f"{trip_date.month:02d}" / f"{trip_date.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)

    trip_index = get_next_trip_index(day_dir)
    trip_dir = day_dir / f"trip_{trip_index:03d}"
    trip_dir.mkdir(parents=True, exist_ok=True)

    if itinerary_pdf and itinerary_pdf.exists():
        shutil.copy2(itinerary_pdf, trip_dir / "itinerary.pdf")

    invoice_count = 0
    total_amount = 0.0
    for idx, invoice in enumerate(invoices, start=1):
        target_name = f"invoice_{idx:03d}.pdf"
        shutil.copy2(invoice.pdf, trip_dir / target_name)
        invoice_count += 1
        total_amount += invoice.amount

    metadata = {
        "date": trip_date.isoformat(),
        "trip": trip_dir.name,
        "invoice_count": invoice_count,
        "total_amount": round(total_amount, 2),
        "source_package": source_package,
        "source_segment": source_segment,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (trip_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return TripRecord(
        trip_dir=trip_dir,
        date_value=trip_date,
        invoice_count=invoice_count,
        amount_total=round(total_amount, 2),
        source_package=source_package,
        source_segment=source_segment,
    )


def process_uploaded_zip(upload_zip_path: Path, workspace: Path) -> list[TripRecord]:
    package_name = upload_zip_path.stem
    extract_root = workspace / package_name
    extract_zip(upload_zip_path, extract_root)

    segment_dirs = find_segment_dirs(extract_root)
    records: list[TripRecord] = []

    for seg in segment_dirs:
        seg_workspace = workspace / f"work_{seg.name}_{int(datetime.now().timestamp() * 1000)}"
        seg_workspace.mkdir(parents=True, exist_ok=True)

        try:
            invoices = extract_invoices(seg / "apply.zip", seg_workspace)
            if not invoices:
                continue

            dates = [item.date_value for item in invoices if item.date_value is not None]
            if not dates:
                continue

            dominant = Counter(dates).most_common(1)[0][0]
            itinerary = extract_itinerary_pdf(seg / "detail.zip", seg_workspace)

            record = save_trip(
                trip_date=dominant,
                itinerary_pdf=itinerary,
                invoices=invoices,
                source_package=package_name,
                source_segment=seg.name,
            )
            records.append(record)
        finally:
            if seg_workspace.exists():
                shutil.rmtree(seg_workspace, ignore_errors=True)

    return records


def list_days() -> list[dict]:
    days: list[dict] = []
    for year_dir in sorted(STORAGE_DIR.glob("*"), reverse=True):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for month_dir in sorted(year_dir.glob("*"), reverse=True):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            for day_dir in sorted(month_dir.glob("*"), reverse=True):
                if not day_dir.is_dir() or not day_dir.name.isdigit():
                    continue

                trip_dirs = sorted([d for d in day_dir.glob("trip_*") if d.is_dir()])
                trip_count = len(trip_dirs)
                total = 0.0
                for trip in trip_dirs:
                    meta = trip / "metadata.json"
                    if meta.exists():
                        data = json.loads(meta.read_text(encoding="utf-8"))
                        total += float(data.get("total_amount", 0))

                days.append(
                    {
                        "year": int(year_dir.name),
                        "month": int(month_dir.name),
                        "day": int(day_dir.name),
                        "trip_count": trip_count,
                        "total_amount": round(total, 2),
                    }
                )
    return days


def get_day_dir(year: int, month: int, day: int) -> Path:
    day_dir = STORAGE_DIR / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"
    if not day_dir.exists():
        raise HTTPException(status_code=404, detail="指定日期不存在")
    return day_dir


def within_storage(path: Path) -> bool:
    try:
        path.resolve().relative_to(STORAGE_DIR.resolve())
        return True
    except ValueError:
        return False


@app.get("/", response_class=HTMLResponse)
def index(request: Request, message: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "days": list_days(),
            "message": message,
        },
    )


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> RedirectResponse:
    if not files:
        return RedirectResponse(url="/?message=未选择文件", status_code=303)

    workspace = TMP_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    workspace.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    trip_count = 0

    try:
        for upload_file in files:
            if not upload_file.filename:
                continue
            if not upload_file.filename.lower().endswith(".zip"):
                continue

            target = workspace / Path(upload_file.filename).name
            with target.open("wb") as f:
                shutil.copyfileobj(upload_file.file, f)

            records = process_uploaded_zip(target, workspace)
            processed_count += 1
            trip_count += len(records)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    msg = f"上传完成：处理压缩包 {processed_count} 个，新增行程 {trip_count} 段"
    return RedirectResponse(url=f"/?message={msg}", status_code=303)


@app.get("/day/{year}/{month}/{day}", response_class=HTMLResponse)
def view_day(year: int, month: int, day: int, request: Request) -> HTMLResponse:
    day_dir = get_day_dir(year, month, day)
    trips = []

    for trip_dir in sorted(day_dir.glob("trip_*")):
        if not trip_dir.is_dir():
            continue
        metadata_path = trip_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        itinerary = (trip_dir / "itinerary.pdf") if (trip_dir / "itinerary.pdf").exists() else None
        invoices = sorted(trip_dir.glob("invoice_*.pdf"))

        trips.append(
            {
                "name": trip_dir.name,
                "metadata": metadata,
                "itinerary": itinerary,
                "invoices": invoices,
            }
        )

    return templates.TemplateResponse(
        "day.html",
        {
            "request": request,
            "year": year,
            "month": month,
            "day": day,
            "trips": trips,
        },
    )


@app.get("/download/day/{year}/{month}/{day}")
def download_day(year: int, month: int, day: int) -> StreamingResponse:
    day_dir = get_day_dir(year, month, day)

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in day_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(day_dir))

    memory_file.seek(0)
    filename = f"{year:04d}-{month:02d}-{day:02d}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(memory_file, media_type="application/zip", headers=headers)


@app.get("/file/{year}/{month}/{day}/{trip}/{filename}")
def download_file(year: int, month: int, day: int, trip: str, filename: str) -> FileResponse:
    file_path = STORAGE_DIR / f"{year:04d}" / f"{month:02d}" / f"{day:02d}" / trip / filename
    if not file_path.exists() or not file_path.is_file() or not within_storage(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=file_path)
