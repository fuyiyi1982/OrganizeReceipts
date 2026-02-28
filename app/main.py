from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import shutil
import sqlite3
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = DATA_DIR / "storage"
TMP_DIR = DATA_DIR / "tmp"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"

for directory in (DATA_DIR, STORAGE_DIR, TMP_DIR, UPLOADS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


DATE_PATTERN = re.compile(r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})")
TAG_DATE_PATTERNS = (
    re.compile(r"<IssueTime>(.*?)</IssueTime>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<StartDatesOfPassage>(.*?)</StartDatesOfPassage>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<EndDatesOfPassage>(.*?)</EndDatesOfPassage>", re.IGNORECASE | re.DOTALL),
)
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
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "replace-this-in-production"),
    max_age=60 * 60 * 24 * 7,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invite_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                created_by INTEGER NOT NULL,
                used_by INTEGER,
                created_at TEXT NOT NULL,
                used_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upload_records (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                original_name TEXT NOT NULL,
                stored_name TEXT,
                sha256 TEXT,
                size INTEGER,
                uploaded_at TEXT NOT NULL,
                status TEXT NOT NULL,
                trip_count INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                deleted_at TEXT
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$", 2)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return secrets.compare_digest(actual, expected)


def ensure_admin_user() -> None:
    with db_conn() as conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
        if row:
            return
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
            (
                "admin",
                hash_password("fyy525200"),
                1,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


init_db()
ensure_admin_user()


def parse_date_from_text(text: str) -> date | None:
    def build_date(year: int, month: int, day: int) -> date | None:
        current_year = datetime.now().year
        if year < 2018 or year > current_year + 1:
            return None
        try:
            return date(year, month, day)
        except ValueError:
            return None

    def parse_one(value: str) -> date | None:
        value = value.strip()

        sep_match = DATE_PATTERN.search(value)
        if sep_match:
            return build_date(int(sep_match.group(1)), int(sep_match.group(2)), int(sep_match.group(3)))

        compact_match = re.search(r"(20\d{2})(\d{2})(\d{2})", value)
        if compact_match:
            return build_date(int(compact_match.group(1)), int(compact_match.group(2)), int(compact_match.group(3)))

        return None

    candidates: list[date] = []

    for pattern in TAG_DATE_PATTERNS:
        for matched in pattern.findall(text):
            parsed = parse_one(matched)
            if parsed:
                candidates.append(parsed)

    if candidates:
        return Counter(candidates).most_common(1)[0][0]

    fallback_candidates: list[date] = []
    for match in DATE_PATTERN.finditer(text):
        try_date = build_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if try_date:
            fallback_candidates.append(try_date)

    if not fallback_candidates:
        return None

    return Counter(fallback_candidates).most_common(1)[0][0]


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


def user_storage_root(user_id: int) -> Path:
    root = STORAGE_DIR / f"user_{user_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def user_upload_root(user_id: int) -> Path:
    root = UPLOADS_DIR / f"user_{user_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_trip(
    storage_root: Path,
    trip_date: date,
    itinerary_pdf: Path | None,
    invoices: Iterable[InvoiceItem],
    source_package: str,
    source_segment: str,
) -> TripRecord:
    day_dir = storage_root / f"{trip_date.year:04d}" / f"{trip_date.month:02d}" / f"{trip_date.day:02d}"
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


def process_uploaded_zip(upload_zip_path: Path, workspace: Path, storage_root: Path) -> list[TripRecord]:
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
                storage_root=storage_root,
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


def list_days(storage_root: Path) -> list[dict]:
    days: list[dict] = []
    if not storage_root.exists():
        return days

    for year_dir in sorted(storage_root.glob("*"), reverse=True):
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


def get_day_dir(storage_root: Path, year: int, month: int, day: int) -> Path:
    day_dir = storage_root / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"
    if not day_dir.exists():
        raise HTTPException(status_code=404, detail="指定日期不存在")
    return day_dir


def within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_user_by_id(user_id: int) -> dict | None:
    with db_conn() as conn:
        row = conn.execute("SELECT id, username, is_admin, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_current_user(request: Request) -> dict | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(int(user_id))


def ensure_login(request: Request) -> dict | None:
    user = get_current_user(request)
    if not user:
        return None
    return user


def list_uploads_for_user(user_id: int) -> list[dict]:
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, original_name, stored_name, sha256, size, uploaded_at, status, trip_count, message, deleted_at
            FROM upload_records
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
            """,
            (user_id,),
        ).fetchall()

    upload_root = user_upload_root(user_id)
    result = []
    for row in rows:
        item = dict(row)
        stored_name = item.get("stored_name")
        item["file_exists"] = bool(stored_name and (upload_root / stored_name).exists())
        result.append(item)
    return result


def render_page(request: Request, template: str, context: dict) -> HTMLResponse:
    merged = {"request": request, "current_user": get_current_user(request)}
    merged.update(context)
    return templates.TemplateResponse(template, merged)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, message: str | None = None) -> HTMLResponse:
    user = ensure_login(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    storage_root = user_storage_root(user["id"])
    return render_page(
        request,
        "index.html",
        {
            "days": list_days(storage_root),
            "uploads": list_uploads_for_user(user["id"]),
            "message": message,
        },
    )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, message: str | None = None) -> HTMLResponse:
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)
    return render_page(request, "login.html", {"message": message})


@app.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> RedirectResponse:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, is_admin FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

    if not row or not verify_password(password, row["password_hash"]):
        return RedirectResponse(url=f"/login?message={quote('用户名或密码错误')}", status_code=303)

    request.session["user_id"] = int(row["id"])
    return RedirectResponse(url="/", status_code=303)


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, message: str | None = None) -> HTMLResponse:
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)
    return render_page(request, "register.html", {"message": message})


@app.post("/register")
def register_submit(
    username: str = Form(...),
    password: str = Form(...),
    invite_code: str = Form(...),
) -> RedirectResponse:
    username = username.strip()
    invite_code = invite_code.strip().upper()

    if len(username) < 3:
        return RedirectResponse(url=f"/register?message={quote('用户名至少3个字符')}", status_code=303)
    if len(password) < 6:
        return RedirectResponse(url=f"/register?message={quote('密码至少6位')}", status_code=303)

    with db_conn() as conn:
        existed = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existed:
            return RedirectResponse(url=f"/register?message={quote('用户名已存在')}", status_code=303)

        invite = conn.execute(
            "SELECT id FROM invite_codes WHERE code = ? AND used_by IS NULL",
            (invite_code,),
        ).fetchone()
        if not invite:
            return RedirectResponse(url=f"/register?message={quote('邀请码无效或已使用')}", status_code=303)

        cur = conn.execute(
            "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, 0, ?)",
            (username, hash_password(password), datetime.now().isoformat(timespec="seconds")),
        )
        user_id = int(cur.lastrowid)

        conn.execute(
            "UPDATE invite_codes SET used_by = ?, used_at = ? WHERE id = ?",
            (user_id, datetime.now().isoformat(timespec="seconds"), int(invite["id"])),
        )
        conn.commit()

    return RedirectResponse(url=f"/login?message={quote('注册成功，请登录')}", status_code=303)


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/admin/invites", response_class=HTMLResponse)
def admin_invites(request: Request, message: str | None = None) -> HTMLResponse:
    user = ensure_login(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not user.get("is_admin"):
        return RedirectResponse(url=f"/?message={quote('仅管理员可访问')}", status_code=303)

    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT i.code, i.created_at, i.used_at, u.username AS used_by_name
            FROM invite_codes i
            LEFT JOIN users u ON i.used_by = u.id
            ORDER BY i.created_at DESC
            """
        ).fetchall()

    return render_page(
        request,
        "admin_invites.html",
        {"invites": [dict(r) for r in rows], "message": message},
    )


@app.post("/admin/invites/create")
def admin_create_invites(request: Request, count: int = Form(1)) -> RedirectResponse:
    user = ensure_login(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not user.get("is_admin"):
        return RedirectResponse(url=f"/?message={quote('仅管理员可操作')}", status_code=303)

    count = max(1, min(count, 100))
    created = 0
    codes: list[str] = []

    with db_conn() as conn:
        for _ in range(count):
            code = f"INV-{secrets.token_hex(4).upper()}"
            try:
                conn.execute(
                    "INSERT INTO invite_codes (code, created_by, created_at) VALUES (?, ?, ?)",
                    (code, user["id"], datetime.now().isoformat(timespec="seconds")),
                )
                created += 1
                codes.append(code)
            except sqlite3.IntegrityError:
                continue
        conn.commit()

    message = f"已生成邀请码 {created} 个：{'，'.join(codes[:10])}"
    return RedirectResponse(url=f"/admin/invites?message={quote(message)}", status_code=303)


@app.post("/upload")
async def upload(request: Request, files: list[UploadFile] = File(...)) -> RedirectResponse:
    user = ensure_login(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    if not files:
        return RedirectResponse(url=f"/?message={quote('未选择文件')}", status_code=303)

    user_id = int(user["id"])
    uploads_root = user_upload_root(user_id)
    storage_root = user_storage_root(user_id)

    workspace = TMP_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    workspace.mkdir(parents=True, exist_ok=True)

    with db_conn() as conn:
        rows = conn.execute(
            "SELECT sha256 FROM upload_records WHERE user_id = ? AND status = 'processed' AND sha256 IS NOT NULL",
            (user_id,),
        ).fetchall()
    active_hashes = {r["sha256"] for r in rows}

    processed_count = 0
    trip_count = 0
    duplicate_count = 0

    try:
        for upload_file in files:
            if not upload_file.filename:
                continue
            if not upload_file.filename.lower().endswith(".zip"):
                continue

            upload_id = uuid.uuid4().hex
            safe_name = Path(upload_file.filename).name
            stored_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{upload_id}_{safe_name}"
            target = uploads_root / stored_name

            with target.open("wb") as f:
                shutil.copyfileobj(upload_file.file, f)

            file_hash = sha256_of_file(target)
            file_size = target.stat().st_size

            if file_hash in active_hashes:
                duplicate_count += 1
                target.unlink(missing_ok=True)
                with db_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO upload_records (id, user_id, original_name, stored_name, sha256, size, uploaded_at, status, trip_count, message)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            upload_id,
                            user_id,
                            safe_name,
                            "",
                            file_hash,
                            file_size,
                            datetime.now().isoformat(timespec="seconds"),
                            "duplicate",
                            0,
                            "重复上传，已跳过处理",
                        ),
                    )
                    conn.commit()
                continue

            records = process_uploaded_zip(target, workspace, storage_root)
            processed_count += 1
            trip_count += len(records)
            active_hashes.add(file_hash)

            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO upload_records (id, user_id, original_name, stored_name, sha256, size, uploaded_at, status, trip_count, message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        upload_id,
                        user_id,
                        safe_name,
                        stored_name,
                        file_hash,
                        file_size,
                        datetime.now().isoformat(timespec="seconds"),
                        "processed",
                        len(records),
                        "处理成功",
                    ),
                )
                conn.commit()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    msg = f"上传完成：处理压缩包 {processed_count} 个，新增行程 {trip_count} 段，重复跳过 {duplicate_count} 个"
    return RedirectResponse(url=f"/?message={quote(msg)}", status_code=303)


@app.get("/day/{year}/{month}/{day}", response_class=HTMLResponse)
def view_day(year: int, month: int, day: int, request: Request) -> HTMLResponse:
    user = ensure_login(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    storage_root = user_storage_root(int(user["id"]))
    day_dir = get_day_dir(storage_root, year, month, day)
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

    return render_page(
        request,
        "day.html",
        {
            "year": year,
            "month": month,
            "day": day,
            "trips": trips,
        },
    )


@app.post("/days/clear")
def clear_days(request: Request, selected_day: list[str] = Form(default=[])) -> RedirectResponse:
    user = ensure_login(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    storage_root = user_storage_root(int(user["id"]))
    cleared = 0

    for item in selected_day:
        parts = item.split("-")
        if len(parts) != 3:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue

        day_dir = storage_root / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"
        if day_dir.exists() and day_dir.is_dir() and within_root(day_dir, storage_root):
            shutil.rmtree(day_dir, ignore_errors=True)
            cleared += 1

    msg = f"已批量清空 {cleared} 个日期"
    return RedirectResponse(url=f"/?message={quote(msg)}", status_code=303)


@app.post("/day/{year}/{month}/{day}/clear")
def clear_day(year: int, month: int, day: int, request: Request) -> RedirectResponse:
    user = ensure_login(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    storage_root = user_storage_root(int(user["id"]))
    day_dir = storage_root / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"
    if day_dir.exists() and day_dir.is_dir() and within_root(day_dir, storage_root):
        shutil.rmtree(day_dir, ignore_errors=True)
        msg = f"已清空 {year:04d}-{month:02d}-{day:02d} 的数据"
    else:
        msg = f"{year:04d}-{month:02d}-{day:02d} 不存在"
    return RedirectResponse(url=f"/?message={quote(msg)}", status_code=303)


@app.get("/download/day/{year}/{month}/{day}")
def download_day(year: int, month: int, day: int, request: Request) -> StreamingResponse:
    user = ensure_login(request)
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")

    storage_root = user_storage_root(int(user["id"]))
    day_dir = get_day_dir(storage_root, year, month, day)

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
def download_file(year: int, month: int, day: int, trip: str, filename: str, request: Request) -> FileResponse:
    user = ensure_login(request)
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")

    storage_root = user_storage_root(int(user["id"]))
    file_path = storage_root / f"{year:04d}" / f"{month:02d}" / f"{day:02d}" / trip / filename
    if not file_path.exists() or not file_path.is_file() or not within_root(file_path, storage_root):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=file_path)


@app.get("/uploads/{upload_id}/download")
def download_upload(upload_id: str, request: Request) -> FileResponse:
    user = ensure_login(request)
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")

    user_id = int(user["id"])
    with db_conn() as conn:
        row = conn.execute(
            "SELECT original_name, stored_name, status FROM upload_records WHERE id = ? AND user_id = ?",
            (upload_id, user_id),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="上传记录不存在")

    stored_name = row["stored_name"] or ""
    if not stored_name or row["status"] == "deleted":
        raise HTTPException(status_code=404, detail="该记录没有可下载源文件")

    uploads_root = user_upload_root(user_id)
    file_path = uploads_root / stored_name
    if not file_path.exists() or not file_path.is_file() or not within_root(file_path, uploads_root):
        raise HTTPException(status_code=404, detail="源文件不存在")

    filename = row["original_name"] or file_path.name
    return FileResponse(path=file_path, filename=filename)


@app.post("/uploads/{upload_id}/delete")
def delete_upload(upload_id: str, request: Request) -> RedirectResponse:
    user = ensure_login(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    user_id = int(user["id"])
    uploads_root = user_upload_root(user_id)
    deleted_name = ""

    with db_conn() as conn:
        row = conn.execute(
            "SELECT id, original_name, stored_name, status FROM upload_records WHERE id = ? AND user_id = ?",
            (upload_id, user_id),
        ).fetchone()

        if not row or row["status"] == "deleted":
            return RedirectResponse(url=f"/?message={quote('未找到可删除的上传记录')}", status_code=303)

        stored_name = row["stored_name"] or ""
        if stored_name:
            file_path = uploads_root / stored_name
            if file_path.exists() and file_path.is_file() and within_root(file_path, uploads_root):
                file_path.unlink(missing_ok=True)
            deleted_name = row["original_name"] or stored_name

        conn.execute(
            "UPDATE upload_records SET status = 'deleted', deleted_at = ? WHERE id = ? AND user_id = ?",
            (datetime.now().isoformat(timespec="seconds"), upload_id, user_id),
        )
        conn.commit()

    msg = f"已删除上传文件：{deleted_name or upload_id}"
    return RedirectResponse(url=f"/?message={quote(msg)}", status_code=303)
