# app.py — Full backend for the Analytics Dashboard
# Flask + SQLAlchemy (Postgres/MySQL/SQLite) + Celery (worker & beat) + Redis + JWT + SMTP alerts
# Compatible with your React frontend expecting fields: date, value, revenue, category

import os
import io
import json
import logging
from datetime import datetime, timedelta, date

from functools import wraps
from flask import Flask, request, jsonify, g
from flask_cors import CORS

import pandas as pd
import redis
import jwt  # PyJWT

from celery import Celery
from celery.schedules import crontab

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, text

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analytics-backend")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (ENV-friendly). Reasonable defaults for local dev.
# ──────────────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = {"csv"}

SECRET_KEY = os.getenv("SECRET_KEY", "your_very_secret_key")
JWT_EXP_HOURS = int(os.getenv("JWT_EXP_HOURS", "6"))

# Database: set DATABASE_URL for Postgres/MySQL in prod.
# Examples:
#  - postgresql+psycopg2://user:pass@localhost:5432/analytics
#  - mysql+pymysql://user:pass@localhost:3306/analytics
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///analytics.db")
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Redis (cache & Celery broker/result)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "1"))
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# SMTP (optional; alerts are skipped if not configured)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
ALERT_TO = os.getenv("ALERT_TO")  # comma-separated list allowed

# ──────────────────────────────────────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../build", static_url_path="/")
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    SQLALCHEMY_DATABASE_URI=DATABASE_URL,
    SQLALCHEMY_TRACK_MODIFICATIONS=SQLALCHEMY_TRACK_MODIFICATIONS,
    CELERY_BROKER_URL=CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND=CELERY_RESULT_BACKEND,
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CORS(app, resources={r"/api/*": {"origins": "*"}}, expose_headers=["Content-Disposition"])

# ──────────────────────────────────────────────────────────────────────────────
# SQLAlchemy
# ──────────────────────────────────────────────────────────────────────────────
db = SQLAlchemy(app)

class IngestRecord(db.Model):
    __tablename__ = "ingest_records"
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, index=True, nullable=False)
    value = db.Column(db.Float, default=0.0)
    revenue = db.Column(db.Float, nullable=True)
    category = db.Column(db.String(128), index=True)
    lat = db.Column(db.Float, nullable=True)   # optional for maps
    lng = db.Column(db.Float, nullable=True)   # optional for maps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class CategoryAggregate(db.Model):
    __tablename__ = "category_aggregates"
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(128), index=True)
    period = db.Column(db.String(16), index=True)   # 'daily'|'weekly'|'monthly'
    period_start = db.Column(db.Date, index=True)
    total_items = db.Column(db.Float, default=0.0)
    total_revenue = db.Column(db.Float, default=0.0)
    avg_revenue = db.Column(db.Float, default=0.0)
    growth_rate = db.Column(db.Float, default=0.0)  # pct vs previous period
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class Anomaly(db.Model):
    __tablename__ = "anomalies"
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, index=True, nullable=False)
    category = db.Column(db.String(128), index=True)
    value = db.Column(db.Float, nullable=True)  # anomalous revenue value
    description = db.Column(db.String(512))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

with app.app_context():
    db.create_all()

# ──────────────────────────────────────────────────────────────────────────────
# Celery (worker + beat)
# ──────────────────────────────────────────────────────────────────────────────
celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"])
celery.conf.update(result_backend=app.config["CELERY_RESULT_BACKEND"])

# Daily at 01:00 (server time): refresh cache and send alerts on failure/anomalies
celery.conf.beat_schedule = {
    "daily-refresh-01:00": {
        "task": "app.refresh_cache_from_db",
        "schedule": crontab(minute=0, hour=1),
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Redis (cache)
# ──────────────────────────────────────────────────────────────────────────────
r = None
try:
    r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    r.ping()
    if not r.exists("dashboard_data"):
        r.set("dashboard_data", json.dumps({"raw_data": [], "summary_data": [], "anomaly_data": []}))
    log.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
except Exception as e:
    log.error(f"Redis connection failed: {e}")
    r = None

# ──────────────────────────────────────────────────────────────────────────────
# Mock Users / JWT Auth
# ──────────────────────────────────────────────────────────────────────────────
users = {
    "admin": {"password": "admin", "role": "admin"},
    "viewer": {"password": "viewer", "role": "viewer"},
}

def make_token(username: str, role: str) -> str:
    payload = {"username": username, "role": role, "exp": datetime.utcnow() + timedelta(hours=JWT_EXP_HOURS)}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token if isinstance(token, str) else token.decode("utf-8")

def parse_bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None

def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = parse_bearer_token()
        if not token:
            return jsonify({"error": "Token is missing!"}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            g.username = data.get("username")
            g.user_role = data.get("role")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Token is invalid."}), 401
        return f(*args, **kwargs)
    return wrapper

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ──────────────────────────────────────────────────────────────────────────────
# Email (optional)
# ──────────────────────────────────────────────────────────────────────────────
def send_alert_email(body: str, subject: str = "Analytics Alert"):
    # Gracefully no-op if SMTP not configured
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_TO]):
        log.warning("SMTP not configured; skipping email alert.")
        return
    import smtplib
    from email.message import EmailMessage

    recipients = [x.strip() for x in str(ALERT_TO).split(",") if x.strip()]
    if not recipients:
        log.warning("No ALERT_TO recipients; skipping email.")
        return

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    log.info(f"Alert email sent to {recipients}")

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, de-duplicate, normalize, and add period columns."""
    # Required columns: date, value, revenue, category (after mapping)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Numeric coercion
    df["value"] = pd.to_numeric(df.get("value", 0), errors="coerce").fillna(0).astype(float)
    if "revenue" not in df.columns:
        df["revenue"] = float("nan")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    # Categories
    df["category"] = df.get("category", "Unknown").fillna("Unknown").astype(str).str.strip()

    # De-duplication (by date+category)
    df = df.sort_values(["date"]).drop_duplicates(subset=["date", "category"], keep="last")

    # Period columns
    dt = pd.to_datetime(df["date"])
    df["day"] = dt.dt.date
    # ISO week start (Mon) – use W-MON for clarity
    df["week"] = dt.dt.to_period("W-MON").apply(lambda p: p.start_time.date())
    df["month"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())

    return df

def _compute_aggregates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return daily/weekly/monthly aggregate DataFrames with growth_rate for monthly."""
    def agg_for(period_col, period_name):
        g = df.groupby([period_col, "category"]).agg(
            total_items=("value", "sum"),
            total_revenue=("revenue", "sum"),
            avg_revenue=("revenue", "mean"),
        ).reset_index().rename(columns={period_col: "period_start"})
        g["period"] = period_name
        return g

    daily = agg_for("day", "daily")
    weekly = agg_for("week", "weekly")
    monthly = agg_for("month", "monthly").sort_values(["category", "period_start"])

    # Growth rate (period-over-period) on total_revenue
    monthly["growth_rate"] = monthly.groupby("category")["total_revenue"].pct_change().fillna(0.0)
    return daily, weekly, monthly

def _detect_anomalies(df: pd.DataFrame) -> list[dict]:
    anomalies = []
    if df["revenue"].notna().any():
        mean_rev = df["revenue"].mean(skipna=True)
        std_rev = df["revenue"].std(skipna=True)
        if std_rev and std_rev > 0:
            z = (df["revenue"] - mean_rev) / std_rev
            flagged = df[z.abs() > 2]
            for _, row in flagged.iterrows():
                anomalies.append({
                    "date": pd.to_datetime(row["date"]).date(),
                    "category": row.get("category"),
                    "value": float(row["revenue"]) if pd.notna(row["revenue"]) else None,
                    "description": "Revenue is significantly different from the mean.",
                })
    return anomalies

def _persist_ingest(df: pd.DataFrame):
    """Bulk insert cleaned rows into IngestRecord."""
    rows = []
    for _, row in df.iterrows():
        rows.append(IngestRecord(
            date=pd.to_datetime(row["date"]).date(),
            value=float(row["value"]) if pd.notna(row["value"]) else 0.0,
            revenue=float(row["revenue"]) if pd.notna(row["revenue"]) else None,
            category=row.get("category"),
            lat=float(row["lat"]) if "lat" in df.columns and pd.notna(row["lat"]) else None,
            lng=float(row["lng"]) if "lng" in df.columns and pd.notna(row["lng"]) else None,
        ))
    db.session.bulk_save_objects(rows)
    db.session.commit()

def _persist_aggregates(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame):
    """Replace aggregates for overlapping periods to avoid duplicates."""
    def upsert(df: pd.DataFrame, period: str):
        # Delete existing rows for the period_start values present to avoid dupes
        if not len(df):
            return
        starts = set(df["period_start"].tolist())
        db.session.query(CategoryAggregate).filter(
            CategoryAggregate.period == period,
            CategoryAggregate.period_start.in_(starts)
        ).delete(synchronize_session=False)

        rows = []
        for _, r in df.iterrows():
            rows.append(CategoryAggregate(
                category=r["category"],
                period=period,
                period_start=r["period_start"],
                total_items=float(r["total_items"]) if pd.notna(r["total_items"]) else 0.0,
                total_revenue=float(r["total_revenue"]) if pd.notna(r["total_revenue"]) else 0.0,
                avg_revenue=float(r["avg_revenue"]) if pd.notna(r["avg_revenue"]) else 0.0,
                growth_rate=float(r["growth_rate"]) if "growth_rate" in r and pd.notna(r["growth_rate"]) else 0.0,
            ))
        db.session.bulk_save_objects(rows)
        db.session.commit()

    upsert(daily, "daily")
    upsert(weekly, "weekly")
    upsert(monthly, "monthly")

def _persist_anomalies(anoms: list[dict]):
    if not anoms:
        return
    # (Optional) dedupe by date+category+value
    for a in anoms:
        db.session.add(Anomaly(
            date=a["date"],
            category=a.get("category"),
            value=a.get("value"),
            description=a.get("description"),
        ))
    db.session.commit()

def _build_dashboard_payload(limit_raw: int = 1000) -> dict:
    """Build the shape expected by the frontend from DB (and return it)."""
    # raw_data
    q = db.session.query(IngestRecord).order_by(IngestRecord.date.asc(), IngestRecord.id.asc())
    if limit_raw:
        q = q.limit(limit_raw)
    raw = []
    for rec in q:
        raw.append({
            "date": rec.date.isoformat(),
            "value": float(rec.value) if rec.value is not None else 0.0,
            "revenue": float(rec.revenue) if rec.revenue is not None else None,
            "category": rec.category,
            # optional for maps:
            "lat": float(rec.lat) if rec.lat is not None else None,
            "lng": float(rec.lng) if rec.lng is not None else None,
        })

    # summary_data: per-category totals/average across all time
    sums = db.session.query(
        IngestRecord.category,
        func.sum(IngestRecord.value),
        func.avg(IngestRecord.revenue)
    ).group_by(IngestRecord.category).all()
    summary = []
    for category, total_items, avg_rev in sums:
        summary.append({
            "category": category,
            "total_items": float(total_items) if total_items is not None else 0.0,
            "average_revenue": float(avg_rev) if avg_rev is not None else 0.0
        })

    # anomalies: last 180 days
    horizon = date.today() - timedelta(days=180)
    anoms = db.session.query(Anomaly).filter(Anomaly.date >= horizon).order_by(Anomaly.date.desc()).limit(500).all()
    anomaly_data = []
    for a in anoms:
        anomaly_data.append({
            "date": a.date.isoformat(),
            "category": a.category,
            "value": float(a.value) if a.value is not None else None,
            "description": a.description,
        })

    return {"raw_data": raw, "summary_data": summary, "anomaly_data": anomaly_data}

# ──────────────────────────────────────────────────────────────────────────────
# Celery Tasks
# ──────────────────────────────────────────────────────────────────────────────
@celery.task(bind=True, name="app.process_csv_file")
def process_csv_file(self, csv_content: str, column_mapping: dict):
    """
    Background: clean/standardize, persist to DB, compute aggregates & anomalies,
    refresh Redis cache, and send email if anomalies detected.
    """
    try:
        with app.app_context():
            df = pd.read_csv(io.StringIO(csv_content))

            # Validate mapping keys
            required = {"date_col", "item_count_col", "revenue_col", "category_col"}
            if not isinstance(column_mapping, dict) or not required.issubset(set(column_mapping.keys())):
                return {"status": "error", "message": "Invalid column_mapping: required keys missing."}

            # Ensure required columns exist in CSV
            mapped = {
                "date": column_mapping["date_col"],
                "value": column_mapping["item_count_col"],
                "revenue": column_mapping["revenue_col"],
                "category": column_mapping["category_col"],
            }
            missing = [col for col in mapped.values() if col not in df.columns]
            if missing:
                return {"status": "error", "message": f"CSV missing columns: {missing}"}

            # Standardize names
            df = df.rename(columns={
                mapped["date"]: "date",
                mapped["value"]: "value",
                mapped["revenue"]: "revenue",
                mapped["category"]: "category",
            })

            # Clean & enrich
            df = _normalize_and_enrich(df)

            # Persist to DB
            _persist_ingest(df)

            # Aggregates
            daily, weekly, monthly = _compute_aggregates(df)
            _persist_aggregates(daily, weekly, monthly)

            # Anomalies
            anomalies = _detect_anomalies(df)
            _persist_anomalies(anomalies)

            # Refresh Redis cache
            refresh_cache_from_db.apply_async(countdown=0)

            # Email if anomalies exist
            if anomalies:
                body = "Anomalies detected:\n" + "\n".join(
                    [f"- {a['date']} | {a.get('category')} | revenue={a.get('value')}" for a in anomalies[:20]]
                )
                send_alert_email(body, subject="Analytics: Anomalies Detected")

            return {"status": "success", "message": "File processed and dashboard updated."}
    except Exception as e:
        log.exception("CSV processing task failed.")
        send_alert_email(f"CSV processing failed: {e}", subject="Analytics: Processing Failed")
        return {"status": "error", "message": f"Error processing CSV: {e}"}

@celery.task(bind=True, name="app.refresh_cache_from_db")
def refresh_cache_from_db(self):
    """Rebuild Redis cache from DB; email on failure."""
    try:
        with app.app_context():
            payload = _build_dashboard_payload()
            if r:
                r.set("dashboard_data", json.dumps(payload))
                log.info("dashboard_data refreshed in Redis.")
            else:
                log.warning("Redis not available; cache not refreshed.")
            return {"status": "ok"}
    except Exception as e:
        log.exception("Cache refresh failed.")
        send_alert_email(f"Cache refresh failed: {e}", subject="Analytics: Cache Refresh Failed")
        return {"status": "error", "message": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    # DB check (lightweight)
    db_ok = True
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return jsonify({"status": "ok", "redis": bool(r), "db": db_ok}), 200

@app.route("/api/login", methods=["POST"])
def login():
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload."}), 400
    username = (body or {}).get("username")
    password = (body or {}).get("password")
    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    user = users.get(username)
    if user and user.get("password") == password:
        token = make_token(username, user["role"])
        return jsonify({"message": "Login successful", "token": token, "role": user["role"]}), 200
    return jsonify({"error": "Invalid username or password"}), 401

@app.route("/api/data", methods=["GET"])
@token_required
def get_data():
    if g.user_role not in ["admin", "viewer"]:
        return jsonify({"error": "Access denied"}), 403
    # Prefer cache; if absent, rebuild from DB on the fly
    try:
        if r:
            data_str = r.get("dashboard_data")
            if data_str:
                return jsonify(json.loads(data_str)), 200
        # Fallback: build from DB
        payload = _build_dashboard_payload()
        if r:
            r.set("dashboard_data", json.dumps(payload))
        return jsonify(payload), 200
    except Exception:
        log.exception("Failed to provide /api/data")
        return jsonify({"error": "Failed to retrieve data."}), 500

@app.route("/api/upload", methods=["POST"])
@token_required
def upload_file():
    if g.user_role != "admin":
        return jsonify({"error": "Permission denied. Only admins can upload files."}), 403
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Please upload a CSV."}), 400

    try:
        csv_content = file.read().decode("utf-8")
    except Exception as e:
        log.error(f"Failed to read uploaded file: {e}")
        return jsonify({"error": "Failed to read uploaded file."}), 400

    # Optional column mapping (defaults align with your React)
    try:
        mapping_raw = request.form.get("column_mapping")
        if mapping_raw:
            column_mapping = json.loads(mapping_raw)
        else:
            column_mapping = {
                "date_col": "date",
                "item_count_col": "value",  # IMPORTANT: frontend uses 'value' for count
                "revenue_col": "revenue",
                "category_col": "category",
            }
    except Exception as e:
        log.error(f"Invalid column mapping: {e}")
        return jsonify({"error": "Invalid column mapping format."}), 400

    try:
        task = process_csv_file.delay(csv_content, column_mapping)
        log.info(f"Started CSV processing task id={task.id}")
        return jsonify({"message": "File uploaded and processing started.", "task_id": task.id}), 202
    except Exception:
        log.exception("Failed to start processing task")
        return jsonify({"error": "Failed to start processing task."}), 500

@app.route("/api/export", methods=["GET"])
@token_required
def export_summary_csv():
    """Export category-level monthly aggregates as CSV."""
    if g.user_role not in ["admin", "viewer"]:
        return jsonify({"error": "Access denied"}), 403

    # Get latest month present in aggregates
    latest = db.session.query(
        func.max(CategoryAggregate.period_start)
    ).filter(CategoryAggregate.period == "monthly").scalar()

    q = db.session.query(CategoryAggregate).filter(CategoryAggregate.period == "monthly")
    if latest:
        q = q.filter(CategoryAggregate.period_start == latest)

    rows = q.order_by(CategoryAggregate.category.asc()).all()

    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["category", "period", "period_start", "total_items", "total_revenue", "avg_revenue", "growth_rate"])
    for r_ in rows:
        w.writerow([
            r_.category,
            r_.period,
            r_.period_start.isoformat() if r_.period_start else "",
            f"{r_.total_items:.6f}",
            f"{r_.total_revenue:.6f}",
            f"{r_.avg_revenue:.6f}",
            f"{r_.growth_rate:.6f}",
        ])
    buf.seek(0)
    return (buf.read(), 200, {
        "Content-Type": "text/csv",
        "Content-Disposition": "attachment; filename=summary_monthly.csv"
    })

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=os.getenv("FLASK_DEBUG", "True") == "True")
