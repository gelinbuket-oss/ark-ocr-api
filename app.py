# -*- coding: utf-8 -*-
"""ARK OCR Pipeline Application

This Flask application powers a modern processing pipeline that accepts
user uploads, extracts text with OCR, enriches the content with OpenAI,
and notifies the user via WhatsApp.  It exposes a contemporary landing
page plus lightweight JSON endpoints that power a realtime status UI.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

load_dotenv()

try:  # pragma: no cover - optional dependency guard
    import openai
except Exception:  # pragma: no cover - handled gracefully at runtime
    openai = None  # type: ignore

import requests

from ocr_utils import extract_text


def create_app() -> Flask:
    """Application factory so the app can be created for tests or wsgi."""
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "ark-ocr-dev-key"),
        SQLALCHEMY_DATABASE_URI=os.environ.get(
            "DATABASE_URL", "sqlite:///ark_ocr_platform.db"
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=os.path.join(app.root_path, "uploads"),
        MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_SIZE", 20)) * 1024 * 1024,
    )

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    register_routes(app)
    configure_logging(app)
    register_shutdown_hook()

    return app


db = SQLAlchemy()


class ProcessingJob(db.Model):
    __tablename__ = "processing_jobs"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    phone_number = db.Column(db.String(32), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(32), nullable=False, default="queued")
    ocr_text = db.Column(db.Text, nullable=True)
    ai_summary = db.Column(db.Text, nullable=True)
    whatsapp_message_id = db.Column(db.String(128), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.public_id,
            "status": self.status,
            "phone_number": self.phone_number,
            "original_filename": self.original_filename,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "ocr_excerpt": (self.ocr_text or "")[:400],
            "ai_summary": self.ai_summary,
            "whatsapp_message_id": self.whatsapp_message_id,
            "error": self.error_message,
        }


EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKER_THREADS", 4)))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_PROMPT = os.environ.get(
    "OPENAI_PROMPT",
    "You are an assistant that receives raw OCR text. Summarise key facts, "
    "extract important numbers, and produce a concise Turkish response."
)

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_API_TOKEN")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_API_VERSION = os.environ.get("WHATSAPP_API_VERSION", "v20.0")
WHATSAPP_SENDER_NAME = os.environ.get("WHATSAPP_SENDER_NAME", "ARK OCR Platform")
PRESERVE_UPLOADS = os.environ.get("PRESERVE_UPLOADS", "false").lower() == "true"


def configure_logging(app: Flask) -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level)
    app.logger.setLevel(level)
    app.logger.info("Logging initialised at %s", level)


def register_shutdown_hook() -> None:
    def _shutdown() -> None:
        EXECUTOR.shutdown(wait=False)

    atexit.register(_shutdown)


def register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        recent_jobs = (
            ProcessingJob.query.order_by(ProcessingJob.created_at.desc())
            .limit(6)
            .all()
        )
        return render_template("index.html", recent_jobs=recent_jobs)

    @app.route("/upload", methods=["POST"])
    def upload():
        if "file" not in request.files:
            return jsonify({"error": "Dosya bulunamadı."}), 400

        upload_file: FileStorage = request.files["file"]  # type: ignore[assignment]
        phone_number = request.form.get("phone")

        if not phone_number:
            return jsonify({"error": "Lütfen WhatsApp numaranızı girin."}), 400

        if upload_file.filename == "":
            return jsonify({"error": "Gönderilecek dosya seçilmedi."}), 400

        if not allowed_file(upload_file.filename):
            return (
                jsonify({"error": "PDF, PNG veya JPG dosyaları destekleniyor."}),
                400,
            )

        safe_name = secure_filename(upload_file.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        storage_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        upload_file.save(storage_path)

        job = ProcessingJob(
            phone_number=normalise_phone_number(phone_number),
            original_filename=safe_name,
            stored_filename=unique_name,
            status="queued",
        )
        db.session.add(job)
        db.session.commit()

        EXECUTOR.submit(process_job, app, job.public_id, app.config["UPLOAD_FOLDER"])

        status_url = url_for("job_status", job_id=job.public_id, _external=True)
        return jsonify({"job_id": job.public_id, "status_url": status_url}), 202

    @app.route("/status/<job_id>")
    def job_status(job_id: str):
        job = ProcessingJob.query.filter_by(public_id=job_id).first()
        if not job:
            return jsonify({"error": "İşlem bulunamadı."}), 404
        return jsonify(job.to_dict())

    @app.route("/jobs/<job_id>")
    def job_detail(job_id: str):
        job = ProcessingJob.query.filter_by(public_id=job_id).first_or_404()
        return render_template("job_detail.html", job=job)

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        if not PRESERVE_UPLOADS:
            return redirect(url_for("index"))
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


def allowed_file(filename: str) -> bool:
    allowed_extensions = {"pdf", "png", "jpg", "jpeg"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def normalise_phone_number(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits.startswith("00"):
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 11:
        digits = "90" + digits[1:]
    if not digits.startswith("+") and not digits.startswith("90") and len(digits) >= 10:
        digits = f"90{digits}" if len(digits) == 10 else digits
    return digits


def process_job(flask_app: Flask, job_id: str, upload_folder: str) -> None:
    with flask_app.app_context():
        job = ProcessingJob.query.filter_by(public_id=job_id).first()
        if not job:
            flask_app.logger.error("Processing job %s bulunamadı", job_id)
            return

        file_path = os.path.join(upload_folder, job.stored_filename)

        try:
            update_job_status(job, "ocr_processing")
            ocr_text = extract_text(file_path)
            if not ocr_text.strip():
                raise RuntimeError("OCR herhangi bir metin çıkaramadı.")

            job.ocr_text = ocr_text
            db.session.commit()

            update_job_status(job, "ai_processing")
            summary = run_openai_pipeline(ocr_text)
            job.ai_summary = summary
            db.session.commit()

            update_job_status(job, "notifying")
            whatsapp_id = dispatch_whatsapp(job.phone_number, summary)
            job.whatsapp_message_id = whatsapp_id
            job.completed_at = datetime.utcnow()
            update_job_status(job, "completed")

        except Exception as exc:  # pragma: no cover - runtime guard
            flask_app.logger.exception("Processing error for job %s", job_id)
            job.error_message = str(exc)
            update_job_status(job, "error")

        finally:
            if not PRESERVE_UPLOADS and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    flask_app.logger.warning("Dosya silinemedi: %s", file_path)
            db.session.commit()


def update_job_status(job: ProcessingJob, status: str) -> None:
    job.status = status
    job.updated_at = datetime.utcnow()
    if status == "completed":
        job.completed_at = datetime.utcnow()
    db.session.commit()


def run_openai_pipeline(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise RuntimeError("İşlenecek metin bulunamadı.")

    if not OPENAI_API_KEY:
        return (
            "OpenAI API anahtarı tanımlı olmadığı için OCR çıktısı olduğu gibi "
            "gönderiliyor.\n\n" + cleaned
        )

    if openai is None:
        raise RuntimeError("openai paketi yüklü değil.")

    openai.api_key = OPENAI_API_KEY

    try:
        response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": OPENAI_PROMPT},
                {
                    "role": "user",
                    "content": cleaned,
                },
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"OpenAI isteği başarısız: {exc}")


def dispatch_whatsapp(phone_number: str, message: str) -> Optional[str]:
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        logging.getLogger(__name__).warning(
            "WhatsApp API bilgileri tanımlı değil; mesaj gönderilmeyecek."
        )
        return None

    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": f"{WHATSAPP_SENDER_NAME}\n\n{message[:4000]}",
        },
    }

    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=20)
    if response.status_code >= 400:
        try:
            details = response.json()
        except json.JSONDecodeError:
            details = response.text
        raise RuntimeError(f"WhatsApp API hatası: {details}")

    data = response.json()
    messages = data.get("messages") if isinstance(data, dict) else None
    if messages and isinstance(messages, list) and messages[0].get("id"):
        return messages[0]["id"]
    return None


app = create_app()


@app.template_filter("status_badge")
def status_badge_filter(status: str) -> str:
    palette = {
        "queued": "badge-neutral",
        "ocr_processing": "badge-info",
        "ai_processing": "badge-info",
        "notifying": "badge-warning",
        "completed": "badge-success",
        "error": "badge-danger",
    }
    return palette.get(status, "badge-neutral")


@app.context_processor
def inject_counts():
    total_jobs = db.session.query(func.count(ProcessingJob.id)).scalar() or 0
    completed_jobs = (
        db.session.query(func.count(ProcessingJob.id))
        .filter(ProcessingJob.status == "completed")
        .scalar()
        or 0
    )
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "datetime": datetime,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
