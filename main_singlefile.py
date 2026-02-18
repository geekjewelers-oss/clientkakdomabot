# Auto-generated single-file merge of your bot modules

# Sections: config, keyboards, states, bitrix_api, mrz_parser, ocr_fallback, ocr_quality, metrics, ocr_orchestrator, handlers



# ==== Begin: config.py ====


import os

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BITRIX_WEBHOOK_URL = os.getenv("BITRIX_WEBHOOK_URL")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

YANDEX_VISION_API_KEY = os.getenv("YANDEX_VISION_API_KEY")
YANDEX_VISION_FOLDER_ID = os.getenv("YANDEX_VISION_FOLDER_ID")

OCR_SLA_MAX_LOCAL_ATTEMPTS = _int_env("OCR_SLA_MAX_LOCAL_ATTEMPTS", 2)
OCR_SLA_FALLBACK_AFTER_FAILURES = _int_env("OCR_SLA_FALLBACK_AFTER_FAILURES", 2)
OCR_SLA_FALLBACK_PROVIDER = os.getenv("OCR_SLA_FALLBACK_PROVIDER", "yandex_vision")
OCR_SLA_FALLBACK_ATTEMPTS = _int_env("OCR_SLA_FALLBACK_ATTEMPTS", 1)
OCR_SLA_FALLBACK_TIMEOUT_SECONDS = _int_env("OCR_SLA_FALLBACK_TIMEOUT_SECONDS", 5)
OCR_SLA_TOTAL_TIMEOUT_SECONDS = _int_env("OCR_SLA_TOTAL_TIMEOUT_SECONDS", 8)
OCR_SLA_FALLBACK_THRESHOLD_CONFIDENCE = _float_env("OCR_SLA_FALLBACK_THRESHOLD_CONFIDENCE", 0.55)
OCR_SLA_AUTO_ACCEPT_CONFIDENCE = _float_env("OCR_SLA_AUTO_ACCEPT_CONFIDENCE", 0.80)
OCR_SLA_MANUAL_INPUT_AFTER_SECOND_CYCLE = _bool_env("OCR_SLA_MANUAL_INPUT_AFTER_SECOND_CYCLE", True)
OCR_SLA_BREACH_THRESHOLD_RATIO = _float_env("OCR_SLA_BREACH_THRESHOLD_RATIO", 0.9)

OCR_LOG_METRICS_ENABLED = _bool_env("OCR_LOG_METRICS_ENABLED", False)
OCR_METRICS_BACKEND = os.getenv("OCR_METRICS_BACKEND", "noop")



# ==== End: config.py ====



# ==== Begin: registration_kb.py ====


from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

MANAGERS = [
    "Менеджер Анна",
    "Менеджер Борис",
    "Менеджер Светлана",
]

DISTRICTS = [
    "Центральный",
    "Северный",
    "Южный",
    "Западный",
    "Восточный",
    "Другой район",
]

YES_TEXT = "Да"
NO_TEXT = "Нет"

CONFIRM_TEXT = "Подтвердить"
CANCEL_TEXT = "Отменить"
BACK_TEXT = "⬅ Назад"
RETRY_PASSPORT_TEXT = "🔁 Пересканировать паспорт"
BAD_PHOTO_TEXT = "📷 Плохое фото"
EDIT_ADDRESS_TEXT = "✏ Исправить адрес"
GLOBAL_CANCEL_TEXT = "❌ Отменить регистрацию"

ADD_ANOTHER_YES_TEXT = "Добавить ещё"
ADD_ANOTHER_NO_TEXT = "Продолжить"


def manager_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=manager)] for manager in MANAGERS],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def district_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=d)] for d in DISTRICTS] + [[KeyboardButton(text=BACK_TEXT)], [KeyboardButton(text=GLOBAL_CANCEL_TEXT)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def yes_no_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=YES_TEXT), KeyboardButton(text=NO_TEXT)],
            [KeyboardButton(text=GLOBAL_CANCEL_TEXT)],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def confirm_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=CONFIRM_TEXT), KeyboardButton(text=CANCEL_TEXT)],
            [KeyboardButton(text=EDIT_ADDRESS_TEXT)],
            [KeyboardButton(text=GLOBAL_CANCEL_TEXT)],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def add_another_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=ADD_ANOTHER_YES_TEXT)],
            [KeyboardButton(text=ADD_ANOTHER_NO_TEXT)],
            [KeyboardButton(text=GLOBAL_CANCEL_TEXT)],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def back_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BACK_TEXT)], [KeyboardButton(text=GLOBAL_CANCEL_TEXT)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def retry_passport_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=YES_TEXT), KeyboardButton(text=NO_TEXT)],
            [KeyboardButton(text=RETRY_PASSPORT_TEXT)],
            [KeyboardButton(text=GLOBAL_CANCEL_TEXT)],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def bad_photo_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BAD_PHOTO_TEXT)],
            [KeyboardButton(text=GLOBAL_CANCEL_TEXT)],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def edit_address_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=EDIT_ADDRESS_TEXT)],
            [KeyboardButton(text=GLOBAL_CANCEL_TEXT)],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def cancel_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=GLOBAL_CANCEL_TEXT)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


# ==== End: registration_kb.py ====



# ==== Begin: states.py ====


from aiogram.fsm.state import State, StatesGroup


class Form(StatesGroup):
    choosing_manager = State()
    ask_district = State()
    ask_address = State()
    ask_num_people = State()
    ask_passport_photo = State()
    rescan_passport = State()
    manual_input_mode = State()
    confirm_passport_fields = State()
    ask_add_another_passport = State()
    ask_contacts = State()
    ask_move_in_date = State()
    ask_payment_details = State()
    final_confirmation = State()
    done = State()


# ==== End: states.py ====



# ==== Begin: bitrix_api.py ====


import logging

import requests

from bitrix_fields import BITRIX_DEAL_FIELDS
from config import BITRIX_WEBHOOK_URL

logger = logging.getLogger(__name__)


def bitrix_call(method, params):
    """
    Simple wrapper: expects BITRIX_WEBHOOK_URL like https://yourdomain/rest/1/yourhook/
    and will POST to {BITRIX_WEBHOOK_URL}{method}.json
    """
    if not BITRIX_WEBHOOK_URL:
        logger.warning("BITRIX_WEBHOOK_URL не задана")
        return None
    url = BITRIX_WEBHOOK_URL.rstrip("/") + f"/{method}.json"
    try:
        r = requests.post(url, json=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.exception("Bitrix call failed: %s", e)
        return None


def create_lead_and_deal(client_data):
    """
    client_data: dict with keys: surname, given_names, passport_number, phone, address, etc.
    Возвращает (lead_id, deal_id)
    """
    correlation_id = client_data.get("correlation_id")
    lead_fields = {
        "TITLE": f"Лид: {client_data.get('surname', '')} {client_data.get('given_names','')}",
        "NAME": client_data.get('given_names',''),
        "LAST_NAME": client_data.get('surname',''),
        "PHONE": [{"VALUE": client_data.get('phone',''), "VALUE_TYPE": "WORK"}],
        "COMMENTS": f"Авто-лид из Telegram-бота. correlation_id={correlation_id}" if correlation_id else "Авто-лид из Telegram-бота"
    }
    res_lead = bitrix_call("crm.lead.add", {"fields": lead_fields})
    lead_id = None
    if res_lead and 'result' in res_lead:
        lead_id = res_lead['result']

    deal_fields = {
        "TITLE": f"Сделка аренда: {client_data.get('surname','')}",
        "CATEGORY_ID": 0,
        "OPPORTUNITY": client_data.get('amount',''),
        "CURRENCY_ID": "RUB",
        "LEAD_ID": lead_id,
    }
    if correlation_id:
        deal_fields["COMMENTS"] = f"correlation_id={correlation_id}"

    for client_key, bitrix_field in BITRIX_DEAL_FIELDS.items():
        value = client_data.get(client_key)
        if value:
            deal_fields[bitrix_field] = value

    res_deal = bitrix_call("crm.deal.add", {"fields": deal_fields})
    deal_id = None
    if res_deal and 'result' in res_deal:
        deal_id = res_deal['result']

    if lead_id is None:
        logger.error("Не удалось создать лид в Bitrix. response=%s", res_lead)
    if deal_id is None:
        logger.error("Не удалось создать сделку в Bitrix. response=%s", res_deal)

    return lead_id, deal_id


# ==== End: bitrix_api.py ====



# ==== Begin: mrz_parser.py ====


import hashlib
import io
import logging
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

MRZ_REGEX = re.compile(r"([A-Z0-9<]{20,})\s*[\n\r]+([A-Z0-9<]{20,})", re.MULTILINE)
_CHECKSUM_WEIGHTS = (7, 3, 1)
NUM_MAP = {"O": "0", "Q": "0", "I": "1", "L": "1", "B": "8", "S": "5", "G": "6"}


def compute_mrz_hash(line1: str | None, line2: str | None) -> str | None:
    l1 = (line1 or "").strip()
    l2 = (line2 or "").strip()
    if not l1 and not l2:
        return None
    value = f"{l1}|{l2}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest().lower()


def image_bytes_to_pil(img_bytes):
    return Image.open(io.BytesIO(img_bytes))


def preprocess_for_mrz_cv(image: Image.Image):
    """OpenCV preprocessing to enhance MRZ readability"""
    return preprocess_for_mrz_cv_mode(image, mode="current")


def preprocess_for_mrz_cv_mode(image: Image.Image, mode: str = "current"):
    """Preprocess image for MRZ OCR using one of supported modes."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )

    if mode == "morphology":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # current threshold mode
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# Note: use pytesseract directly on the whole image, then search for MRZ lines.
def extract_text_from_image_bytes(img_bytes):
    # PIL -> pytesseract
    pil = image_bytes_to_pil(img_bytes)
    text = pytesseract.image_to_string(pil, lang='eng')  # MRZ uses Latin charset
    return text


def extract_mrz_from_image_bytes(img_bytes):
    """Run MRZ extraction on multiple preprocess variants until MRZ lines are found."""
    pil = image_bytes_to_pil(img_bytes)
    preprocess_modes = ("current", "adaptive", "morphology")

    for mode in preprocess_modes:
        try:
            processed = preprocess_for_mrz_cv_mode(pil, mode=mode)
            text = pytesseract.image_to_string(processed, lang='eng')
        except Exception as exc:
            logger.warning("[OCR] MRZ preprocess failed: mode=%s, error=%s", mode, exc)
            continue

        line1, line2 = find_mrz_from_text(text)
        if line1 and line2:
            logger.info("[OCR] MRZ found with preprocess=%s", mode)
            return line1, line2, text, mode

    return None, None, "", None


def find_mrz_from_text(text):
    # Normalize: remove spaces on MRZ lines
    # We look for two consecutive lines with many '<'
    candidates = MRZ_REGEX.findall(text.replace(" ", "").replace("\r", "\n"))
    if candidates:
        # MRZ_REGEX returns tuples (line1, line2)
        for l1, l2 in candidates:
            # choose first plausible (length check)
            if len(l1) >= 30 and len(l2) >= 30:
                return l1.strip(), l2.strip()
    # fallback: search for sequences with many '<'
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i in range(len(lines)-1):
        a, b = lines[i], lines[i+1]
        if a.count('<') >= 3 and b.count('<') >= 3 and len(a) >= 25 and len(b) >= 25:
            return a.replace(" ", ""), b.replace(" ", "")
    return None, None


def _mrz_char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if 'A' <= ch <= 'Z':
        return ord(ch) - ord('A') + 10
    if ch == '<':
        return 0
    return 0


def compute_mrz_checksum(value: str) -> int:
    total = 0
    for idx, ch in enumerate(value):
        total += _mrz_char_value(ch) * _CHECKSUM_WEIGHTS[idx % 3]
    return total % 10


def normalize_for_numeric(s: str) -> str:
    s = s.upper()
    return "".join(NUM_MAP.get(ch, ch) for ch in s)


def validate_mrz_checksum(value: str, check_char: str) -> bool:
    if not check_char or not check_char.isdigit():
        return False
    return compute_mrz_checksum(value) == int(check_char)


def validate_td3_composite(l2: str) -> bool:
    """Validate TD3 composite checksum from line 2 (position 43)."""
    if len(l2) < 44:
        l2 = l2 + "<" * (44 - len(l2))

    composite_check = l2[43]

    part_doc = normalize_for_numeric(l2[0:10])     # passport + check
    part_birth = normalize_for_numeric(l2[13:20])  # birth + check
    part_exp = normalize_for_numeric(l2[21:28])    # expiry + check
    optional = l2[28:43]                           # may contain letters → no normalize

    composite_value = part_doc + part_birth + part_exp + optional
    return validate_mrz_checksum(composite_value, composite_check)


def parse_td3_mrz(line1: str, line2: str):
    """Parse TD3 passport MRZ (2 lines, 44 chars each normally). Returns dict with fields if possible."""
    # pad to expected lengths to avoid IndexError
    l1 = line1 + "<" * (44 - len(line1)) if len(line1) < 44 else line1
    l2 = line2 + "<" * (44 - len(line2)) if len(line2) < 44 else line2
    data = {}
    checks = {}
    try:
        # line1
        data['document_type'] = l1[0]
        data['issuing_country'] = l1[2:5]
        names = l1[5:44].split('<<')
        surname = names[0].replace('<', ' ').strip()
        given = names[1].replace('<', ' ').strip() if len(names) > 1 else ""
        data['surname'] = surname
        data['given_names'] = given

        # line2
        passport_number_raw = l2[0:9]
        passport_check = l2[9]
        birth_date_raw = l2[13:19]
        birth_check = l2[19]
        expiry_raw = l2[21:27]
        expiry_check = l2[27]

        passport_number_norm = normalize_for_numeric(passport_number_raw)
        birth_date_norm = normalize_for_numeric(birth_date_raw)
        expiry_norm = normalize_for_numeric(expiry_raw)

        data['passport_number'] = passport_number_raw.replace('<', '').strip()
        data['passport_number_check'] = passport_check
        data['nationality'] = l2[10:13].replace('<', '').strip()
        data['birth_date'] = f"{birth_date_raw[0:2]}{birth_date_raw[2:4]}{birth_date_raw[4:6]}"  # YYMMDD
        data['sex'] = l2[20]
        data['expiry_date'] = f"{expiry_raw[0:2]}{expiry_raw[2:4]}{expiry_raw[4:6]}"

        checks["passport_number"] = validate_mrz_checksum(passport_number_norm, passport_check)
        checks["birth_date"] = validate_mrz_checksum(birth_date_norm, birth_check)
        checks["expiry_date"] = validate_mrz_checksum(expiry_norm, expiry_check)
        checks["composite"] = validate_td3_composite(l2)

        if not checks["passport_number"]:
            logger.warning(
                "[OCR] MRZ checksum failed: field=passport_number hash=%s len=%s normalized_len=%s check_char=%s computed=%s",
                compute_mrz_hash(passport_number_raw, None),
                len(passport_number_raw),
                len(passport_number_norm),
                passport_check,
                compute_mrz_checksum(passport_number_norm),
            )
        if not checks["birth_date"]:
            logger.warning(
                "[OCR] MRZ checksum failed: field=birth_date hash=%s len=%s normalized_len=%s check_char=%s computed=%s",
                compute_mrz_hash(birth_date_raw, None),
                len(birth_date_raw),
                len(birth_date_norm),
                birth_check,
                compute_mrz_checksum(birth_date_norm),
            )
        if not checks["expiry_date"]:
            logger.warning(
                "[OCR] MRZ checksum failed: field=expiry_date hash=%s len=%s normalized_len=%s check_char=%s computed=%s",
                compute_mrz_hash(expiry_raw, None),
                len(expiry_raw),
                len(expiry_norm),
                expiry_check,
                compute_mrz_checksum(expiry_norm),
            )
        if not checks["composite"]:
            part_doc = normalize_for_numeric(l2[0:10])
            part_birth = normalize_for_numeric(l2[13:20])
            part_exp = normalize_for_numeric(l2[21:28])
            optional = l2[28:43]
            composite_value = part_doc + part_birth + part_exp + optional
            logger.warning(
                "[OCR] MRZ checksum failed: field=composite hash=%s len=%s check_char=%s computed=%s",
                compute_mrz_hash(composite_value, None),
                len(composite_value),
                l2[43],
                compute_mrz_checksum(composite_value),
            )
    except Exception as e:
        logger.exception("[OCR] Error parsing MRZ: %s", e)
        checks = {"passport_number": False, "birth_date": False, "expiry_date": False, "composite": False}

    check_weights = {
        "passport_number": 0.2,
        "birth_date": 0.2,
        "expiry_date": 0.2,
        "composite": 0.4,
    }
    mrz_confidence_score = sum(weight for key, weight in check_weights.items() if checks.get(key))
    checksum_ok = all(checks.get(key, False) for key in check_weights)

    data["_mrz_checksum_ok"] = checksum_ok
    data["mrz_confidence_score"] = float(mrz_confidence_score)
    return data


# ==== End: mrz_parser.py ====



# ==== Begin: ocr_fallback.py ====


import io
import logging

import easyocr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"])
    return _reader


def easyocr_extract_text(image_bytes):
    logger.info("fallback started")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    reader = _get_reader()
    result = reader.readtext(image_np)

    texts = [item[1] for item in result if len(item) > 1 and item[1]]
    joined_text = " ".join(texts).strip()

    logger.info("number of boxes found: %s", len(result))
    logger.info("fallback text length: %s", len(joined_text))

    return joined_text


# ==== End: ocr_fallback.py ====



# ==== Begin: ocr_quality.py ====


import cv2
import numpy as np


def blur_score(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def exposure_score(gray: np.ndarray) -> float:
    mean = float(np.mean(gray))
    if mean < 60:
        return 0.2
    if mean > 200:
        return 0.3
    return 1.0


def is_blur_bad(score: float) -> bool:
    return score < 80


def is_image_low_quality(mrz_data: dict, blur: float, exposure: float) -> bool:
    conf = float(mrz_data.get("mrz_confidence_score", 0.0))
    return (
        conf < 0.55
        or not mrz_data.get("_mrz_checksum_ok", False)
        or is_blur_bad(blur)
        or exposure < 0.5
    )


def build_ocr_quality_report(mrz_data: dict, blur: float, exposure: float) -> dict:
    conf = float(mrz_data.get("mrz_confidence_score", 0.0))

    return {
        "confidence": conf,
        "checksum_ok": mrz_data.get("_mrz_checksum_ok", False),
        "blur_score": blur,
        "blur_bad": is_blur_bad(blur),
        "exposure_score": exposure,
        "needs_retry": is_image_low_quality(mrz_data, blur, exposure),
    }


# ==== End: ocr_quality.py ====



# ==== Begin: metrics.py ====


import logging
from typing import Any

import config

logger = logging.getLogger(__name__)


_MEMORY_COUNTERS: dict[str, int] = {}
_METRICS_BACKEND_WARNED = False

_METRIC_NAME_MAP = {
    "ocr.sla.soft_fail": "ocr_soft_fail_total",
    "ocr.sla.auto_accept": "ocr_auto_accept_total",
    "ocr.sla.breach": "ocr_sla_breach_total",
    "ocr.sla.fallback_used": "ocr_fallback_used_total",
    "ocr.attempt": "ocr_attempt_total",
    "ocr.manual_input": "ocr_manual_input_total",
    "ocr.cycle": "ocr_cycle_total",
}


def _sanitize_metric_name(name: str) -> str:
    return _METRIC_NAME_MAP.get(name, name)


def metrics_increment(name: str) -> None:
    global _METRICS_BACKEND_WARNED
    try:
        if not config.OCR_LOG_METRICS_ENABLED:
            return

        backend = (config.OCR_METRICS_BACKEND or "noop").strip().lower()
        if backend not in {"noop", "memory"}:
            if not _METRICS_BACKEND_WARNED:
                logger.warning("[METRICS] unsupported backend=%s, fallback=noop", backend)
                _METRICS_BACKEND_WARNED = True
            backend = "noop"

        metric_name = _sanitize_metric_name(name)
        if backend == "memory":
            _MEMORY_COUNTERS[metric_name] = int(_MEMORY_COUNTERS.get(metric_name, 0)) + 1
    except Exception:
        logger.exception("[METRICS] increment failed")


def inc(name: str, value: int = 1) -> None:
    for _ in range(max(0, int(value))):
        metrics_increment(name)


def gauge(name: str, value: float) -> None:
    _ = (name, value)


# ==== End: metrics.py ====



# ==== Begin: ocr_orchestrator.py ====


import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

import cv2
import numpy as np

import metrics
from mrz_parser import compute_mrz_hash, extract_mrz_from_image_bytes, extract_text_from_image_bytes, parse_td3_mrz
from ocr_fallback import easyocr_extract_text
from ocr_quality import blur_score, build_ocr_quality_report, exposure_score
def yandex_vision_extract_text(image_bytes):
    if not YANDEX_VISION_API_KEY or not YANDEX_VISION_FOLDER_ID:
        logger.info("Yandex Vision credentials are not configured")
        return ""

    content = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "folderId": YANDEX_VISION_FOLDER_ID,
        "analyze_specs": [
            {
                "content": content,
                "features": [
                    {
                        "type": "TEXT_DETECTION",
                        "text_detection_config": {
                            "languageCodes": ["en"]
                        },
                    }
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Api-Key {YANDEX_VISION_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            YANDEX_VISION_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Yandex Vision request failed")
        return ""

    data = response.json()

    words = []
    for analyzed in data.get("results", []):
        for result in analyzed.get("results", []):
            text_detection = result.get("textDetection", {})
            for page in text_detection.get("pages", []):
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        for word in line.get("words", []):
                            text = word.get("text")
                            if text:
                                words.append(text)

    extracted_text = " ".join(words).strip()
    logger.info("Yandex Vision text length: %s", len(extracted_text))
    return extracted_text


logger = logging.getLogger(__name__)


def _decode_gray_image(img_bytes: bytes) -> np.ndarray | None:
    np_buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def _attach_quality(result: dict[str, Any], gray: np.ndarray | None) -> dict[str, Any]:
    parsed = result.get("parsed") or {}
    if gray is None:
        quality = build_ocr_quality_report(parsed, blur=0.0, exposure=0.0)
    else:
        blur = blur_score(gray)
        exposure = exposure_score(gray)
        quality = build_ocr_quality_report(parsed, blur=blur, exposure=exposure)
    result["quality"] = quality
    return result


def _local_ocr_attempt(img_bytes: bytes, gray: np.ndarray | None) -> dict[str, Any]:
    line1, line2, mrz_text, _mode = extract_mrz_from_image_bytes(img_bytes)
    if line1 and line2:
        parsed = parse_td3_mrz(line1, line2)
        checksum_ok = parsed.get("_mrz_checksum_ok", False)
        confidence = "high" if checksum_ok else "medium"
        parsed["mrz_confidence_score"] = 0.9 if checksum_ok else 0.6
        text_value = mrz_text or ""
        logger.info("[OCR] OCR stage: mrz, text_len=%s", len(text_value))
        return _attach_quality({
            "text": text_value,
            "source": "mrz",
            "confidence": confidence,
            "parsed": parsed,
            "mrz_lines": (line1, line2),
        }, gray)

    text = extract_text_from_image_bytes(img_bytes)
    logger.info("[OCR] OCR stage: tesseract, text_len=%s", len(text or ""))

    easy_text = easyocr_extract_text(img_bytes)
    logger.info("[OCR] OCR stage: easyocr, text_len=%s", len(easy_text or ""))
    if easy_text and len(easy_text) > 40:
        return _attach_quality({
            "text": easy_text,
            "source": "easyocr",
            "confidence": "medium",
            "parsed": {},
            "mrz_lines": None,
        }, gray)

    return _attach_quality({
        "text": easy_text or text or "",
        "source": "tesseract",
        "confidence": "low",
        "parsed": {},
        "mrz_lines": None,
    }, gray)


def _fallback_ocr_attempt(img_bytes: bytes, current_text: str) -> str:
    if len((current_text or "").strip()) >= 60:
        return current_text
    return yandex_vision_extract_text(img_bytes)


def _run_fallback_with_timeout(img_bytes: bytes, current_text: str) -> tuple[str, bool]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fallback_ocr_attempt, img_bytes, current_text)
        try:
            text = future.result(timeout=config.OCR_SLA_FALLBACK_TIMEOUT_SECONDS)
            logger.info("[OCR] OCR stage: vision, text_len=%s", len(text or ""))
            return text or "", False
        except FutureTimeoutError:
            logger.warning("[OCR] Vision fallback timeout after %ss", config.OCR_SLA_FALLBACK_TIMEOUT_SECONDS)
            return current_text or "", True


def _build_retry_reason_flags(
    quality: dict[str, Any],
    confidence: float,
    timeout_flag: bool,
    fallback_used: bool,
) -> dict[str, bool]:
    return {
        "blur_bad": bool(quality.get("blur_bad", False)),
        "exposure_bad": float(quality.get("exposure_score", 1.0)) < 0.5,
        "checksum_fail": not bool(quality.get("checksum_ok", False)),
        "low_confidence": confidence < config.OCR_SLA_FALLBACK_THRESHOLD_CONFIDENCE,
        "timeout": timeout_flag,
        "fallback_used": fallback_used,
    }


def _soft_fail_response(
    local_count: int,
    fallback_count: int,
    started_at: float,
    timeout_flag: bool,
    used_fallback_provider: str | None,
    correlation_id: str | None,
) -> dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    sla_threshold_ms = int(config.OCR_SLA_TOTAL_TIMEOUT_SECONDS * 1000 * config.OCR_SLA_BREACH_THRESHOLD_RATIO)
    sla_breach = elapsed_ms >= sla_threshold_ms
    quality = build_ocr_quality_report({}, blur=0.0, exposure=0.0)
    retry_reason_flags = _build_retry_reason_flags(
        quality=quality,
        confidence=float(quality.get("confidence", 0.0)),
        timeout_flag=timeout_flag,
        fallback_used=bool(used_fallback_provider),
    )

    metrics_inc: list[str] = ["ocr_attempt_total", "ocr_soft_fail_total"]
    metrics.inc("ocr.sla.soft_fail")
    if used_fallback_provider:
        metrics.inc("ocr.sla.fallback_used")
        metrics_inc.append("ocr_fallback_used_total")
    if sla_breach:
        metrics.inc("ocr.sla.breach")
        metrics_inc.append("ocr_sla_breach_total")

    return {
        "text": "",
        "source": "sla_soft_fail",
        "confidence": "low",
        "parsed": {},
        "mrz_lines": None,
        "quality": quality,
        "attempt_local_count": local_count,
        "attempt_fallback_count": fallback_count,
        "total_elapsed_ms": elapsed_ms,
        "decision_branch": "soft_fail",
        "used_fallback_provider": used_fallback_provider,
        "timeout_flag": timeout_flag,
        "retry_reason_flags": retry_reason_flags,
        "sla_breach": sla_breach,
        "sla_breach_flag": sla_breach,
        "correlation_id": correlation_id,
        "passport_hash": None,
        "passport_mrz_len": 0,
        "metrics_inc": metrics_inc,
        "logger_version": "ocr_sla_v1",
    }


def ocr_pipeline_extract(img_bytes: bytes, correlation_id: str | None = None) -> dict[str, Any]:
    metrics.inc("ocr.attempt")
    started_at = time.monotonic()
    gray = _decode_gray_image(img_bytes)
    local_attempts = 0
    fallback_attempts = 0
    local_failures = 0
    last_result: dict[str, Any] = {}
    used_fallback_provider: str | None = None
    timeout_flag = False

    for _ in range(config.OCR_SLA_MAX_LOCAL_ATTEMPTS):
        local_attempts += 1
        if (time.monotonic() - started_at) > config.OCR_SLA_TOTAL_TIMEOUT_SECONDS:
            timeout_flag = True
            return _soft_fail_response(
                local_count=local_attempts,
                fallback_count=fallback_attempts,
                started_at=started_at,
                timeout_flag=timeout_flag,
                used_fallback_provider=used_fallback_provider,
                correlation_id=correlation_id,
            )

        result = _local_ocr_attempt(img_bytes, gray)
        last_result = result
        quality = result.get("quality") or {}
        parsed = result.get("parsed") or {}
        conf = float(quality.get("confidence", 0.0))

        local_failed = not parsed or bool(quality.get("needs_retry", False)) or conf < config.OCR_SLA_FALLBACK_THRESHOLD_CONFIDENCE
        if local_failed:
            local_failures += 1
        if not local_failed:
            break

    should_use_fallback = (
        local_attempts >= config.OCR_SLA_MAX_LOCAL_ATTEMPTS
        or local_failures >= config.OCR_SLA_FALLBACK_AFTER_FAILURES
    )

    if should_use_fallback:
        current_text = (last_result.get("text") if last_result else "") or ""
        for _ in range(config.OCR_SLA_FALLBACK_ATTEMPTS):
            fallback_attempts += 1
            used_fallback_provider = config.OCR_SLA_FALLBACK_PROVIDER

            if (time.monotonic() - started_at) > config.OCR_SLA_TOTAL_TIMEOUT_SECONDS:
                timeout_flag = True
                return _soft_fail_response(
                    local_count=local_attempts,
                    fallback_count=fallback_attempts,
                    started_at=started_at,
                    timeout_flag=timeout_flag,
                    used_fallback_provider=used_fallback_provider,
                    correlation_id=correlation_id,
                )

            fallback_text, fallback_timeout = _run_fallback_with_timeout(img_bytes, current_text)
            timeout_flag = timeout_flag or fallback_timeout

            if fallback_text:
                last_result = _attach_quality({
                    "text": fallback_text,
                    "source": "vision",
                    "confidence": "medium",
                    "parsed": {},
                    "mrz_lines": None,
                }, gray)
                break

    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    if elapsed_ms > config.OCR_SLA_TOTAL_TIMEOUT_SECONDS * 1000:
        timeout_flag = True

    sla_threshold_ms = int(config.OCR_SLA_TOTAL_TIMEOUT_SECONDS * 1000 * config.OCR_SLA_BREACH_THRESHOLD_RATIO)
    sla_breach = elapsed_ms >= sla_threshold_ms

    if timeout_flag:
        return _soft_fail_response(
            local_count=local_attempts,
            fallback_count=fallback_attempts,
            started_at=started_at,
            timeout_flag=timeout_flag,
            used_fallback_provider=used_fallback_provider,
            correlation_id=correlation_id,
        )

    if not last_result:
        return _soft_fail_response(
            local_count=local_attempts,
            fallback_count=fallback_attempts,
            started_at=started_at,
            timeout_flag=timeout_flag,
            used_fallback_provider=used_fallback_provider,
            correlation_id=correlation_id,
        )

    quality = last_result.get("quality") or {}
    confidence_value = float(quality.get("confidence", 0.0))
    needs_retry = bool(quality.get("needs_retry", False))

    if needs_retry:
        decision_branch = "soft_fail"
    elif confidence_value >= config.OCR_SLA_AUTO_ACCEPT_CONFIDENCE:
        decision_branch = "auto_accept"
    elif confidence_value >= config.OCR_SLA_FALLBACK_THRESHOLD_CONFIDENCE:
        decision_branch = "preview_required"
    else:
        decision_branch = "soft_fail"

    metrics_inc: list[str] = ["ocr_attempt_total"]
    if used_fallback_provider:
        metrics.inc("ocr.sla.fallback_used")
        metrics_inc.append("ocr_fallback_used_total")
    if sla_breach:
        metrics.inc("ocr.sla.breach")
        metrics_inc.append("ocr_sla_breach_total")

    retry_reason_flags = _build_retry_reason_flags(
        quality=quality,
        confidence=confidence_value,
        timeout_flag=timeout_flag,
        fallback_used=bool(used_fallback_provider),
    )

    if decision_branch == "auto_accept":
        metrics.inc("ocr.sla.auto_accept")
        metrics_inc.append("ocr_auto_accept_total")
    elif decision_branch == "soft_fail":
        metrics.inc("ocr.sla.soft_fail")
        metrics_inc.append("ocr_soft_fail_total")

    mrz_lines = last_result.get("mrz_lines")
    line1 = mrz_lines[0] if mrz_lines else None
    line2 = mrz_lines[1] if mrz_lines else None
    passport_hash = compute_mrz_hash(line1, line2)
    passport_mrz_len = len((line1 or "").strip() + (line2 or "").strip())

    last_result.update({
        "attempt_local_count": local_attempts,
        "attempt_fallback_count": fallback_attempts,
        "total_elapsed_ms": elapsed_ms,
        "decision_branch": decision_branch,
        "used_fallback_provider": used_fallback_provider,
        "timeout_flag": timeout_flag,
        "retry_reason_flags": retry_reason_flags,
        "sla_breach": sla_breach,
        "sla_breach_flag": sla_breach,
        "correlation_id": correlation_id,
        "passport_hash": passport_hash,
        "passport_mrz_len": passport_mrz_len,
        "metrics_inc": metrics_inc,
        "logger_version": "ocr_sla_v1",
    })
    return last_result


# ==== End: ocr_orchestrator.py ====



# ==== Begin: handlers_registration.py ====


import io
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from aiogram import F, Router

from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, ReplyKeyboardMarkup, ReplyKeyboardRemove

import metrics
from fsm.states import Form
from keyboards.registration_kb import (
    ADD_ANOTHER_NO_TEXT,
    ADD_ANOTHER_YES_TEXT,
    BACK_TEXT,
    BAD_PHOTO_TEXT,
    CANCEL_TEXT,
    CONFIRM_TEXT,
    DISTRICTS,
    EDIT_ADDRESS_TEXT,
    GLOBAL_CANCEL_TEXT,
    MANAGERS,
    NO_TEXT,
    RETRY_PASSPORT_TEXT,
    YES_TEXT,
    add_another_keyboard,
    back_kb,
    bad_photo_kb,
    confirm_keyboard,
    district_keyboard,
    manager_keyboard,
    retry_passport_kb,
)
from ocr_orchestrator import ocr_pipeline_extract

logger = logging.getLogger(__name__)
router = Router(name="registration")


def _new_session() -> dict[str, Any]:
    return {
        "flow": "registration",
        "correlation_id": str(uuid4()),
        "manager_id": None,
        "district": None,
        "address": None,
        "num_people_expected": 0,
        "passports": [],
        "current_passport_index": 1,
        "phone": None,
        "move_in_date": None,
        "payment": {},
        "ocr_cycle_counter": 0,
        "ocr_retry_counter": 0,
        "last_ocr_decision": {},
        "last_retry_reason_flags": {},
    }


def _ensure_session_hardening(session: dict[str, Any]) -> dict[str, Any]:
    if not session.get("correlation_id"):
        session["correlation_id"] = str(uuid4())
    session.setdefault("ocr_cycle_counter", 0)
    session.setdefault("ocr_retry_counter", 0)
    if not isinstance(session.get("last_ocr_decision"), dict):
        session["last_ocr_decision"] = {}
    if not isinstance(session.get("last_retry_reason_flags"), dict):
        session["last_retry_reason_flags"] = {}
    return session


def _session_summary(session: dict[str, Any]) -> str:
    payment = session.get("payment", {})
    passports = session.get("passports", [])
    lines = [
        "Проверьте данные перед отправкой:",
        f"• Менеджер: {session.get('manager_id')}",
        f"• Район: {session.get('district')}",
        f"• Адрес: {session.get('address')}",
        f"• Жильцов: {session.get('num_people_expected')}",
        f"• Паспортов подтверждено: {len(passports)}",
        f"• Телефон: {session.get('phone')}",
        f"• Дата заезда: {session.get('move_in_date')}",
        f"• Аренда: {payment.get('rent')}",
        f"• Депозит: {payment.get('deposit')}",
        f"• Комиссия: {payment.get('commission')}",
    ]
    return "\n".join(lines)


def _is_valid_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\+?[0-9()\-\s]{10,20}", phone.strip()))


def _quality_retry_reasons(quality: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if quality.get("blur_bad"):
        reasons.append("Фото размыто")
    if not quality.get("checksum_ok", False):
        reasons.append("MRZ не читается")
    if float(quality.get("exposure_score", 1.0)) < 0.5:
        reasons.append("Слишком темное/светлое фото")
    return reasons


def _retry_reasons_from_flags(flags: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if flags.get("blur_bad"):
        reasons.append("Фото размыто")
    if flags.get("exposure_bad"):
        reasons.append("Слишком темное/светлое фото")
    if flags.get("checksum_fail"):
        reasons.append("MRZ не читается")
    if flags.get("low_confidence"):
        reasons.append("Низкая уверенность OCR")
    if flags.get("timeout"):
        reasons.append("Превышен таймаут OCR")
    if flags.get("fallback_used"):
        reasons.append("Использован fallback OCR")
    return reasons


def _parse_manual_passport_input(raw_text: str) -> dict[str, str] | None:
    parts = [part.strip() for part in raw_text.split(";")]
    if len(parts) != 6:
        return None
    return {
        "surname": parts[0],
        "given_names": parts[1],
        "passport_number": parts[2],
        "nationality": parts[3],
        "birth_date": parts[4],
        "expiry_date": parts[5],
    }


async def _get_session(state: FSMContext) -> dict[str, Any]:
    data = await state.get_data()
    session = data.get("session", _new_session())
    session = _ensure_session_hardening(session)
    await state.update_data(session=session)
    return session


async def _go_to_step(
    message: Message,
    state: FSMContext,
    *,
    next_state: Any,
    text: str,
    keyboard: ReplyKeyboardMarkup | ReplyKeyboardRemove | None = None,
    log_step: str,
) -> None:
    await state.set_state(next_state)
    logger.info("FSM step entered: %s", log_step)
    kwargs = {"reply_markup": keyboard} if keyboard is not None else {}
    await message.answer(text, **kwargs)


@router.message(CommandStart())
async def start_registration(message: Message, state: FSMContext) -> None:
    session = _new_session()
    await state.set_data({"session": session})
    await _go_to_step(
        message,
        state,
        next_state=Form.choosing_manager,
        text="Здравствуйте! Начнем регистрацию арендатора. Выберите менеджера:",
        keyboard=manager_keyboard(),
        log_step="choosing_manager",
    )


@router.message(F.text == GLOBAL_CANCEL_TEXT)
async def process_global_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    logger.info("REGISTRATION_CANCELLED")
    await message.answer("Регистрация отменена", reply_markup=ReplyKeyboardRemove())
    await start_registration(message, state)


@router.message(Form.ask_district, F.text == BACK_TEXT)
async def back_from_ask_district(message: Message, state: FSMContext) -> None:
    session = await _get_session(state)
    session["district"] = None
    await state.update_data(session=session)
    logger.info("FSM_BACK_STEP from=ask_district to=choosing_manager")
    await _go_to_step(
        message,
        state,
        next_state=Form.choosing_manager,
        text="Выберите менеджера:",
        keyboard=manager_keyboard(),
        log_step="choosing_manager",
    )


@router.message(Form.ask_address, F.text == BACK_TEXT)
async def back_from_ask_address(message: Message, state: FSMContext) -> None:
    session = await _get_session(state)
    session["address"] = None
    await state.update_data(session=session)
    logger.info("FSM_BACK_STEP from=ask_address to=ask_district")
    await _go_to_step(
        message,
        state,
        next_state=Form.ask_district,
        text="Укажите район объекта:",
        keyboard=district_keyboard(),
        log_step="ask_district",
    )


@router.message(Form.ask_num_people, F.text == BACK_TEXT)
async def back_from_ask_num_people(message: Message, state: FSMContext) -> None:
    session = await _get_session(state)
    session["num_people_expected"] = 0
    session["current_passport_index"] = 1
    session["passports"] = []
    await state.update_data(session=session)
    logger.info("FSM_BACK_STEP from=ask_num_people to=ask_address")
    await _go_to_step(
        message,
        state,
        next_state=Form.ask_address,
        text="Введите полный адрес:",
        keyboard=back_kb(),
        log_step="ask_address",
    )


@router.message(Form.ask_contacts, F.text == BACK_TEXT)
async def back_from_ask_contacts(message: Message, state: FSMContext) -> None:
    session = await _get_session(state)
    session["phone"] = None
    await state.update_data(session=session)
    logger.info("FSM_BACK_STEP from=ask_contacts to=ask_add_another_passport")
    await _go_to_step(
        message,
        state,
        next_state=Form.ask_add_another_passport,
        text="Добавить еще один паспорт?",
        keyboard=add_another_keyboard(),
        log_step="ask_add_another_passport",
    )


@router.message(Form.ask_move_in_date, F.text == BACK_TEXT)
async def back_from_ask_move_in_date(message: Message, state: FSMContext) -> None:
    session = await _get_session(state)
    session["move_in_date"] = None
    await state.update_data(session=session)
    logger.info("FSM_BACK_STEP from=ask_move_in_date to=ask_contacts")
    await _go_to_step(
        message,
        state,
        next_state=Form.ask_contacts,
        text="Введите контактный телефон:",
        keyboard=back_kb(),
        log_step="ask_contacts",
    )


@router.message(Form.ask_payment_details, F.text == BACK_TEXT)
async def back_from_ask_payment_details(message: Message, state: FSMContext) -> None:
    session = await _get_session(state)
    session["payment"] = {}
    await state.update_data(session=session)
    logger.info("FSM_BACK_STEP from=ask_payment_details to=ask_move_in_date")
    await _go_to_step(
        message,
        state,
        next_state=Form.ask_move_in_date,
        text="Введите дату заезда в формате YYYY-MM-DD",
        keyboard=back_kb(),
        log_step="ask_move_in_date",
    )


@router.message(Form.confirm_passport_fields, F.text == RETRY_PASSPORT_TEXT)
async def process_retry_passport(message: Message, state: FSMContext) -> None:
    session = await _get_session(state)
    passport_index = session.get("current_passport_index", 1)
    session["passports"] = [p for p in session.get("passports", []) if p.get("index") != passport_index]
    await state.update_data(session=session)
    logger.info("PASSPORT_RETRY | passport index=%s", passport_index)
    await _go_to_step(
        message,
        state,
        next_state=Form.ask_passport_photo,
        text=f"Отправьте новое фото для паспорта №{passport_index}.",
        keyboard=bad_photo_kb(),
        log_step=f"ask_passport_photo | passport index={passport_index}",
    )


@router.message(Form.ask_passport_photo, F.text == BAD_PHOTO_TEXT)
async def process_bad_photo_hint(message: Message) -> None:
    logger.info("BAD_PHOTO_HINT_SHOWN")
    await message.answer(
        "Подсказка по фото паспорта:\n"
        "• без бликов\n"
        "• весь разворот\n"
        "• читаемая MRZ зона\n"
        "• без обрезки краёв"
    )


@router.message(Form.final_confirmation, F.text == EDIT_ADDRESS_TEXT)
async def process_edit_address(message: Message, state: FSMContext) -> None:
    logger.info("FSM_BACK_STEP from=final_confirmation to=ask_address")
    await _go_to_step(
        message,
        state,
        next_state=Form.ask_address,
        text="Введите полный адрес:",
        keyboard=back_kb(),
        log_step="ask_address",
    )


@router.message(Form.choosing_manager)
async def process_manager(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if text not in MANAGERS:
        await message.answer("Выберите менеджера с клавиатуры ниже.", reply_markup=manager_keyboard())
        return

    session = await _get_session(state)
    session["manager_id"] = text
    await state.update_data(session=session)

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_district,
        text="Укажите район объекта:",
        keyboard=district_keyboard(),
        log_step="ask_district",
    )


@router.message(Form.ask_district)
async def process_district(message: Message, state: FSMContext) -> None:
    district = (message.text or "").strip()
    if not district:
        await message.answer("Район не должен быть пустым.")
        return

    if district not in DISTRICTS:
        await message.answer("Выберите район из списка или нажмите 'Другой район'.", reply_markup=district_keyboard())
        return

    session = await _get_session(state)
    session["district"] = district
    await state.update_data(session=session)

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_address,
        text="Введите полный адрес:",
        keyboard=back_kb(),
        log_step="ask_address",
    )


@router.message(Form.ask_address)
async def process_address(message: Message, state: FSMContext) -> None:
    address = (message.text or "").strip()
    if not address:
        await message.answer("Адрес не должен быть пустым. Введите адрес еще раз.")
        return

    session = await _get_session(state)
    session["address"] = address
    await state.update_data(session=session)

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_num_people,
        text="Сколько человек будет проживать?",
        keyboard=back_kb(),
        log_step="ask_num_people",
    )


@router.message(Form.ask_num_people)
async def process_num_people(message: Message, state: FSMContext) -> None:
    value = (message.text or "").strip()
    if not value.isdigit() or int(value) <= 0:
        await message.answer("Введите целое число больше 0.")
        return

    num_people = int(value)
    session = await _get_session(state)
    session["num_people_expected"] = num_people
    session["current_passport_index"] = 1
    session["passports"] = []
    await state.update_data(session=session)

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_passport_photo,
        text=f"Пришлите фото паспорта №{session['current_passport_index']} (как фото, не файл).",
        keyboard=bad_photo_kb(),
        log_step=f"ask_passport_photo | passport index={session['current_passport_index']}",
    )


@router.message(Form.ask_passport_photo, ~F.photo)
@router.message(Form.rescan_passport, ~F.photo)
async def process_passport_not_photo(message: Message) -> None:
    await message.answer("На этом шаге нужно отправить фотографию паспорта.")


async def _process_passport_photo_common(message: Message, state: FSMContext, *, source_state: str) -> None:
    session = await _get_session(state)
    session = _ensure_session_hardening(session)
    passport_index = session.get("current_passport_index", 1)
    logger.info("FSM step entered: %s | passport index=%s", source_state, passport_index)

    photo = message.photo[-1]
    file = await message.bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await message.bot.download(file, destination=buf)
    img_bytes = buf.getvalue()

    correlation_id = session.get("correlation_id")
    if not correlation_id:
        correlation_id = str(uuid4())
        session["correlation_id"] = correlation_id
        await state.update_data(session=session)

    ocr_cycle_counter = int(session.get("ocr_cycle_counter", 0))
    ocr_retry_counter = int(session.get("ocr_retry_counter", 0))

    metrics.metrics_increment("ocr_cycle_total")
    ocr_result = ocr_pipeline_extract(img_bytes, correlation_id=correlation_id)
    text = ocr_result.get("text") or ""
    parsed_fields = ocr_result.get("parsed") or {}
    parsed = dict(parsed_fields)
    mrz_lines = ocr_result.get("mrz_lines")
    source = ocr_result.get("source") or "unknown"
    confidence = ocr_result.get("confidence") or "low"
    quality = ocr_result.get("quality") or {}
    conf = float(quality.get("confidence", 0.0))

    decision_branch = ocr_result.get("decision_branch") or "soft_fail"
    timeout_flag = bool(ocr_result.get("timeout_flag", False))
    retry_reason_flags = ocr_result.get("retry_reason_flags") or {}
    local_attempts = int(ocr_result.get("attempt_local_count", 0))
    fallback_attempts = int(ocr_result.get("attempt_fallback_count", 0))
    total_elapsed_ms = int(ocr_result.get("total_elapsed_ms", 0))
    used_fallback_provider = ocr_result.get("used_fallback_provider")
    sla_breach = bool(ocr_result.get("sla_breach", False))
    passport_hash = ocr_result.get("passport_hash")
    passport_mrz_len = int(ocr_result.get("passport_mrz_len", 0))
    metrics_inc = list(ocr_result.get("metrics_inc") or [])
    logger_version = ocr_result.get("logger_version") or "ocr_sla_v1"

    session["passport_quality"] = quality
    session["passport_confidence"] = conf
    session["passport_needs_retry"] = bool(quality.get("needs_retry", False))
    session["last_ocr_decision"] = {
        "correlation_id": correlation_id,
        "passport_index": passport_index,
        "ocr_cycle_counter": ocr_cycle_counter,
        "ocr_retry_counter": ocr_retry_counter,
        "attempt_local_count": local_attempts,
        "attempt_fallback_count": fallback_attempts,
        "used_fallback_provider": used_fallback_provider,
        "decision_branch": decision_branch,
        "confidence": conf,
        "total_elapsed_ms": total_elapsed_ms,
        "timeout_flag": timeout_flag,
        "sla_breach_flag": sla_breach,
        "passport_hash": passport_hash,
    }
    session["ocr_retry_reason_flags"] = retry_reason_flags
    session["last_retry_reason_flags"] = retry_reason_flags
    session["ocr_retry_counter"] = int(session.get("ocr_retry_counter", 0)) + 1

    manual_mode_triggered = False
    if decision_branch == "soft_fail":
        session["ocr_cycle_counter"] = int(session.get("ocr_cycle_counter", 0)) + 1
        if (
            config.OCR_SLA_MANUAL_INPUT_AFTER_SECOND_CYCLE
            and int(session.get("ocr_cycle_counter", 0)) >= 2
        ):
            manual_mode_triggered = True

    await state.update_data(session=session)

    logger.info(
        "OCR_QUALITY: blur=%s confidence=%s checksum_ok=%s needs_retry=%s",
        quality.get("blur_score"),
        conf,
        quality.get("checksum_ok", False),
        quality.get("needs_retry", False),
    )

    logger.info("[OCR] handler stage: source=%s, confidence=%s, text_len=%d", source, confidence, len(text))

    if decision_branch == "soft_fail" and "ocr_soft_fail_total" not in metrics_inc:
        metrics.inc("ocr.sla.soft_fail")
        metrics_inc.append("ocr_soft_fail_total")
    if decision_branch == "auto_accept" and "ocr_auto_accept_total" not in metrics_inc:
        metrics.inc("ocr.sla.auto_accept")
        metrics_inc.append("ocr_auto_accept_total")
    if used_fallback_provider and "ocr_fallback_used_total" not in metrics_inc:
        metrics.inc("ocr.sla.fallback_used")
        metrics_inc.append("ocr_fallback_used_total")
    if sla_breach and "ocr_sla_breach_total" not in metrics_inc:
        metrics.inc("ocr.sla.breach")
        metrics_inc.append("ocr_sla_breach_total")

    ocr_sla_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": "INFO",
        "logger": "OCR_SLA_DECISION",
        "correlation_id": correlation_id,
        "passport_index": passport_index,
        "ocr_cycle_counter": ocr_cycle_counter,
        "ocr_retry_counter": ocr_retry_counter,
        "deal_id": session.get("deal_id"),
        "lead_id": session.get("lead_id"),
        "passport_hash": passport_hash,
        "passport_mrz_len": passport_mrz_len,
        "attempt_local_count": local_attempts,
        "attempt_fallback_count": fallback_attempts,
        "total_elapsed_ms": total_elapsed_ms,
        "decision_branch": decision_branch,
        "confidence": conf,
        "used_fallback_provider": used_fallback_provider,
        "timeout_flag": timeout_flag,
        "sla_breach": sla_breach,
        "sla_breach_flag": sla_breach,
        "retry_reason_flags": retry_reason_flags,
        "metrics_inc": metrics_inc,
        "logger_version": logger_version,
    }
    logger.info("OCR_SLA_DECISION", extra=ocr_sla_log)

    if decision_branch == "soft_fail" or timeout_flag or quality.get("needs_retry"):
        reasons = _retry_reasons_from_flags(retry_reason_flags) or _quality_retry_reasons(quality)
        reasons_text = f"\nПричины: {', '.join(reasons)}." if reasons else ""

        if manual_mode_triggered:
            await _go_to_step(
                message,
                state,
                next_state=Form.manual_input_mode,
                text=(
                    "Автораспознавание не удалось после двух циклов. "
                    "Перейдите к ручному вводу в формате:\n"
                    "Фамилия;Имя;Номер паспорта;Гражданство;Дата рождения;Срок действия"
                ),
                keyboard=back_kb(),
                log_step=f"manual_input_mode | passport index={passport_index}",
            )
            return

        await _go_to_step(
            message,
            state,
            next_state=Form.rescan_passport,
            text=(
                "Фото плохо читается. Пожалуйста пришлите более четкое фото паспорта "
                "(без бликов, полностью MRZ зона)."
                f"{reasons_text}"
            ),
            keyboard=bad_photo_kb(),
            log_step=f"rescan_passport | passport index={passport_index}",
        )
        return

    auto_confirm_passport = decision_branch == "auto_accept"

    if not parsed_fields:
        await _go_to_step(
            message,
            state,
            next_state=Form.rescan_passport,
            text="Не удалось распознать паспортные данные. Отправьте более четкое фото этого же паспорта.",
            keyboard=bad_photo_kb(),
            log_step=f"rescan_passport | passport index={passport_index}",
        )
        return

    passport_entry = {
        "index": passport_index,
        "photo_file_id": photo.file_id,
        "parsed": parsed,
        "mrz_lines": mrz_lines,
        "passport_hash": passport_hash,
        "ocr_source": source,
        "ocr_confidence": confidence,
        "ocr_quality": quality,
        "ocr_blur": quality.get("blur_score"),
        "ocr_exposure": quality.get("exposure_score"),
        "confirmed": False,
    }

    if auto_confirm_passport:
        passport_entry["confirmed"] = True

    passports = [p for p in session.get("passports", []) if p.get("index") != passport_index]
    passports.append(passport_entry)
    passports.sort(key=lambda x: x["index"])
    session["passports"] = passports
    await state.update_data(session=session)

    parsed_text = "\n".join(
        [
            f"Фамилия: {parsed.get('surname', '—')}",
            f"Имя: {parsed.get('given_names', '—')}",
            f"Номер паспорта: {parsed.get('passport_number', '—')}",
            f"Гражданство: {parsed.get('nationality', '—')}",
            f"Дата рождения: {parsed.get('birth_date', '—')}",
            f"Срок действия: {parsed.get('expiry_date', '—')}",
        ]
    )

    if auto_confirm_passport:
        await _go_to_step(
            message,
            state,
            next_state=Form.ask_add_another_passport,
            text=f"Паспорт №{passport_index} распознан и автоматически подтвержден.",
            keyboard=add_another_keyboard(),
            log_step=f"ask_add_another_passport | passport index={passport_index}",
        )
        return

    await _go_to_step(
        message,
        state,
        next_state=Form.confirm_passport_fields,
        text=f"Паспорт №{passport_index} распознан:\n\n{parsed_text}\n\nВсе верно?",
        keyboard=retry_passport_kb(),
        log_step=f"confirm_passport_fields | passport index={passport_index}",
    )


@router.message(Form.ask_passport_photo, F.photo)
async def process_passport_photo(message: Message, state: FSMContext) -> None:
    await _process_passport_photo_common(message, state, source_state="ask_passport_photo")


@router.message(Form.rescan_passport, F.photo)
async def process_passport_rescan_photo(message: Message, state: FSMContext) -> None:
    await _process_passport_photo_common(message, state, source_state="rescan_passport")


@router.message(Form.manual_input_mode)
async def process_manual_input_mode(message: Message, state: FSMContext) -> None:
    session = await _get_session(state)
    session = _ensure_session_hardening(session)
    passport_index = session.get("current_passport_index", 1)
    correlation_id = session.get("correlation_id")
    parsed = _parse_manual_passport_input((message.text or "").strip())
    if not parsed:
        await message.answer(
            "Неверный формат. Введите данные так:\nФамилия;Имя;Номер паспорта;Гражданство;Дата рождения;Срок действия"
        )
        return

    passport_entry = {
        "index": passport_index,
        "photo_file_id": None,
        "parsed": parsed,
        "mrz_lines": None,
        "passport_hash": None,
        "ocr_source": "manual_input",
        "manual_override": True,
        "ocr_confidence": "manual",
        "ocr_quality": session.get("passport_quality", {}),
        "ocr_blur": None,
        "ocr_exposure": None,
        "confirmed": True,
    }

    passports = [p for p in session.get("passports", []) if p.get("index") != passport_index]
    passports.append(passport_entry)
    passports.sort(key=lambda x: x["index"])
    session["passports"] = passports
    await state.update_data(session=session)

    metrics.inc("ocr.manual_input")
    logger.info(
        "MANUAL_INPUT_USED",
        extra={
            "correlation_id": correlation_id,
            "passport_index": passport_index,
        },
    )

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_add_another_passport,
        text=f"Паспорт №{passport_index} сохранен в ручном режиме.",
        keyboard=add_another_keyboard(),
        log_step=f"ask_add_another_passport | passport index={passport_index}",
    )


@router.message(Form.confirm_passport_fields)
async def process_passport_confirmation(message: Message, state: FSMContext) -> None:
    answer = (message.text or "").strip()
    if answer not in {YES_TEXT, NO_TEXT}:
        await message.answer("Пожалуйста, выберите Да или Нет.", reply_markup=retry_passport_kb())
        return

    session = await _get_session(state)
    passport_index = session.get("current_passport_index", 1)
    passports = session.get("passports", [])

    for passport in passports:
        if passport.get("index") == passport_index:
            passport["confirmed"] = answer == YES_TEXT
            break

    logger.info("confirmation result=%s | passport index=%s", answer, passport_index)

    session["passports"] = passports
    await state.update_data(session=session)

    if answer == NO_TEXT:
        await _go_to_step(
            message,
            state,
            next_state=Form.ask_passport_photo,
            text=f"Хорошо, отправьте новое фото для паспорта №{passport_index}.",
            keyboard=bad_photo_kb(),
            log_step=f"ask_passport_photo | passport index={passport_index}",
        )
        return

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_add_another_passport,
        text="Добавить еще один паспорт?",
        keyboard=add_another_keyboard(),
        log_step=f"ask_add_another_passport | passport index={passport_index}",
    )


@router.message(Form.ask_add_another_passport)
async def process_add_another_passport(message: Message, state: FSMContext) -> None:
    answer = (message.text or "").strip()
    if answer not in {ADD_ANOTHER_YES_TEXT, ADD_ANOTHER_NO_TEXT}:
        await message.answer("Выберите вариант на клавиатуре.", reply_markup=add_another_keyboard())
        return

    session = await _get_session(state)

    confirmed_count = sum(1 for p in session.get("passports", []) if p.get("confirmed"))
    expected = session.get("num_people_expected", 0)

    if answer == ADD_ANOTHER_YES_TEXT:
        session["current_passport_index"] = session.get("current_passport_index", 1) + 1
        await state.update_data(session=session)
        await _go_to_step(
            message,
            state,
            next_state=Form.ask_passport_photo,
            text=f"Пришлите фото паспорта №{session['current_passport_index']}.",
            keyboard=bad_photo_kb(),
            log_step=f"ask_passport_photo | passport index={session['current_passport_index']}",
        )
        return

    if confirmed_count < expected:
        await message.answer(
            f"Подтверждено паспортов: {confirmed_count} из {expected}. Добавьте оставшиеся.",
            reply_markup=add_another_keyboard(),
        )
        return

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_contacts,
        text="Введите контактный телефон:",
        keyboard=back_kb(),
        log_step="ask_contacts",
    )


@router.message(Form.ask_contacts)
async def process_contacts(message: Message, state: FSMContext) -> None:
    phone = (message.text or "").strip()
    if not _is_valid_phone(phone):
        await message.answer("Введите корректный телефон, например +79991234567")
        return

    session = await _get_session(state)
    session["phone"] = phone
    await state.update_data(session=session)

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_move_in_date,
        text="Введите дату заезда в формате YYYY-MM-DD",
        keyboard=back_kb(),
        log_step="ask_move_in_date",
    )


@router.message(Form.ask_move_in_date)
async def process_move_in_date(message: Message, state: FSMContext) -> None:
    date_text = (message.text or "").strip()
    try:
        datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        await message.answer("Неверный формат даты. Используйте YYYY-MM-DD")
        return

    session = await _get_session(state)
    session["move_in_date"] = date_text
    await state.update_data(session=session)

    await _go_to_step(
        message,
        state,
        next_state=Form.ask_payment_details,
        text="Введите платежи в формате: аренда, депозит, комиссия",
        keyboard=back_kb(),
        log_step="ask_payment_details",
    )


@router.message(Form.ask_payment_details)
async def process_payment_details(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    chunks = [c.strip().replace(" ", "") for c in raw.split(",")]
    if len(chunks) != 3 or not all(re.fullmatch(r"\d+(\.\d+)?", c) for c in chunks):
        await message.answer("Нужен формат: аренда, депозит, комиссия. Например: 50000, 30000, 25000")
        return

    session = await _get_session(state)
    session["payment"] = {
        "rent": float(chunks[0]),
        "deposit": float(chunks[1]),
        "commission": float(chunks[2]),
    }
    await state.update_data(session=session)

    await _go_to_step(
        message,
        state,
        next_state=Form.final_confirmation,
        text=_session_summary(session),
        keyboard=confirm_keyboard(),
        log_step="final_confirmation",
    )


@router.message(Form.final_confirmation)
async def process_final_confirmation(message: Message, state: FSMContext) -> None:
    answer = (message.text or "").strip()
    if answer not in {CONFIRM_TEXT, CANCEL_TEXT}:
        await message.answer("Выберите Подтвердить или Отменить.", reply_markup=confirm_keyboard())
        return

    session = await _get_session(state)

    if answer == CANCEL_TEXT:
        await state.clear()
        logger.info("REGISTRATION_CANCELLED")
        await message.answer("Регистрация отменена", reply_markup=ReplyKeyboardRemove())
        await start_registration(message, state)
        return

    logger.info("confirmation result=%s | flow=%s", answer, session.get("flow"))
    await _go_to_step(
        message,
        state,
        next_state=Form.done,
        text="Спасибо! Регистрация завершена ✅",
        keyboard=ReplyKeyboardRemove(),
        log_step="done",
    )
    await state.clear()


# ==== End: handlers_registration.py ====



# ==== Begin: main_merged.py ====


# =========================
# Imports
# =========================
import asyncio
import base64
import hashlib
import io
import logging
import os
import re
from pathlib import Path
from typing import Any

import boto3
import cv2
import numpy as np
import pytesseract
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup, ReplyKeyboardRemove
from botocore.client import Config
from dotenv import load_dotenv
from PIL import Image


# =========================
# Config / env loading
# =========================
load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN", "")
BITRIX_WEBHOOK_URL = os.getenv("BITRIX_WEBHOOK_URL", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_BUCKET = os.getenv("S3_BUCKET", "")
YANDEX_VISION_API_KEY = os.getenv("YANDEX_VISION_API_KEY", "")
YANDEX_VISION_FOLDER_ID = os.getenv("YANDEX_VISION_FOLDER_ID", "")

S3_REGION = os.getenv("S3_REGION", "us-east-1")
DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR", "downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

OCR_MIN_EASYOCR_LEN = _int_env("OCR_MIN_EASYOCR_LEN", 40)
OCR_SKIP_VISION_IF_LEN = _int_env("OCR_SKIP_VISION_IF_LEN", 60)

# Enterprise OCR config (use existing values if present)
OCR_MIN_EASYOCR_LEN = globals().get("OCR_MIN_EASYOCR_LEN", OCR_MIN_EASYOCR_LEN if "OCR_MIN_EASYOCR_LEN" in globals() else 30)
OCR_SKIP_VISION_IF_LEN = globals().get("OCR_SKIP_VISION_IF_LEN", OCR_SKIP_VISION_IF_LEN if "OCR_SKIP_VISION_IF_LEN" in globals() else 100)

# Yandex config safeguards
YANDEX_VISION_API_KEY = globals().get("YANDEX_VISION_API_KEY") or os.getenv("YANDEX_VISION_API_KEY")
YANDEX_VISION_FOLDER_ID = globals().get("YANDEX_VISION_FOLDER_ID") or os.getenv("YANDEX_VISION_FOLDER_ID")
YANDEX_VISION_ENDPOINT = globals().get("YANDEX_VISION_ENDPOINT", "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze")
YANDEX_VISION_ENABLED = bool(YANDEX_VISION_API_KEY or YANDEX_VISION_FOLDER_ID)
YANDEX_VISION_RETRY = int(os.getenv("YANDEX_VISION_RETRY", "2"))
YANDEX_VISION_TIMEOUT = float(os.getenv("YANDEX_VISION_TIMEOUT", "10"))
YANDEX_VISION_BACKOFF_BASE = float(os.getenv("YANDEX_VISION_BACKOFF_BASE", "0.5"))

BITRIX_DEAL_FIELDS = {
    "surname": "UF_CRM_PASSPORT_SURNAME",
    "given_names": "UF_CRM_PASSPORT_NAME",
    "passport_number": "UF_CRM_PASSPORT_NUMBER",
    "nationality": "UF_CRM_PASSPORT_NATION",
    "birth_date": "UF_CRM_BIRTH_DATE",
    "expiry_date": "UF_CRM_PASSPORT_EXPIRY",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# MRZ parsing functions
# =========================
MRZ_REGEX = re.compile(r"([A-Z0-9<]{20,})\s*[\n\r]+([A-Z0-9<]{20,})", re.MULTILINE)
_CHECKSUM_WEIGHTS = (7, 3, 1)
NUM_MAP = {"O": "0", "Q": "0", "I": "1", "L": "1", "B": "8", "S": "5", "G": "6"}


def compute_mrz_hash(line1: str | None, line2: str | None) -> str | None:
    l1 = (line1 or "").strip()
    l2 = (line2 or "").strip()
    if not l1 and not l2:
        return None
    value = f"{l1}|{l2}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest().lower()


def image_bytes_to_pil(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes))


def preprocess_for_mrz_cv_mode(image: Image.Image, mode: str = "current") -> np.ndarray:
    """Preprocess image for MRZ OCR using one of supported modes."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )

    if mode == "morphology":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # current threshold mode
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def extract_text_from_image_bytes(img_bytes: bytes) -> str:
    pil = image_bytes_to_pil(img_bytes)
    return pytesseract.image_to_string(pil, lang="eng")


def find_mrz_from_text(text: str) -> tuple[str | None, str | None]:
    candidates = MRZ_REGEX.findall(text.replace(" ", "").replace("\r", "\n"))
    if candidates:
        for l1, l2 in candidates:
            if len(l1) >= 30 and len(l2) >= 30:
                return l1.strip(), l2.strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for idx in range(len(lines) - 1):
        line_a, line_b = lines[idx], lines[idx + 1]
        if line_a.count("<") >= 3 and line_b.count("<") >= 3 and len(line_a) >= 25 and len(line_b) >= 25:
            return line_a.replace(" ", ""), line_b.replace(" ", "")

    return None, None


def extract_mrz_from_image_bytes(img_bytes: bytes) -> tuple[str | None, str | None, str, str | None]:
    """Run MRZ extraction on multiple preprocess variants until MRZ lines are found."""
    pil = image_bytes_to_pil(img_bytes)
    preprocess_modes = ("current", "adaptive", "morphology")

    for mode in preprocess_modes:
        try:
            processed = preprocess_for_mrz_cv_mode(pil, mode=mode)
            text = pytesseract.image_to_string(processed, lang="eng")
        except Exception as exc:
            logger.warning("[OCR] MRZ preprocess failed: mode=%s, error=%s", mode, exc)
            continue

        line1, line2 = find_mrz_from_text(text)
        if line1 and line2:
            logger.info("[OCR] MRZ found with preprocess=%s", mode)
            return line1, line2, text, mode

    return None, None, "", None


def _mrz_char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    if ch == "<":
        return 0
    return 0


def compute_mrz_checksum(value: str) -> int:
    total = 0
    for idx, ch in enumerate(value):
        total += _mrz_char_value(ch) * _CHECKSUM_WEIGHTS[idx % 3]
    return total % 10


def normalize_for_numeric(value: str) -> str:
    value = value.upper()
    return "".join(NUM_MAP.get(ch, ch) for ch in value)


def validate_mrz_checksum(value: str, check_char: str) -> bool:
    if not check_char or not check_char.isdigit():
        return False
    return compute_mrz_checksum(value) == int(check_char)


def validate_td3_composite(line2: str) -> bool:
    if len(line2) < 44:
        line2 = line2 + "<" * (44 - len(line2))

    composite_check = line2[43]
    part_doc = normalize_for_numeric(line2[0:10])
    part_birth = normalize_for_numeric(line2[13:20])
    part_exp = normalize_for_numeric(line2[21:28])
    optional = line2[28:43]

    composite_value = part_doc + part_birth + part_exp + optional
    return validate_mrz_checksum(composite_value, composite_check)


def parse_td3_mrz(line1: str, line2: str) -> dict[str, Any]:
    """Parse TD3 passport MRZ (2 lines, 44 chars each normally)."""
    l1 = line1 + "<" * (44 - len(line1)) if len(line1) < 44 else line1
    l2 = line2 + "<" * (44 - len(line2)) if len(line2) < 44 else line2

    data: dict[str, Any] = {}
    checks: dict[str, bool] = {}

    try:
        data["document_type"] = l1[0]
        data["issuing_country"] = l1[2:5]
        names = l1[5:44].split("<<")
        data["surname"] = names[0].replace("<", " ").strip()
        data["given_names"] = names[1].replace("<", " ").strip() if len(names) > 1 else ""

        passport_number_raw = l2[0:9]
        passport_check = l2[9]
        birth_date_raw = l2[13:19]
        birth_check = l2[19]
        expiry_raw = l2[21:27]
        expiry_check = l2[27]

        passport_number_norm = normalize_for_numeric(passport_number_raw)
        birth_date_norm = normalize_for_numeric(birth_date_raw)
        expiry_norm = normalize_for_numeric(expiry_raw)

        data["passport_number"] = passport_number_raw.replace("<", "").strip()
        data["passport_number_check"] = passport_check
        data["nationality"] = l2[10:13].replace("<", "").strip()
        data["birth_date"] = f"{birth_date_raw[0:2]}{birth_date_raw[2:4]}{birth_date_raw[4:6]}"
        data["sex"] = l2[20]
        data["expiry_date"] = f"{expiry_raw[0:2]}{expiry_raw[2:4]}{expiry_raw[4:6]}"

        checks["passport_number"] = validate_mrz_checksum(passport_number_norm, passport_check)
        checks["birth_date"] = validate_mrz_checksum(birth_date_norm, birth_check)
        checks["expiry_date"] = validate_mrz_checksum(expiry_norm, expiry_check)
        checks["composite"] = validate_td3_composite(l2)
    except Exception:
        logger.exception("[OCR] Error parsing MRZ")
        checks = {"passport_number": False, "birth_date": False, "expiry_date": False, "composite": False}

    check_weights = {
        "passport_number": 0.2,
        "birth_date": 0.2,
        "expiry_date": 0.2,
        "composite": 0.4,
    }
    mrz_confidence_score = sum(weight for key, weight in check_weights.items() if checks.get(key))
    data["_mrz_checksum_ok"] = all(checks.get(key, False) for key in check_weights)
    data["mrz_confidence_score"] = float(mrz_confidence_score)
    return data


# ==== Begin: ocr_fallback (enterprise) ====
import threading
import time
import random
import json
from typing import List, Optional

# Optional imports handled gracefully
try:
    import easyocr
except Exception:
    easyocr = None
try:
    import numpy as np
except Exception:
    np = None
try:
    from PIL import Image
except Exception:
    Image = None

# Optional metrics (use if prometheus_client available)
try:
    from prometheus_client import Counter, Histogram
    METRICS_ENABLED = True
    METRICS_EASYOCR_CALLS = Counter("ocr_easyocr_calls_total", "EasyOCR calls")
    METRICS_VISION_CALLS = Counter("ocr_vision_calls_total", "Vision API calls")
    METRICS_EASYOCR_TIME = Histogram("ocr_easyocr_duration_seconds", "EasyOCR duration")
    METRICS_VISION_TIME = Histogram("ocr_vision_duration_seconds", "Vision API duration")
except Exception:
    METRICS_ENABLED = False

_logger = logging.getLogger(__name__)

# Thread-safe lazy singleton for EasyOCR reader
_easyocr_reader_lock = threading.Lock()
_easyocr_reader = None


def _get_easyocr_reader(langs: List[str] = ["en"]) -> Optional[object]:
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    if easyocr is None:
        _logger.warning("[OCR][EasyOCR] easyocr not installed")
        return None
    with _easyocr_reader_lock:
        if _easyocr_reader is not None:
            return _easyocr_reader
        try:
            # initialize reader (let easyocr choose device)
            _easyocr_reader = easyocr.Reader(langs)
            _logger.info("[OCR][EasyOCR] initialized reader langs=%s", langs)
        except Exception as e:
            _logger.exception("[OCR][EasyOCR] init failed: %s", e)
            _easyocr_reader = None
        return _easyocr_reader


def easyocr_extract_text(image_bytes: bytes) -> str:
    """
    Enterprise EasyOCR wrapper:
    - lazy reader
    - convert bytes -> PIL -> numpy
    - call reader.readtext(...)
    - join texts
    - metrics & detailed logging
    """
    _logger.info("[OCR][EasyOCR] fallback started")
    start = time.time()
    if METRICS_ENABLED:
        METRICS_EASYOCR_CALLS.inc()
    reader = _get_easyocr_reader(["en"])
    if reader is None:
        _logger.info("[OCR][EasyOCR] reader unavailable, returning empty string")
        return ""
    if Image is None or np is None:
        _logger.warning("[OCR][EasyOCR] PIL or numpy missing")
        return ""
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil)
        # detail=1 returns (bbox, text, conf); paragraph=False to avoid auto-joining
        results = reader.readtext(img_np, detail=1, paragraph=False)
        texts = []
        for r in results:
            if isinstance(r, (list, tuple)):
                if len(r) >= 2 and r[1]:
                    texts.append(str(r[1]))
                elif len(r) == 1:
                    texts.append(str(r[0]))
            elif isinstance(r, dict) and "text" in r:
                texts.append(str(r.get("text", "")))
            else:
                texts.append(str(r))
        joined = " ".join([t.strip() for t in texts if t and t.strip()])
        duration = time.time() - start
        _logger.info("[OCR][EasyOCR] done len=%d boxes=%d dur=%.3fs", len(joined), len(results), duration)
        if METRICS_ENABLED:
            METRICS_EASYOCR_TIME.observe(duration)
        return joined
    except Exception as e:
        _logger.exception("[OCR][EasyOCR] readtext failed: %s", e)
        return ""


# Robust Yandex Vision OCR wrapper with retries/backoff
def _sleep_backoff(attempt: int):
    # exponential backoff with jitter
    base = float(globals().get("YANDEX_VISION_BACKOFF_BASE", 0.5))
    sleep_for = base * (2 ** attempt) + random.uniform(0, base)
    time.sleep(sleep_for)


def _parse_yandex_vision_response(resp_json: dict) -> str:
    """
    Try several known response shapes:
      - batchAnalyze -> results -> results -> textDetection -> pages -> blocks -> lines -> words
      - ocr/recognizeText -> results -> pages -> blocks -> lines -> words
    """
    texts = []
    # Try batchAnalyze shape
    for top in resp_json.get("results", []) or []:
        for sub in top.get("results", []) or []:
            # nested object might be textDetection or text_detection
            td = sub.get("textDetection") or sub.get("text_detection") or sub.get("ocr") or {}
            pages = td.get("pages") or td.get("pages", []) or []
            for page in pages:
                for block in page.get("blocks", []) or []:
                    for line in block.get("lines", []) or []:
                        # words may be dicts with 'text'
                        words = []
                        for w in line.get("words", []) or []:
                            if isinstance(w, dict):
                                t = w.get("text") or w.get("word") or ""
                                if t:
                                    words.append(t)
                            else:
                                # some shapes might have direct strings
                                words.append(str(w))
                        if words:
                            texts.append(" ".join(words))
    # fallback: sometimes API returns top-level 'text' or 'fullTextAnnotation'
    if not texts:
        if "text" in resp_json and isinstance(resp_json["text"], str):
            texts.append(resp_json["text"])
        if "fullTextAnnotation" in resp_json:
            # try to extract paragraphs/blocks
            fta = resp_json["fullTextAnnotation"]
            if isinstance(fta, dict):
                for page in fta.get("pages", []) or []:
                    for block in page.get("blocks", []) or []:
                        for paragraph in block.get("paragraphs", []) or []:
                            line_texts = []
                            for word in paragraph.get("words", []) or []:
                                sym_texts = [s.get("text", "") for s in (word.get("symbols", []) or [])]
                                if sym_texts:
                                    line_texts.append("".join(sym_texts))
                            if line_texts:
                                texts.append(" ".join(line_texts))
    return " ".join([t.strip() for t in texts if t and t.strip()])


def yandex_vision_extract_text(image_bytes: bytes) -> str:
    """
    Yandex Vision API call with:
      - retries (YANDEX_VISION_RETRY)
      - timeout (YANDEX_VISION_TIMEOUT)
      - optional Api-Key (Authorization: Api-Key ...)
      - support for multiple response shapes
    Returns '' on any fatal error or if credentials missing.
    """
    if not globals().get("YANDEX_VISION_ENABLED", False):
        _logger.info("[OCR][Vision] disabled (no creds)")
        return ""
    api_key = globals().get("YANDEX_VISION_API_KEY")
    folder = globals().get("YANDEX_VISION_FOLDER_ID")
    endpoint = globals().get("YANDEX_VISION_ENDPOINT", "")
    if not endpoint:
        endpoint = "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze"
    payload_template = {
        "analyze_specs": [
            {
                # content will be filled per request
                "content": None,
                "features": [
                    {
                        "type": "TEXT_DETECTION",
                        "text_detection_config": {
                            "language_codes": ["en"]
                        }
                    }
                ],
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
    if folder:
        # some endpoints expect folderId in payload, some expect x-folder-id header
        headers["x-folder-id"] = folder

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = payload_template.copy()
    payload["analyze_specs"][0]["content"] = b64
    retry = int(globals().get("YANDEX_VISION_RETRY", 2))
    timeout = float(globals().get("YANDEX_VISION_TIMEOUT", 10))
    for attempt in range(retry + 1):
        try:
            if METRICS_ENABLED:
                METRICS_VISION_CALLS.inc()
            _logger.debug("[OCR][Vision] calling endpoint=%s attempt=%d", endpoint, attempt)
            t0 = time.time()
            r = requests.post(endpoint, data=json.dumps(payload), headers=headers, timeout=timeout)
            r.raise_for_status()
            dur = time.time() - t0
            if METRICS_ENABLED:
                METRICS_VISION_TIME.observe(dur)
            _logger.debug("[OCR][Vision] http status=%s dur=%.3f", r.status_code, dur)
            j = r.json()
            text = _parse_yandex_vision_response(j)
            _logger.info("[OCR][Vision] text_len=%d", len(text))
            # TODO: add circuit-breaker cooldown integration for persistent upstream failures.
            return text
        except requests.RequestException as re:
            _logger.warning("[OCR][Vision] request failed attempt=%d: %s", attempt, re)
            if attempt < retry:
                _sleep_backoff(attempt)
                continue
            _logger.exception("[OCR][Vision] final failure: %s", re)
            return ""
        except Exception as e:
            _logger.exception("[OCR][Vision] unexpected error: %s", e)
            return ""
# ==== End: ocr_fallback (enterprise) ====


def ocr_pipeline_extract(img_bytes: bytes) -> dict[str, Any]:
    # existing code: extract MRZ lines as before
    line1, line2, mrz_text, _mode = extract_mrz_from_image_bytes(img_bytes)
    if line1 and line2:
        # preserve exact previous behavior: return MRZ as source if MRZ exists
        logger.info("MRZ found")
        parsed = parse_td3_mrz(line1, line2)
        checksum_ok = parsed.get("_mrz_checksum_ok", False)
        confidence = "high" if checksum_ok else "medium"
        return {
            "text": mrz_text or "",
            "source": "mrz",
            "confidence": confidence,
            "parsed": parsed,
            "mrz_lines": (line1, line2),
        }

    # No MRZ — use Tesseract first (existing)
    text_from_tesseract = extract_text_from_image_bytes(img_bytes)  # preserve existing function
    logger.info("[OCR] Tesseract text_len=%d", len(text_from_tesseract or ""))

    # Run EasyOCR as primary local fallback
    logger.info("MRZ not found — running EasyOCR")
    try:
        easy_text = easyocr_extract_text(img_bytes)
    except Exception:
        logger.exception("EasyOCR fallback failed")
        easy_text = ""

    # Determine if EasyOCR output is sufficient per existing config
    min_len = int(globals().get("OCR_MIN_EASYOCR_LEN", OCR_MIN_EASYOCR_LEN))
    skip_vision_if_len = int(globals().get("OCR_SKIP_VISION_IF_LEN", OCR_SKIP_VISION_IF_LEN))

    # If easy_text meets threshold -> accept it
    if easy_text and len(easy_text.strip()) >= min_len:
        logger.info("[OCR] EasyOCR accepted len=%d", len(easy_text.strip()))
        return {
            "text": easy_text,
            "source": "easyocr",
            "confidence": "medium",
            "parsed": {},
            "mrz_lines": None,
        }

    # If easy_text is weak or empty decide whether to call Vision
    # If tesseract/easy combined already long enough to skip vision, keep tesseract
    combined_len = len((easy_text or "") + (text_from_tesseract or ""))
    if combined_len >= skip_vision_if_len:
        # prefer longer of easy_text and tesseract
        preferred = easy_text if (easy_text and len(easy_text) > len(text_from_tesseract or "")) else (text_from_tesseract or "")
        src = "easyocr" if preferred == easy_text and easy_text else "tesseract"
        logger.info("[OCR] Skipping Vision (combined_len=%d >= %d) -> source=%s", combined_len, skip_vision_if_len, src)
        return {
            "text": preferred,
            "source": src,
            "confidence": "low",
            "parsed": {},
            "mrz_lines": None,
        }

    # Otherwise escalate to Yandex Vision if enabled
    if globals().get("YANDEX_VISION_ENABLED", False):
        logger.info("EasyOCR weak — running Vision API")
        try:
            vision_text = yandex_vision_extract_text(img_bytes)
        except Exception:
            logger.exception("Vision fallback failed")
            vision_text = ""
    else:
        logger.info("Vision disabled or no creds; skipping Vision API")
        vision_text = ""

    # Choose best available: prioritize non-empty and longer text
    candidates = [
        ("vision", vision_text or ""),
        ("easyocr", easy_text or ""),
        ("tesseract", text_from_tesseract or ""),
    ]
    # select longest non-empty
    chosen_source, chosen_text = max(candidates, key=lambda kv: len(kv[1]))
    if not chosen_text:
        chosen_text = ""
        chosen_source = "tesseract"
    logger.info("[OCR] final source=%s len=%d", chosen_source, len(chosen_text))
    return {
        "text": chosen_text,
        "source": chosen_source,
        "confidence": "low",
        "parsed": {},
        "mrz_lines": None,
    }


# =========================
# S3 upload functions
# =========================
def get_s3_client():
    if not (S3_ENDPOINT_URL and S3_ACCESS_KEY and S3_SECRET_KEY and S3_BUCKET):
        return None

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name=S3_REGION,
    )


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "application/octet-stream") -> str | None:
    s3 = get_s3_client()
    if s3 is None:
        logger.warning("[S3] S3 credentials are not configured")
        return None

    fileobj = io.BytesIO(data)
    s3.upload_fileobj(
        Fileobj=fileobj,
        Bucket=S3_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": content_type, "ACL": "private"},
    )

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=60 * 60 * 24 * 7,
    )


# =========================
# Bitrix API functions
# =========================
def bitrix_call(method: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if not BITRIX_WEBHOOK_URL:
        logger.warning("[Bitrix] BITRIX_WEBHOOK_URL is not configured")
        return None

    url = BITRIX_WEBHOOK_URL.rstrip("/") + f"/{method}.json"
    try:
        response = requests.post(url, json=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.exception("[Bitrix] Request failed: method=%s", method)
        return None


def create_lead_and_deal(client_data: dict[str, Any]) -> tuple[Any, Any]:
    correlation_id = client_data.get("correlation_id")
    lead_fields = {
        "TITLE": f"Лид: {client_data.get('surname', '')} {client_data.get('given_names', '')}",
        "NAME": client_data.get("given_names", ""),
        "LAST_NAME": client_data.get("surname", ""),
        "PHONE": [{"VALUE": client_data.get("phone", ""), "VALUE_TYPE": "WORK"}],
        "COMMENTS": f"Авто-лид из Telegram-бота. correlation_id={correlation_id}" if correlation_id else "Авто-лид из Telegram-бота",
    }

    lead_resp = bitrix_call("crm.lead.add", {"fields": lead_fields})
    lead_id = lead_resp.get("result") if lead_resp else None

    deal_fields = {
        "TITLE": f"Сделка аренда: {client_data.get('surname', '')}",
        "CATEGORY_ID": 0,
        "OPPORTUNITY": client_data.get("amount", ""),
        "CURRENCY_ID": "RUB",
        "LEAD_ID": lead_id,
    }
    if correlation_id:
        deal_fields["COMMENTS"] = f"correlation_id={correlation_id}"

    for client_key, bitrix_field in BITRIX_DEAL_FIELDS.items():
        value = client_data.get(client_key)
        if value:
            deal_fields[bitrix_field] = value

    deal_resp = bitrix_call("crm.deal.add", {"fields": deal_fields})
    deal_id = deal_resp.get("result") if deal_resp else None

    if lead_id is None:
        logger.error("[Bitrix] Failed creating lead")
    if deal_id is None:
        logger.error("[Bitrix] Failed creating deal")

    return lead_id, deal_id


# =========================
# Telegram bot handler functions
# =========================
class Form(StatesGroup):
    waiting_checklist_confirmation = State()
    waiting_passport_photo = State()
    waiting_field_corrections = State()


def yes_no_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Да, у меня есть все документы")],
            [KeyboardButton(text="Нет, хочу отправить позже")],
        ],
        resize_keyboard=True,
    )


def all_correct_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Всё верно")]],
        resize_keyboard=True,
    )


def format_parsed(parsed: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Фамилия: {parsed.get('surname', '(не найдено)')}",
            f"Имя: {parsed.get('given_names', '(не найдено)')}",
            f"Номер паспорта: {parsed.get('passport_number', '(не найдено)')}",
            f"Гражданство: {parsed.get('nationality', '(не найдено)')}",
            f"Дата рождения (YYMMDD): {parsed.get('birth_date', '(не найдено)')}",
            f"Срок действия (YYMMDD): {parsed.get('expiry_date', '(не найдено)')}",
        ]
    )


def register_handlers(dp: Dispatcher, bot: Bot) -> None:
    @dp.message(Command("start"))
    async def cmd_start(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer(
            "Привет! Я помогу загрузить документы для заселения.\n"
            "Сначала проверим, есть ли у вас все необходимые документы.",
            reply_markup=yes_no_keyboard(),
        )
        await state.set_state(Form.waiting_checklist_confirmation)

    @dp.message(Form.waiting_checklist_confirmation, F.text == "Да, у меня есть все документы")
    async def got_checklist_yes(message: Message, state: FSMContext) -> None:
        await message.answer(
            "Отлично. Пришлите, пожалуйста, фотографию паспорта (главная страница или MRZ).",
            reply_markup=ReplyKeyboardRemove(),
        )
        await state.set_state(Form.waiting_passport_photo)

    @dp.message(Form.waiting_checklist_confirmation, F.text == "Нет, хочу отправить позже")
    async def got_checklist_no(message: Message, state: FSMContext) -> None:
        await message.answer("Хорошо. Напишите /start когда будете готовы.", reply_markup=ReplyKeyboardRemove())
        await state.clear()

    @dp.message(Form.waiting_passport_photo)
    async def passport_received(message: Message, state: FSMContext) -> None:
        file_id = None
        content_type = "image/jpeg"

        if message.photo:
            file_id = message.photo[-1].file_id
        elif message.document and message.document.mime_type and message.document.mime_type.startswith("image"):
            file_id = message.document.file_id
            content_type = message.document.mime_type

        if not file_id:
            await message.answer("Пожалуйста, отправьте фото паспорта в виде фото или image-файла.")
            return

        file_info = await bot.get_file(file_id)
        image_stream = await bot.download_file(file_info.file_path)
        image_bytes = image_stream.read()

        await message.answer("Получил фото. Пытаюсь распознать данные... Пару секунд.")
        correlation_id = data.get("correlation_id") or f"{message.from_user.id}-{message.message_id}"
        metrics.metrics_increment("ocr_cycle_total")
        ocr_result = ocr_pipeline_extract(image_bytes)
        parsed = ocr_result.get("parsed") or {}
        if not parsed:
            line1, line2 = find_mrz_from_text(ocr_result.get("text", ""))
            if line1 and line2:
                parsed = parse_td3_mrz(line1, line2)

        await state.update_data({
            "parsed": parsed,
            "passport_bytes": image_bytes,
            "passport_content_type": content_type,
            "correlation_id": correlation_id,
        })

        msg = (
            "Вот что я нашёл:\n\n"
            + format_parsed(parsed)
            + "\n\nЕсли что-то неверно — пришлите исправления в формате `поле: значение` "
            + "(например `Номер паспорта: AB12345`), или нажмите кнопку 'Всё верно'."
        )
        await message.answer(msg, reply_markup=all_correct_keyboard(), parse_mode="Markdown")
        await state.set_state(Form.waiting_field_corrections)

    @dp.message(Form.waiting_field_corrections)
    async def corrections_handler(message: Message, state: FSMContext) -> None:
        text = (message.text or "").strip()
        data = await state.get_data()
        parsed = data.get("parsed", {})

        if text == "Всё верно":
            await message.answer("Отлично — сохраняю данные и создаю лид в CRM...", reply_markup=ReplyKeyboardRemove())

            correlation_id = data.get("correlation_id") or f"{message.from_user.id}-{message.message_id}"
            client_data = {
                "surname": parsed.get("surname"),
                "given_names": parsed.get("given_names"),
                "passport_number": parsed.get("passport_number"),
                "nationality": parsed.get("nationality"),
                "birth_date": parsed.get("birth_date"),
                "expiry_date": parsed.get("expiry_date"),
                "correlation_id": correlation_id,
            }
            lead_id, deal_id = create_lead_and_deal(client_data)

            passport_bytes = data.get("passport_bytes", b"")
            if passport_bytes:
                s3_key = f"passports/{message.from_user.id}_{message.message_id}.jpg"
                try:
                    file_url = upload_bytes_to_s3(passport_bytes, key=s3_key, content_type=data.get("passport_content_type", "image/jpeg"))
                    if file_url and deal_id:
                        bitrix_call(
                            "crm.activity.add",
                            {
                                "fields": {
                                    "OWNER_ID": deal_id,
                                    "OWNER_TYPE_ID": 2,
                                    "SUBJECT": "Фото паспорта",
                                    "DESCRIPTION": f"{file_url}\ncorrelation_id={correlation_id}",
                                    "SETTINGS": {"CORRELATION_ID": correlation_id},
                                }
                            },
                        )
                except Exception:
                    logger.exception("[S3] Failed to upload passport image")

            await message.answer(f"Лид создан: {lead_id}, Сделка: {deal_id}")
            await state.clear()
            return

        if ":" in text:
            key, val = text.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            field_map = {
                "фамилия": "surname",
                "имя": "given_names",
                "номер паспорта": "passport_number",
                "паспорт": "passport_number",
                "гражданство": "nationality",
                "дата рождения": "birth_date",
                "срок действия": "expiry_date",
            }
            if key in field_map:
                parsed[field_map[key]] = val
                await state.update_data({"parsed": parsed})
                await message.answer(
                    f"Поле `{key}` обновлено на `{val}`. Если всё готово — нажмите 'Всё верно'.",
                    parse_mode="Markdown",
                )
            else:
                await message.answer("Не распознал поле для правки. Пример: `Фамилия: Иванов`", parse_mode="Markdown")
            return

        await message.answer(
            "Не понял. Для подтверждения нажмите 'Всё верно' или пришлите исправление в формате `Поле: значение`.",
            parse_mode="Markdown",
        )


# =========================
# Main polling loop
# =========================
async def run_bot() -> None:
    if not TELEGRAM_TOKEN:
        raise SystemExit("TELEGRAM_TOKEN (or BOT_TOKEN) required")

    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    register_handlers(dp, bot)

    logger.info("Запускаю Telegram-бота...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(run_bot())


# ==== End: main_merged.py ====


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    print('main_singlefile_v2.py created. This file contains merged bot modules.\n')
    print('To run the bot, install requirements and run via an aiogram runner. This script will not auto-start the bot to avoid side effects.')


# === Reference: simple handlers from original main (1).py (kept as comment) ===
"""
Reference snippet omitted.
"""
