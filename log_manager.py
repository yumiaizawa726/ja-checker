import json
import os
from datetime import datetime
from collections import Counter
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from constants import LOG_FILE

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
SHEET_NAME = "ja-checker-log"

def get_sheet():
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )
        gc = gspread.authorize(creds)
        return gc.open(SHEET_NAME).sheet1
    except Exception:
        return None

def load_log():
    sheet = get_sheet()
    if sheet:
        try:
            rows = sheet.get_all_records()
            return rows
        except Exception:
            pass
    # ローカルフォールバック
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_log(entries):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def record_action(alert_type, sentence, suggestion, action):
    sheet = get_sheet()
    if sheet:
        try:
            sheet.append_row([
                datetime.now().isoformat(),
                alert_type,
                sentence,
                suggestion,
                action
            ])
            return
        except Exception:
            pass
    # ローカルフォールバック
    log = load_log()
    if isinstance(log, list) and (not log or isinstance(log[0], dict)):
        log.append({
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "sentence": sentence,
            "suggestion": suggestion,
            "action": action
        })
        save_log(log)

def build_negative_patterns(log, nlp):
    negative_ngrams = {}
    rejected = [e for e in log if e.get("action") == "rejected"]
    for entry in rejected:
        alert_type = entry.get("alert_type", "")
        tokens = list(nlp(entry.get("sentence", "")))
        for i in range(len(tokens) - 1):
            gram = tokens[i].lemma_ + "_" + tokens[i + 1].lemma_
            negative_ngrams.setdefault(alert_type, Counter())[gram] += 1
    return {
        t: {g for g, c in counter.items() if c >= 2}
        for t, counter in negative_ngrams.items()
    }
