import json
import os
from datetime import datetime
from constants import LOG_FILE

def load_log():
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
    log = load_log()
    log.append({
        "timestamp": datetime.now().isoformat(),
        "alert_type": alert_type,
        "sentence": sentence,
        "suggestion": suggestion,
        "action": action
    })
    save_log(log)

def build_negative_patterns(log, nlp):
    from collections import Counter
    negative_ngrams = {}
    rejected = [e for e in log if e["action"] == "rejected"]
    for entry in rejected:
        alert_type = entry["alert_type"]
        tokens = list(nlp(entry["sentence"]))
        for i in range(len(tokens) - 1):
            gram = tokens[i].lemma_ + "_" + tokens[i + 1].lemma_
            negative_ngrams.setdefault(alert_type, Counter())[gram] += 1
    return {
        t: {g for g, c in counter.items() if c >= 2}
        for t, counter in negative_ngrams.items()
    }
