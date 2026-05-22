import re
from constants import (
    DISCOURSE_MARKERS, SKIP_MARKERS, EXPERIENCER_PREDICATES,
    FUNCTION_WORDS, WEAK_CLAIM_PATTERNS, WEAK_CLAIMS_FLAT,
    ACTION_TEMPLATES, MAX_SENTENCE_LEN
)

TOPIC_EXCLUSIONS = {"今日", "昨日", "明日", "今", "ここ", "そこ", "あそこ", "これ", "それ", "あれ"}

def normalize_punctuation(text):
    text = text.replace("．", "。")
    text = text.replace("；", "。")
    text = re.sub(r'([ぁ-んァ-ン一-龥])\.', r'\1。', text)
    text = text.replace(",", "、")
    text = text.replace("，", "、")
    text = text.replace("!", "。")
    text = text.replace("？", "。")
    text = text.replace("?", "。")
    return text

def preprocess_text(text):
    text = normalize_punctuation(text)
    text = re.sub(r'（[^）]*）', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    return text

def classify_subject_type(word, human_words, abstract_words):
    if any(h in word for h in human_words): return "human"
    if any(a in word for a in abstract_words): return "abstract"
    return "entity"

def adjust_confidence(base, sentence):
    if "好き" in sentence or "思う" in sentence or "感じ" in sentence: base -= 0.2
    if sentence.endswith("だ。") or sentence.endswith("である。"): base -= 0.1
    return round(max(0.0, min(base, 1.0)), 2)

def get_compound_prefix(token):
    compounds = [t.text for t in token.lefts if t.dep_ == "compound"]
    return "".join(compounds) + token.text if compounds else token.text

def analyze_sentence_structure(doc, human_words, abstract_words):
    results = []
    for sent in doc.sents:
        subj = root = obj = subj_conf = subj_type = None

        for token in sent:
            if token.dep_ == "ROOT":
                root = token.lemma_

            if token.dep_ == "nsubj" and subj is None:
                if token.head.dep_ in ("acl", "relcl", "compound", "advcl"):
                    continue
                if token.head.lemma_ in EXPERIENCER_PREDICATES:
                    obj = token.text
                    continue
                if token.head.head.pos_ in ("NOUN", "PROPN"):
                    continue
                subj = get_compound_prefix(token)
                subj_type = classify_subject_type(subj, human_words, abstract_words)
                if any(c.text == "が" for c in token.children):
                    base = 0.95
                elif any(c.text == "は" for c in token.children):
                    base = 0.85
                else:
                    base = 0.55
                subj_conf = adjust_confidence(base, sent.text)

            if token.dep_ == "topic" and subj is None:
                if token.head.dep_ in ("acl", "relcl", "compound", "advcl"):
                    continue
                if token.head.lemma_ in EXPERIENCER_PREDICATES:
                    obj = token.text
                    continue
                subj = get_compound_prefix(token)
                subj_type = classify_subject_type(subj, human_words, abstract_words)
                base = 0.80
                subj_conf = adjust_confidence(base, sent.text)

            if token.dep_ in ("obj", "dobj"):
                obj = token.text

        if subj is None:
            tokens = list(sent)
            for i, token in enumerate(tokens):
                if token.text == "は" and i > 0:
                    candidate = tokens[i - 1]
                    if candidate.pos_ in ("NOUN", "PROPN") and candidate.text not in TOPIC_EXCLUSIONS:
                        compounds = [t.text for t in candidate.lefts if t.dep_ == "compound"]
                        subj = "".join(compounds) + candidate.text if compounds else candidate.text
                        subj_type = classify_subject_type(subj, human_words, abstract_words)
                        subj_conf = adjust_confidence(0.70, sent.text)
                        break

        results.append({
            "sentence": sent.text,
            "subject": subj,
            "subject_type": subj_type,
            "subject_confidence": subj_conf,
            "predicate": root,
            "object": obj
        })
    return results

def detect_subject_missing(structure_results):
    return [{"sentence": r["sentence"], "alert": "主語不在アラート"}
            for r in structure_results if r["subject"] is None]

def coherence_score(s1, s2, doc1, doc2, embedding_score):
    semantic_score = embedding_score
    pred1 = {t.lemma_ for t in doc1 if t.dep_ == "ROOT"}
    subj2 = {t.lemma_ for t in doc2 if t.dep_ in ("nsubj", "topic")}
    noun_overlap = (
        {t.lemma_ for t in doc1 if t.pos_ in ("NOUN", "PROPN") and t.lemma_ not in FUNCTION_WORDS} &
        {t.lemma_ for t in doc2 if t.pos_ in ("NOUN", "PROPN") and t.lemma_ not in FUNCTION_WORDS}
    )
    dependency_score = 0.0
    if pred1 & subj2:
        dependency_score += 0.6
    if noun_overlap:
        dependency_score += min(0.4, len(noun_overlap) * 0.1)
    dependency_score = min(1.0, dependency_score)
    STRONG_CLAIMS = ["有意", "増加した", "減少した", "確認された", "示された",
                     "証明された", "観察された", "検出された", "p<", "p <"]
    strong_hit = any(w in s2 for w in STRONG_CLAIMS)
    weak_hit = any(w in s2 for w in WEAK_CLAIMS_FLAT)
    if strong_hit:
        claim_score = 1.0
    elif weak_hit:
        claim_score = 0.2
    else:
        claim_score = 0.5
    final_score = (
        semantic_score   * 0.30 +
        dependency_score * 0.50 +
        claim_score      * 0.20
    )
    return round(min(1.0, final_score), 3)

def classify_jump(score, threshold):
    if score < 0: return "トピックジャンプ"
    if score < threshold: return "弱い関連"
    return None

def semantic_coherence(doc, mode_cfg, nlp, model, util):
    threshold = mode_cfg["coherence_threshold"]
    min_len = 5
    sentences = [sent.text for sent in doc.sents]
    sent_docs = [nlp(s) for s in sentences]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    alerts = []
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        if any(m in s2 for m in SKIP_MARKERS):
            continue
        emb_score = util.cos_sim(embeddings[i], embeddings[i+1]).item()
        adj_threshold = threshold * 0.5 if any(m in s1 for m in DISCOURSE_MARKERS) else threshold
        if len(s1) < min_len or len(s2) < min_len:
            words1 = {t.lemma_ for t in sent_docs[i] if t.pos_ in ("NOUN", "VERB", "PROPN")}
            words2 = {t.lemma_ for t in sent_docs[i+1] if t.pos_ in ("NOUN", "VERB", "PROPN")}
            if len(words1 & words2) == 0 and len(words1) > 0 and len(words2) > 0:
                alerts.append({"pair": (s1, s2), "similarity": None, "alert": "トピックジャンプ（短文）"})
            continue
        score = coherence_score(s1, s2, sent_docs[i], sent_docs[i+1], emb_score)
        jump_type = classify_jump(score, adj_threshold)
        if jump_type:
            alerts.append({"pair": (s1, s2), "similarity": score, "alert": jump_type})
    return alerts

def detect_weak_claims(doc, nlp, negative_patterns=None):
    alerts = []
    for sent in doc.sents:
        for category, patterns in WEAK_CLAIM_PATTERNS.items():
            if any(p in sent.text for p in patterns):
                if negative_patterns:
                    tokens = list(nlp(sent.text))
                    sentence_ngrams = {
                        tokens[i].lemma_ + "_" + tokens[i + 1].lemma_
                        for i in range(len(tokens) - 1)
                    }
                    if sentence_ngrams & negative_patterns.get(category, set()):
                        continue
                alerts.append({
                    "sentence": sent.text,
                    "alert": "弱い主張",
                    "category": category,
                    "reason": ACTION_TEMPLATES[category]
                })
                break
    return alerts

def suggest_split_points(sentence):
    parts = sentence.split("、")
    if len(parts) < 4:
        return None
    SPLIT_TRIGGERS = ["にあたり", "ことで", "として", "ために", "ながら", "けれど", "ものの"]
    best_split = None
    for i, part in enumerate(parts):
        if any(t in part for t in SPLIT_TRIGGERS):
            best_split = i + 1
            break
    if not best_split:
        best_split = len(parts) // 2
    s1 = "、".join(parts[:best_split]) + "。"
    s2 = "、".join(parts[best_split:])
    return f"①{s1}\n②{s2}"

def detect_long_sentences(doc, max_len=MAX_SENTENCE_LEN):
    results = []
    for sent in doc.sents:
        if len(sent.text) > max_len:
            split_suggestion = suggest_split_points(sent.text)
            results.append({
                "sentence": sent.text,
                "alert": "長文アラート",
                "reason": f"1文が{len(sent.text)}文字あります。2〜3文に分割することを検討してください。",
                "split_suggestion": split_suggestion
            })
    return results

def detect_structure_issues(doc, structure_results):
    from constants import STRUCTURE_PATTERNS
    issues = []
    sentences = [sent.text for sent in doc.sents]

    # ① 目的が最後に来ている
    last = sentences[-1] if sentences else ""
    for trigger in STRUCTURE_PATTERNS["目的が最後"]["triggers"]:
        if trigger in last and len(sentences) > 1:
            issues.append({
                "pattern": "目的が最後",
                "sentence": last,
                "advice": STRUCTURE_PATTERNS["目的が最後"]["advice"],
                "template": STRUCTURE_PATTERNS["目的が最後"]["template"]
            })
            break

   # ② 一文に動詞が3つ以上かつ読点が3つ以上
    for sent in doc.sents:
        verb_count = sum(1 for t in sent if t.pos_ == "VERB")
        comma_count = sent.text.count("、")
        if verb_count >= 3 and comma_count >= 3:
            issues.append({
                "pattern": "一文に複数動詞",
                "sentence": sent.text,
                "advice": f"この文に動詞が{verb_count}つ、読点が{comma_count}つあります。1文につき動詞1つを目安に分割してください。",
                "template": None
            })

    # ③ 接続詞の連続
    consecutive = 0
    for sent in sentences:
        if any(t in sent for t in STRUCTURE_PATTERNS["接続詞の連続"]["triggers"]):
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 2:
            issues.append({
                "pattern": "接続詞の連続",
                "sentence": sent,
                "advice": STRUCTURE_PATTERNS["接続詞の連続"]["advice"],
                "template": None
            })
            break

# ④ 主語の不統一
    subjects = [r["subject"] for r in structure_results if r["subject"]]
    if len(set(subjects)) > 2 and len(subjects) >= 3:
        msg = STRUCTURE_PATTERNS["主語の不統一"]["advice"]
        cnt = len(set(subjects))
        issues.append({
            "pattern": "主語の不統一",
            "sentence": "（文章全体）",
            "advice": f"主語が{cnt}種類あります。{msg}",
            "template": None
        })
    return issues

def abstract_density(doc):
    tokens = [t for t in doc if not t.is_punct]
    if not tokens: return 0
    all_patterns = [p for patterns in WEAK_CLAIM_PATTERNS.values() for p in patterns]
    return round(sum(1 for t in tokens if t.text in all_patterns) / len(tokens), 3)

def detect_poetic_density(doc, mode_cfg):
    threshold = mode_cfg["poetic_threshold"]
    min_tokens = mode_cfg["min_tokens"]
    tokens = [t for t in doc if not t.is_punct]
    if len(tokens) < min_tokens: return None
    adj_adv = sum(1 for t in tokens if t.pos_ in ("ADJ", "ADV"))
    content = sum(1 for t in tokens if t.pos_ in ("NOUN", "PROPN", "VERB"))
    if content == 0: return None
    ratio = adj_adv / content
    density = abstract_density(doc)
    alert = None
    if ratio > threshold: alert = "ポエム検知（深夜ラブレター）"
    elif density > 0.15: alert = "抽象度過多"
    return {"ratio": round(ratio, 3), "abstract_density": density, "alert": alert}

def compute_score(structure, subject_alerts, coherence_alerts, weak_claims, poetic, long_sentences):
    confs = [s["subject_confidence"] for s in structure if s["subject_confidence"]]
    subject_score = max(0, round(sum(confs)/len(confs)*100, 1) - len(subject_alerts)*15) if confs else 0.0
    logic_score = max(0, 100 - sum(
        30 if a["alert"] in ("トピックジャンプ", "トピックジャンプ（短文）") else 15
        for a in coherence_alerts
    ))
    poetic_penalty = 0
    if poetic and poetic["alert"]:
        poetic_penalty = min(30, int(poetic.get("ratio", 0)*30) + int(poetic.get("abstract_density", 0)*50))
    long_penalty = min(30, len(long_sentences) * 15)
    overall = round((subject_score*0.4 + logic_score*0.6) - poetic_penalty - long_penalty, 1)
    return {
        "総合スコア": max(0, overall),
        "主語明確性": subject_score,
        "論理整合性": logic_score,
        "ポエム減点": poetic_penalty,
        "長文減点": long_penalty,
        "問題数": {
            "主語": len(subject_alerts),
            "論理": len(coherence_alerts),
            "表現": len(weak_claims),
            "長文": len(long_sentences),
            "ポエム": 1 if poetic and poetic["alert"] else 0
        }
    }

def full_analysis(text, mode_cfg, nlp, model, util, negative_patterns=None):
    clean_text = preprocess_text(text)
    doc = nlp(clean_text)
    from constants import HUMAN_WORDS, ABSTRACT_WORDS
    structure = analyze_sentence_structure(doc, HUMAN_WORDS, ABSTRACT_WORDS)
    subject_alerts = detect_subject_missing(structure)
    coherence_alerts = semantic_coherence(doc, mode_cfg, nlp, model, util)
    weak_claims = detect_weak_claims(doc, nlp, negative_patterns)
    long_sentences = detect_long_sentences(nlp(text))
    structure_issues = detect_structure_issues(doc, structure)
    poetic = detect_poetic_density(doc, mode_cfg)
    score = compute_score(structure, subject_alerts, coherence_alerts, weak_claims, poetic, long_sentences)
    return {
        "score": score,
        "structure": structure,
        "subject_alerts": subject_alerts,
        "coherence_alerts": coherence_alerts,
        "weak_claims": weak_claims,
        "long_sentences": long_sentences,
        "structure_issues": structure_issues,
        "poetic": poetic,
    }
