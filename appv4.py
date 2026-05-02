import streamlit as st
import spacy
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="日本語文章チェッカー", layout="wide")

@st.cache_resource
def load_models():
    nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return nlp, model

nlp, model = load_models()

import re

def normalize_punctuation(text):
    # 句点の統一
    text = text.replace("．", "。")
    text = text.replace("；", "。")
    # ピリオドは日本語文字の後のみ変換
    text = re.sub(r'([ぁ-んァ-ン一-龥])\.', r'\1。', text)
    # 読点の統一
    text = text.replace(",", "、")
    text = text.replace("，", "、")
    # 感嘆符・疑問符
    text = text.replace("!", "。")
    text = text.replace("？", "。")
    text = text.replace("?", "。")
    return text

def preprocess_text(text):
    text = normalize_punctuation(text)
    text = re.sub(r'（[^）]*）', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    return text

# ── 定数 ────────────────────────────────────
DISCOURSE_MARKERS = [
    "しかし", "一方で", "そのため", "したがって", "だから", "でも", "ところが",
    "さらに", "例えば", "なお", "また", "加えて", "一方"
]
SKIP_MARKERS = ["しかし", "一方で", "ところが", "でも", "けれど", "しかしながら"]
HUMAN_WORDS = ["彼", "彼女", "私", "僕", "俺", "あなた", "君", "先生", "社長"]
ABSTRACT_WORDS = ["それ", "これ", "あれ", "こと", "もの", "ため", "わけ"]
EXPERIENCER_PREDICATES = ["好き", "嫌い", "怖い", "得意", "苦手", "好む", "嫌う", "必要", "大切", "大事", "不安", "心配"]
WEAK_PATTERNS = [
    ("重要な示唆を持つ可能性がある", "具体的なデータに基づき重要な示唆を持つ"),
    ("重要であると考えられる",       "具体的な結果に基づき重要であると判断される"),
    ("可能性がある",                 "データが示すように〜の可能性がある"),
    ("示唆される",                   "上記の結果から示唆される"),
    ("検討が必要である",             "今後〇〇の観点から検討が必要である"),
    ("必要である",                   "〇〇という理由から必要である"),
    ("有用である",                   "〇〇の場面において有用である"),
]
ACTION_TEMPLATES = {
    "根拠不足":             "この主張を支持する先行研究を1件引用するか、具体的な数値を追加してください。",
    "比較軸不足":           "何と比較して『高い／低い』のか、比較対象を明示してください。",
    "因果不足":             "なぜその結果が次の結論につながるのか、理由を1文追加してください。",
    "主体不明":             "『考えられる』の主体を明示してください。例：『本研究の結果は〜を示唆する』",
    "抽象語逃げ":           "『可能性がある』を具体的な数値か先行研究の引用に置き換えてください。",
    "結論ジャンプ":         "この結論に至る根拠を1文追加してください。",
    "トピックジャンプ":     "前の文となぜつながるのか、理由を1文追加してください。",
    "トピックジャンプ（短文）": "前の文となぜつながるのか、理由を1文追加してください。",
    "弱い主張":             "抽象的な表現を具体的な数値・事実・引用に置き換えてください。",
    "主語不在":             "この文の主体（誰が・何が）を文頭に明示してください。",
    "弱い関連":             "前の文との関係を『なぜなら』『一方で』などで明示してください。",
}
WEAK_CLAIM_PATTERNS = {
    "根拠不足":     ["重要である", "有用である", "効果的である", "優れている", "高い効果"],
    "比較軸不足":   ["より良い", "最も", "より高い", "より低い", "より多い"],
    "因果不足":     ["そのため", "したがって", "よって", "結果として", "これにより"],
    "主体不明":     ["考えられる", "思われる", "推察される", "想定される"],
    "抽象語逃げ":   ["可能性がある", "示唆される", "観点から", "課題がある", "検討が必要"],
    "結論ジャンプ": ["明らかである", "証明された", "示された", "明らかにした"],  # 「確認された」を削除
}
WEAK_CLAIMS_FLAT = [p for patterns in WEAK_CLAIM_PATTERNS.values() for p in patterns]
MODES = {
    "論文（厳密）": {
        "coherence_threshold": 0.4,
        "poetic_threshold":    0.5,
        "min_tokens":          8,
        "description":         "査読論文向け。抽象表現に厳しく反応します。"
    },
    "論文（標準）": {
        "coherence_threshold": 0.3,
        "poetic_threshold":    0.6,
        "min_tokens":          10,
        "description":         "学位論文・レポート向け。標準的な検出レベルです。"
    },
    "科研費申請書": {
        "coherence_threshold": 0.3,
        "poetic_threshold":    0.7,
        "min_tokens":          12,
        "description":         "申請書向け。意義・必要性の記述を重点チェックします。"
    },
    "一般文章": {
        "coherence_threshold": 0.2,
        "poetic_threshold":    0.6,
        "min_tokens":          10,
        "description":         "ビジネス文書・一般文章向けです。"
    }
}
MAX_SENTENCE_LEN = 80
LOG_FILE = "suggestion_log.json"

# ── ログ ────────────────────────────────────
def load_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

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

# ── 分析エンジン ─────────────────────────────
def classify_subject_type(word):
    if any(h in word for h in HUMAN_WORDS): return "human"
    if any(a in word for a in ABSTRACT_WORDS): return "abstract"
    return "entity"

def adjust_confidence(base, sentence):
    if "好き" in sentence or "思う" in sentence or "感じ" in sentence: base -= 0.2
    if sentence.endswith("だ。") or sentence.endswith("である。"): base -= 0.1
    return round(max(0.0, min(base, 1.0)), 2)

def analyze_sentence_structure(doc):
    results = []
    for sent in doc.sents:
        subj = root = obj = subj_conf = subj_type = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token.lemma_
            if token.dep_ == "nsubj":
                if token.head.dep_ in ("acl", "relcl", "compound", "advcl"):
                    continue
                if token.head.lemma_ in EXPERIENCER_PREDICATES:
                    obj = token.text
                    continue
                if token.head.head.pos_ in ("NOUN", "PROPN"):
                    continue
                subj = token.text
                subj_type = classify_subject_type(token.text)
                if any(c.text == "が" for c in token.children):
                    base = 0.95
                elif any(c.text == "は" for c in token.children):
                    base = 0.75
                else:
                    base = 0.55
                subj_conf = adjust_confidence(base, sent.text)
            if token.dep_ in ("obj", "dobj"):
                obj = token.text
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

FUNCTION_WORDS = {"こと", "もの", "ため", "わけ", "はず", "つもり", "よう"}

def coherence_score(s1, s2, doc1, doc2, embedding_score):
    # ① Semantic similarity
    semantic_score = embedding_score

    # ② Dependency overlap
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

    # ③ Claim strength
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

def semantic_coherence(doc, mode_cfg):
    threshold = mode_cfg["coherence_threshold"]
    min_len = 5
    sentences = [sent.text for sent in doc.sents]
    sent_docs = [nlp(s) for s in sentences]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    alerts = []

    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]

        # 逆接・転換系のみスキップ（追加系はスキップしない）
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

def detect_weak_claims(doc):
    alerts = []
    for sent in doc.sents:
        for category, patterns in WEAK_CLAIM_PATTERNS.items():
            if any(p in sent.text for p in patterns):
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

    # 述語っぽい区切りを探す
    SPLIT_TRIGGERS = ["にあたり", "ことで", "として", "ために", "ながら", "けれど", "ものの"]

    best_split = None
    for i, part in enumerate(parts):
        if any(t in part for t in SPLIT_TRIGGERS):
            best_split = i + 1
            break

    # 見つからなければ中間で切る
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
    # 長文ペナルティ追加
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
def full_analysis(text, mode_cfg):
    clean_text = preprocess_text(text)
    doc = nlp(clean_text)
    structure = analyze_sentence_structure(doc)
    subject_alerts = detect_subject_missing(structure)
    coherence_alerts = semantic_coherence(doc, mode_cfg)
    weak_claims = detect_weak_claims(doc)
    long_sentences = detect_long_sentences(nlp(text))
    poetic = detect_poetic_density(doc, mode_cfg)
    score = compute_score(structure, subject_alerts, coherence_alerts, weak_claims, poetic, long_sentences)
    return {
        "score": score,
        "structure": structure,
        "subject_alerts": subject_alerts,
        "coherence_alerts": coherence_alerts,
        "weak_claims": weak_claims,
        "long_sentences": long_sentences,
        "poetic": poetic
    }

# ── 改善提案 ─────────────────────────────────
def suggest_for_topic_jump(s1, s2, subj2=None):
    example = f"【因果】{s1} そのため、{s2}\n【対比】{s1} 一方で、{s2}\n【例示】{s1} 例えば、{s2}"
    return {
        "advice": ["話題の接続が弱いため、両文をつなぐ1文を追加してください。"],
        "example": example
    }

def suggest_for_weak_relation(sentence, category="抽象語逃げ"):
    action = ACTION_TEMPLATES.get(category, ACTION_TEMPLATES["弱い主張"])
    example = sentence
    matched = False
    for vague, concrete in WEAK_PATTERNS:
        if vague in sentence:
            example = sentence.replace(vague, concrete)
            matched = True
            break
    if not matched:
        # 「弱い関連」は前文との接続を明示する提案に変える
        example = f"前の調査結果を踏まえ、{sentence}"
    return {"advice": [action], "example": example}

def suggest_for_missing_subject(sentence):
    for marker in DISCOURSE_MARKERS:
        if sentence.startswith(marker):
            rest = sentence[len(marker):].lstrip("、")
            return {
                "advice": ["接続詞の後に主語を明示してください。"],
                "example": f"{marker}、（主語）は{rest}"
            }
    return {
        "advice": ["主語が省略されています。誰が・何がの主体を明示してください。"],
        "example": f"（主語）は、{sentence}"
    }

def generate_suggestions(result):
    suggestions = []
    seen = set()
    subj_map = {s["sentence"]: s["subject"] for s in result["structure"]}

    for alert in result.get("coherence_alerts", []):
        s1, s2 = alert["pair"]
        if s2 in seen: continue
        seen.add(s2)
        if "トピックジャンプ" in alert["alert"]:
            sg = suggest_for_topic_jump(s1, s2, subj_map.get(s2))
        else:
            sg = suggest_for_weak_relation(s2)
        suggestions.append({
            "alert_type": alert["alert"],
            "sentence": s2,
            "one_action": ACTION_TEMPLATES.get(alert["alert"], "文章を見直してください。"),
            **sg
        })

    for alert in result.get("weak_claims", []):
        if alert["sentence"] in seen: continue
        seen.add(alert["sentence"])
        sg = suggest_for_weak_relation(alert["sentence"], alert.get("category", "抽象語逃げ"))
        suggestions.append({
            "alert_type": alert["category"],
            "sentence": alert["sentence"],
            "one_action": ACTION_TEMPLATES.get(alert["category"], "文章を見直してください。"),
            **sg
        })

    for alert in result.get("subject_alerts", []):
        if alert["sentence"] in seen: continue
        seen.add(alert["sentence"])
        sg = suggest_for_missing_subject(alert["sentence"])
        suggestions.append({
            "alert_type": "主語不在",
            "sentence": alert["sentence"],
            "one_action": ACTION_TEMPLATES["主語不在"],
            **sg
        })

    return suggestions

# ── Streamlit UI ─────────────────────────────
st.title("🇯🇵 日本語文章チェッカー")
st.caption("構文・論理・表現の問題を検出し、改善提案を生成します")

with st.sidebar:
    st.header("⚙️ 設定")
    mode_name = st.selectbox("モードを選択", list(MODES.keys()))
    mode_cfg = MODES[mode_name]
    st.caption(mode_cfg["description"])
    st.divider()
    st.header("📈 採用ログ")
    log = load_log()
    if log:
        adopted = len([e for e in log if e["action"] == "adopted"])
        rejected = len([e for e in log if e["action"] == "rejected"])
        st.metric("総記録数", len(log))
        st.metric("採用率", f"{round(adopted/len(log)*100, 1)}%")
        st.metric("採用", adopted)
        st.metric("却下", rejected)
    else:
        st.caption("まだログがありません")

text_input = st.text_area(
    "分析するテキストを入力してください",
    height=150,
    placeholder="例：この研究結果は重要な示唆を持つ可能性がある。検討が必要である。"
)

if st.button("分析する", type="primary"):
    if text_input.strip():
        with st.spinner("分析中..."):
            result = full_analysis(text_input, mode_cfg)
            suggestions = generate_suggestions(result)
            st.session_state["result"] = result
            st.session_state["suggestions"] = suggestions
    else:
        st.warning("テキストを入力してください")

if "result" in st.session_state:
    result = st.session_state["result"]
    suggestions = st.session_state["suggestions"]
    score = result["score"]

    st.subheader("📊 スコア")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("総合スコア", score["総合スコア"])
    c2.metric("主語明確性", score["主語明確性"])
    c3.metric("論理整合性", score["論理整合性"])
    c4.metric("長文減点", score["長文減点"])
    c5.metric("ポエム減点", score["ポエム減点"])
    p = score["問題数"]
    st.caption(f"問題数 ／ 主語: {p['主語']}件　論理: {p['論理']}件　表現: {p['表現']}件　長文: {p['長文']}件　ポエム: {p['ポエム']}件")

    st.subheader("🔍 構文解析")
    for s in result["structure"]:
        with st.expander(s["sentence"]):
            cols = st.columns(4)
            cols[0].markdown(f"**主語**　{s['subject'] or '―'}")
            cols[1].markdown(f"**述語**　{s['predicate'] or '―'}")
            cols[2].markdown(f"**目的語**　{s['object'] or '―'}")
            cols[3].markdown(f"**確信度**　{str(int(s['subject_confidence']*100))+'%' if s['subject_confidence'] else '―'}")

    if result.get("long_sentences"):
        st.subheader("📏 長文アラート")
        for a in result["long_sentences"]:
            with st.expander(f"⚠️ {a['sentence'][:30]}..."):
                st.warning(a["reason"])
                st.markdown("**対象文**")
                st.markdown(a["sentence"])
                if a.get("split_suggestion"):
                    st.markdown("**分割案**")
                    st.success(a["split_suggestion"].replace("\n", "\n\n"))
                else:
                    st.info("この文の主語・述語・目的語を1セットに絞り、残りは別文に分割してください。")

    if suggestions:  # ← ここが同じインデントレベルになっているか確認
        st.subheader("💡 改善提案")
        badge_colors = {
            "トピックジャンプ":         "🔴",
            "トピックジャンプ（短文）":  "🔴",
            "弱い関連":                 "🟡",
            "根拠不足":                 "🟠",
            "比較軸不足":               "🟠",
            "因果不足":                 "🟠",
            "主体不明":                 "🟣",
            "抽象語逃げ":               "🟣",
            "結論ジャンプ":             "🔴",
            "主語不在":                 "🔵",
        }
        for i, s in enumerate(suggestions):
            icon = badge_colors.get(s["alert_type"], "⚪")
            with st.expander(f"{icon} {s['alert_type']} ／ {s['sentence'][:25]}..."):
                st.markdown("### 📝 今すぐやること")
                st.info(s["one_action"])
                st.markdown("**書き換え例**")
                st.success(s["example"].replace("\n", "\n\n"))
                with st.expander("対象文を確認"):
                    st.markdown(s["sentence"])
                col1, col2 = st.columns([1, 5])
                if col1.button("✅ 採用", key=f"adopt_{i}"):
                    record_action(s["alert_type"], s["sentence"], s["one_action"], "adopted")
                    st.toast("採用しました")
                if col2.button("スキップ", key=f"skip_{i}"):
                    record_action(s["alert_type"], s["sentence"], s["one_action"], "rejected")
                    st.toast("スキップしました")
