import re
from constants import DISCOURSE_MARKERS, ACTION_TEMPLATES, WEAK_PATTERNS

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
        # 接続詞を除去してからプレースホルダを置く
        CONJUNCTIONS = ["しかし、", "また、", "さらに、", "そして、", "ただし、", "なお、", "一方、", "つまり、"]
        body = sentence
        for conj in CONJUNCTIONS:
            if body.startswith(conj):
                body = body[len(conj):]
                break
        if category in ("結論ジャンプ", "因果不足"):
            example = (
                f"【根拠文の型】「〇〇を用いて検証した結果、△△であった。」
"
                f"【例】「ウェスタンブロット解析の結果、タンパク質発現量に有意な変化は認められなかった。」
"
                f"↓ この文の前に根拠文を追加
"
                f"{body}"
            )
        elif category in ("比較軸不足",):
            example = (
                f"【比較文の型】「〇〇と比較して、△△において□□であった。」
"
                f"【例】「対照群と比較して、処置群では発現量が2倍以上増加した。」
"
                f"↓ この文の前に比較の根拠を追加
"
                f"{body}"
            )
        elif category in ("根拠不足",):
            example = (
                f"【根拠文の型】「〇〇（文献番号）によれば、△△とされている。」
"
                f"【例】「先行研究（Smith et al., 2020）では、同様の条件下で△△が報告されている。」
"
                f"↓ この文の前に根拠を追加
"
                f"{body}"
            )
        else:
            example = (
                f"【追加文の型】「〇〇であることから、△△と考えられる。」
"
                f"↓ この文の前に具体的な内容を追加
"
                f"{body}"
            )
    return {"advice": [action], "example": example}

def suggest_for_missing_subject(sentence):
    """
    主語不在文に対して改善例を生成する。
    トピック句（〜は、〜では、など）がある場合はそのまま残し
    文頭に（主語）プレースホルダを置く。
    接続詞のみの場合はそれを除去してから補う。
    """
    CONJUNCTIONS = ["しかし、", "また、", "さらに、", "そして、", "ただし、", "なお、", "一方、", "つまり、"]
    rest = sentence
    for conj in CONJUNCTIONS:
        if rest.startswith(conj):
            rest = rest[len(conj):]
            break
    if not rest:
        rest = sentence

    # トピック句がある場合：（主語）を文頭に置くだけ（は、を重複させない）
    topic_match = re.match(r'^(.{1,8}[はでにもを]、)', rest)
    if topic_match:
        example = f"（主語）{rest}"
    else:
        example = f"（主語）は、{rest}"

    return {
        "advice": ["主語が省略されています。誰が・何がの主体を明示してください。"],
        "example": example
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
