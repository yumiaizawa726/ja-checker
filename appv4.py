import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer, util

from constants import MODES
from log_manager import load_log, record_action, build_negative_patterns
from analyzer import full_analysis
from suggestions import generate_suggestions

st.set_page_config(page_title="日本語文章チェッカー", layout="wide")

def load_models():
    nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return nlp, model

nlp, model = load_models()

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
        negative_patterns = build_negative_patterns(log, nlp)
        total_patterns = sum(len(v) for v in negative_patterns.values())
        st.divider()
        st.header("🧠 学習状況")
        st.metric("抑制パターン数", total_patterns)
        if total_patterns > 0:
            for alert_type, patterns in negative_patterns.items():
                if patterns:
                    st.caption(f"**{alert_type}**: {len(patterns)}件")
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
            log = load_log()
            negative_patterns = build_negative_patterns(log, nlp)
            result = full_analysis(text_input, mode_cfg, nlp, model, util, negative_patterns)
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

    if result.get("structure_issues"):
        st.subheader("📐 構造アドバイス")
        for issue in result["structure_issues"]:
            with st.expander(f"⚠️ {issue['pattern']} ／ {issue['sentence'][:25]}..."):
                st.warning(issue["advice"])
                if issue.get("template"):
                    st.markdown("**書き直しテンプレート**")
                    st.success(issue["template"])

    if suggestions:
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
