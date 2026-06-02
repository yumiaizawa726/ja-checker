import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer, util
from constants import MODES
from log_manager import load_log, record_action, build_negative_patterns
from analyzer import full_analysis
from suggestions import generate_suggestions

st.set_page_config(page_title="日本語文章チェッカー", layout="wide")

# ─────────────────────────────────────────
# モデルロード
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return nlp, model

nlp, model = load_models()

# ─────────────────────────────────────────
# session_state 初期化
# ─────────────────────────────────────────
def init_state():
    for key, default in {
        "result": None,
        "suggestions": [],
        "working_sentences": [],   # [{index, original, current, adopted}]
        "original_text": "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_state()

# ─────────────────────────────────────────
# 採用処理：文インデックスで該当文を更新
# ─────────────────────────────────────────
def apply_adoption(suggestion_sentence: str, after_text: str, alert_type: str, action: str):
    """提案文に対応するworking_sentencesのcurrentを更新する"""
    for ws in st.session_state["working_sentences"]:
        if ws["original"] == suggestion_sentence and not ws["adopted"]:
            ws["current"] = after_text
            ws["adopted"] = True
            break
    record_action(alert_type, suggestion_sentence, action, "adopted")

def apply_skip(suggestion_sentence: str, alert_type: str, action: str):
    record_action(alert_type, suggestion_sentence, action, "rejected")

def get_working_text() -> str:
    return "".join(ws["current"] for ws in st.session_state["working_sentences"])

# ─────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")
    mode_name = st.selectbox("モードを選択", list(MODES.keys()))
    mode_cfg = MODES[mode_name]
    st.caption(mode_cfg["description"])

    st.divider()
    st.header("📖 使い方")
    st.markdown("""
1. テキストを入力して **「分析する」** を押す
2. 右側にスコアと改善提案が表示される
3. 各提案の **「採用」** を押すと左側の作業テキストに反映される
4. 左下の **「コピー」** で最終文章を取り出す
""")

    st.divider()
    st.header("📈 採用ログ")
    log = load_log()
    if log:
        adopted = len([e for e in log if e["action"] == "adopted"])
        rejected = len([e for e in log if e["action"] == "rejected"])
        st.metric("総記録数", len(log))
        st.metric("採用率", f"{round(adopted / len(log) * 100, 1)}%")
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
        negative_patterns = {}

# ─────────────────────────────────────────
# メインレイアウト：左（作業テキスト） / 右（分析・提案）
# ─────────────────────────────────────────
st.title("🇯🇵 日本語文章チェッカー")

left_col, right_col = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════
# 左カラム：入力 & 作業テキスト
# ══════════════════════════════════════════
with left_col:
    st.subheader("📝 テキスト入力")
    text_input = st.text_area(
        "分析するテキストを入力",
        height=200,
        placeholder="例：この研究結果は重要な示唆を持つ可能性がある。検討が必要である。",
        label_visibility="collapsed",
    )

    run = st.button("分析する", type="primary", use_container_width=True)

    if run:
        if text_input.strip():
            with st.spinner("分析中..."):
                log = load_log()
                negative_patterns = build_negative_patterns(log, nlp)
                result = full_analysis(text_input, mode_cfg, nlp, model, util, negative_patterns)
                suggestions = generate_suggestions(result)

                # result["structure"] から文リストを取る（full_analysis内部と完全一致）
                sentences = [s["sentence"] for s in result["structure"]]

                st.session_state["result"] = result
                st.session_state["suggestions"] = suggestions
                st.session_state["original_text"] = text_input
                st.session_state["working_sentences"] = [
                    {"index": i, "original": s, "current": s, "adopted": False}
                    for i, s in enumerate(sentences)
                ]
        else:
            st.warning("テキストを入力してください")

    # 作業テキスト表示（分析後）
    if st.session_state["working_sentences"]:
        st.divider()
        st.subheader("✏️ 作業テキスト")
        st.caption("採用した提案が反映されます")

        # 文ごとに色分け表示
        for ws in st.session_state["working_sentences"]:
            if ws["adopted"]:
                st.success(ws["current"])
            else:
                st.text(ws["current"])

        st.divider()
        final_text = get_working_text()
        st.text_area(
            "最終テキスト（コピー用）",
            value=final_text,
            height=150,
            key="final_output",
        )

# ══════════════════════════════════════════
# 右カラム：スコア・分析・改善提案
# ══════════════════════════════════════════
with right_col:
    if st.session_state["result"] is None:
        st.info("← テキストを入力して「分析する」を押してください")
    else:
        result = st.session_state["result"]
        suggestions = st.session_state["suggestions"]
        score = result["score"]

        # ── スコア ──────────────────────────────
        st.subheader("📊 スコア")
        st.caption("総合スコア = 主語明確性(40%) + 論理整合性(60%) − 各減点。100点満点。")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "総合スコア",
            score["総合スコア"],
            help="主語明確性と論理整合性を総合した点数。減点要素を差し引いた最終スコアです。"
        )
        c2.metric(
            "主語明確性",
            score["主語明確性"],
            help="各文の主語が明示されているか。主語が不在の文1件につき−20pt。"
        )
        c3.metric(
            "論理整合性",
            score["論理整合性"],
            help="文間の論理的なつながり。トピックジャンプで−30pt、弱い関連で−15pt。"
        )

        c4, c5, c6 = st.columns(3)
        c4.metric(
            "長文減点",
            f"−{score['長文減点']}",
            help=f"80文字超の文1件につき−15pt（最大−30）。該当文: {score['問題数']['長文']}件"
        )
        c5.metric(
            "構造減点",
            f"−{score['構造減点']}",
            help=f"接続詞の連続・複数動詞・主語不統一など。該当: {score['問題数']['構造']}件"
        )
        c6.metric(
            "ポエム減点",
            f"−{score['ポエム減点']}",
            help="抽象語・形容詞・副詞が多すぎる場合の減点（最大−30）。"
        )

        p = score["問題数"]
        st.caption(
            f"検出件数 ／ 主語: {p['主語']}件　論理: {p['論理']}件　"
            f"表現: {p['表現']}件　長文: {p['長文']}件　"
            f"構造: {p['構造']}件　ポエム: {p['ポエム']}件"
        )

        # ── 改善提案 ────────────────────────────
        if suggestions:
            st.divider()
            st.subheader("💡 改善提案")
            st.caption(f"提案 {len(suggestions)} 件 ／ 採用すると左の作業テキストに反映されます")

            badge_colors = {
                "トピックジャンプ": "🔴",
                "トピックジャンプ（短文）": "🔴",
                "弱い関連": "🟡",
                "根拠不足": "🟠",
                "比較軸不足": "🟠",
                "因果不足": "🟠",
                "主体不明": "🟣",
                "抽象語逃げ": "🟣",
                "結論ジャンプ": "🔴",
                "主語不在": "🔵",
            }

            for i, s in enumerate(suggestions):
                icon = badge_colors.get(s["alert_type"], "⚪")

                # 採用済みかチェック
                already_adopted = any(
                    ws["adopted"] and ws["original"] == s["sentence"]
                    for ws in st.session_state["working_sentences"]
                )

                with st.container(border=True):
                    # ヘッダー行
                    h1, h2 = st.columns([3, 1])
                    h1.markdown(f"**{icon} {s['alert_type']}**")
                    if already_adopted:
                        h2.success("✅ 採用済み")

                    # 対象文 → 提案内容 → 書き換え例 の順
                    st.markdown("**対象文**")
                    st.error(s["sentence"])

                    st.markdown("**今すぐやること**")
                    st.info(s["one_action"])

                    # Before / After サイドバイサイド
                    st.markdown("**Before / After**")
                    ba_left, ba_right = st.columns(2)
                    with ba_left:
                        st.caption("Before")
                        st.markdown(
                            f"<div style='background:#ffd7d7;padding:8px;border-radius:6px;"
                            f"font-size:0.9em;color:#333'>{s['sentence']}</div>",
                            unsafe_allow_html=True,
                        )
                    with ba_right:
                        st.caption("After")
                        after_display = s["example"].replace("\n", "<br>")
                        st.markdown(
                            f"<div style='background:#d4edda;padding:8px;border-radius:6px;"
                            f"font-size:0.9em;color:#333'>{after_display}</div>",
                            unsafe_allow_html=True,
                        )

                    # ボタン行
                    if not already_adopted:
                        btn1, btn2 = st.columns([1, 1])
                        if btn1.button("✅ 採用", key=f"adopt_{i}", use_container_width=True):
                            # トピックジャンプは複数例があるので最初の1行だけ採用
                            after_text = s["example"].split("\n")[0]
                            # 【因果】などのプレフィックスを除去
                            if after_text.startswith("【"):
                                after_text = after_text.split("】", 1)[-1].strip()
                            apply_adoption(s["sentence"], after_text, s["alert_type"], s["one_action"])
                            st.rerun()
                        if btn2.button("スキップ", key=f"skip_{i}", use_container_width=True):
                            apply_skip(s["sentence"], s["alert_type"], s["one_action"])
                            st.toast("スキップしました")

        # ── 構文解析（折りたたみ） ───────────────
        with st.expander("🔍 構文解析の詳細", expanded=False):
            st.caption("確信度：GiNZAが主語と判定した確からしさ（100%に近いほど明確）")
            for s in result["structure"]:
                with st.expander(s["sentence"][:40] + "..." if len(s["sentence"]) > 40 else s["sentence"]):
                    cols = st.columns(4)
                    cols[0].markdown(f"**主語** {s['subject'] or '―'}")
                    cols[1].markdown(f"**述語** {s['predicate'] or '―'}")
                    cols[2].markdown(f"**目的語** {s['object'] or '―'}")
                    conf = s['subject_confidence']
                    cols[3].markdown(f"**確信度** {str(int(conf * 100)) + '%' if conf else '―'}")

        # ── 長文アラート（折りたたみ） ────────────
        if result.get("long_sentences"):
            with st.expander(f"📏 長文アラート（{len(result['long_sentences'])}件）", expanded=False):
                for a in result["long_sentences"]:
                    st.warning(a["reason"])
                    st.markdown(f"**対象文：** {a['sentence']}")
                    if a.get("split_suggestion"):
                        st.markdown("**分割案**")
                        st.success(a["split_suggestion"].replace("\n", "\n\n"))
                    else:
                        st.info("主語・述語・目的語を1セットに絞り、残りは別文に分割してください。")

        # ── 構造アドバイス（折りたたみ） ──────────
        if result.get("structure_issues"):
            with st.expander(f"📐 構造アドバイス（{len(result['structure_issues'])}件）", expanded=False):
                for issue in result["structure_issues"]:
                    st.warning(f"**{issue['pattern']}**　{issue['advice']}")
                    st.markdown(f"**対象：** {issue['sentence']}")
                    if issue.get("template"):
                        st.success(f"書き直しテンプレート：{issue['template']}")
