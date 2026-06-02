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
        "working_sentences": [],
        "original_text": "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_state()

# ─────────────────────────────────────────
# ヘルパー関数
# ─────────────────────────────────────────
def apply_adoption(suggestion_sentence: str, after_text: str, alert_type: str, action: str):
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

def get_original_text() -> str:
    return "".join(ws["original"] for ws in st.session_state["working_sentences"])

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
1. テキストを貼り付けて **「分析する」** を押す
2. 改善提案を上から順に確認する
3. **「採用」** を押すと右側の改善後テキストに反映される
4. 全部確認したら右側のテキストをコピーして完成
""")

    st.divider()
    st.header("📈 採用ログ")
    log = load_log()
    if log:
        adopted = len([e for e in log if e["action"] == "adopted"])
        st.metric("総記録数", len(log))
        st.metric("採用率", f"{round(adopted / len(log) * 100, 1)}%")
        negative_patterns = build_negative_patterns(log, nlp)
        total_patterns = sum(len(v) for v in negative_patterns.values())
        if total_patterns > 0:
            st.caption(f"抑制パターン: {total_patterns}件学習済み")
    else:
        st.caption("まだログがありません")
        negative_patterns = {}

# ─────────────────────────────────────────
# タイトル & 入力エリア（最上部・全幅）
# ─────────────────────────────────────────
st.title("🇯🇵 日本語文章チェッカー")
st.caption("文章を貼り付けて「分析する」を押すと、改善提案が表示されます。提案を採用しながら文章を仕上げてください。")

with st.container():
    text_input = st.text_area(
        "分析するテキストを貼り付けてください",
        height=140,
        placeholder="例：本研究では、これまでさまざまな研究を行ってきた。結果として重要な示唆が得られた可能性がある。",
    )
    run = st.button("分析する", type="primary", use_container_width=True)

if run:
    if text_input.strip():
        with st.spinner("分析中..."):
            log = load_log()
            negative_patterns = build_negative_patterns(log, nlp)
            result = full_analysis(text_input, mode_cfg, nlp, model, util, negative_patterns)
            suggestions = generate_suggestions(result)
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

# ─────────────────────────────────────────
# 分析結果エリア
# ─────────────────────────────────────────
if st.session_state["result"] is not None:
    result = st.session_state["result"]
    suggestions = st.session_state["suggestions"]
    score = result["score"]
    p = score["問題数"]

    # ── 問題件数サマリー ──────────────────────
    st.divider()
    total_issues = p["主語"] + p["論理"] + p["長文"] + p["構造"]
    adopted_count = sum(1 for ws in st.session_state["working_sentences"] if ws["adopted"])
    remaining = len(suggestions) - adopted_count

    if total_issues == 0:
        st.success("✅ 改善提案はありません。そのまま使用できます。")
    else:
        col_s, col_l, col_lo, col_st, col_rem = st.columns(5)
        col_s.metric("主語の問題", f"{p['主語']}件", help="主語が省略・不明瞭な文")
        col_l.metric("論理の問題", f"{p['論理']}件", help="文間のつながりが弱い箇所")
        col_lo.metric("長文", f"{p['長文']}件", help="80文字を超える長い文")
        col_st.metric("構造の問題", f"{p['構造']}件", help="接続詞の連続・複数動詞など")
        col_rem.metric(
            "残り提案",
            f"{remaining}件",
            delta=f"-{adopted_count}件採用済み" if adopted_count > 0 else None,
            delta_color="normal",
        )

    # スコア詳細は折りたたみ・デフォルト非表示
    with st.expander("📊 スコア詳細（上級者向け）", expanded=False):
        st.caption("総合スコア = 主語明確性(40%) + 論理整合性(60%) − 各減点。100点満点。")
        c1, c2, c3 = st.columns(3)
        c1.metric("総合スコア", score["総合スコア"])
        c2.metric("主語明確性", score["主語明確性"], help="主語不在1件につき−20pt")
        c3.metric("論理整合性", score["論理整合性"], help="トピックジャンプ−30pt、弱い関連−15pt")
        c4, c5, c6 = st.columns(3)
        c4.metric("長文減点", f"−{score['長文減点']}")
        c5.metric("構造減点", f"−{score['構造減点']}")
        c6.metric("ポエム減点", f"−{score['ポエム減点']}")

    st.divider()

    # ── メインレイアウト：改善提案（左） / Before・After（右） ──
    main_left, main_right = st.columns([1, 1], gap="large")

    # ══════════════════════════════════════════
    # 左：改善提案（主役）
    # ══════════════════════════════════════════
    with main_left:
        st.subheader("💡 改善提案")

        if not suggestions:
            st.success("改善提案はありません。")
        else:
            st.caption(f"{len(suggestions)}件 ／ 上から順に確認してください")

            badge = {
                "トピックジャンプ": "🔴",
                "トピックジャンプ（短文）": "🔴",
                "結論ジャンプ": "🔴",
                "弱い関連": "🟡",
                "根拠不足": "🟠",
                "比較軸不足": "🟠",
                "因果不足": "🟠",
                "主体不明": "🟣",
                "抽象語逃げ": "🟣",
                "主語不在": "🔵",
            }

            for i, s in enumerate(suggestions):
                icon = badge.get(s["alert_type"], "⚪")
                already_adopted = any(
                    ws["adopted"] and ws["original"] == s["sentence"]
                    for ws in st.session_state["working_sentences"]
                )

                with st.container(border=True):
                    # ヘッダー行：種別 + 採用済みバッジ
                    h1, h2 = st.columns([3, 1])
                    h1.markdown(f"**{icon} {s['alert_type']}**")
                    if already_adopted:
                        h2.success("✅ 採用済み")

                    # ① 対象文
                    st.caption("対象文")
                    st.error(s["sentence"])

                    # ② 何が問題か
                    advice_text = s["advice"][0] if isinstance(s["advice"], list) else s["advice"]
                    st.caption("何が問題か")
                    st.warning(advice_text)

                    # ③ 今すぐやること
                    st.caption("今すぐやること")
                    st.info(s["one_action"])

                    # ④ 改善例（Before / After）
                    st.caption("改善例")
                    ba_l, ba_r = st.columns(2)
                    with ba_l:
                        st.markdown("**Before**")
                        st.markdown(
                            f"<div style='background:#ffd7d7;padding:8px;border-radius:6px;"
                            f"font-size:0.88em;color:#333;line-height:1.6'>{s['sentence']}</div>",
                            unsafe_allow_html=True,
                        )
                    with ba_r:
                        st.markdown("**After**")
                        after_display = s["example"].replace("\n", "<br>")
                        st.markdown(
                            f"<div style='background:#d4edda;padding:8px;border-radius:6px;"
                            f"font-size:0.88em;color:#333;line-height:1.6'>{after_display}</div>",
                            unsafe_allow_html=True,
                        )

                    # ⑤ 採用／却下ボタン
                    if not already_adopted:
                        st.markdown("")
                        btn1, btn2 = st.columns(2)
                        if btn1.button("✅ 採用", key=f"adopt_{i}", use_container_width=True, type="primary"):
                            after_text = s["example"].split("\n")[0]
                            if after_text.startswith("【"):
                                after_text = after_text.split("】", 1)[-1].strip()
                            apply_adoption(s["sentence"], after_text, s["alert_type"], s["one_action"])
                            st.toast("✅ 採用しました。右側のテキストに反映されました。")
                            st.rerun()
                        if btn2.button("却下", key=f"skip_{i}", use_container_width=True):
                            apply_skip(s["sentence"], s["alert_type"], s["one_action"])
                            st.toast("却下しました")

        # 構文解析（折りたたみ・最下部）
        st.markdown("")
        with st.expander("🔍 構文解析の詳細（上級者向け）", expanded=False):
            st.caption("確信度：GiNZAが主語と判定した確からしさ（100%に近いほど明確）")
            for s in result["structure"]:
                label = s["sentence"][:35] + "…" if len(s["sentence"]) > 35 else s["sentence"]
                with st.expander(label):
                    cols = st.columns(4)
                    cols[0].markdown(f"**主語** {s['subject'] or '―'}")
                    cols[1].markdown(f"**述語** {s['predicate'] or '―'}")
                    cols[2].markdown(f"**目的語** {s['object'] or '―'}")
                    conf = s["subject_confidence"]
                    cols[3].markdown(f"**確信度** {str(int(conf * 100)) + '%' if conf else '―'}")

        if result.get("long_sentences"):
            with st.expander(f"📏 長文アラート（{len(result['long_sentences'])}件）", expanded=False):
                for a in result["long_sentences"]:
                    st.warning(a["reason"])
                    st.markdown(f"**対象文：** {a['sentence']}")
                    if a.get("split_suggestion"):
                        st.success(a["split_suggestion"].replace("\n", "\n\n"))
                    else:
                        st.info("主語・述語・目的語を1セットに絞り、残りは別文に分割してください。")

        if result.get("structure_issues"):
            with st.expander(f"📐 構造アドバイス（{len(result['structure_issues'])}件）", expanded=False):
                for issue in result["structure_issues"]:
                    st.warning(f"**{issue['pattern']}**　{issue['advice']}")
                    st.markdown(f"**対象：** {issue['sentence']}")
                    if issue.get("template"):
                        st.success(f"書き直しテンプレート：{issue['template']}")

    # ══════════════════════════════════════════
    # 右：原文 / 改善後テキスト（常時表示・DeepL形式）
    # ══════════════════════════════════════════
    with main_right:
        st.subheader("📄 テキスト比較")
        st.caption("採用するたびに右側が更新されます")

        top_l, top_r = st.columns(2)

        with top_l:
            st.markdown("**原文**")
            original_text = get_original_text()
            st.markdown(
                f"<div style='background:#f8f8f8;border:1px solid #ddd;border-radius:8px;"
                f"padding:14px;min-height:220px;font-size:0.92em;line-height:1.9;"
                f"color:#555;white-space:pre-wrap'>{original_text}</div>",
                unsafe_allow_html=True,
            )

        with top_r:
            adopted_count = sum(1 for ws in st.session_state["working_sentences"] if ws["adopted"])
            label = f"**改善後**　" + (f"（{adopted_count}件反映済み）" if adopted_count > 0 else "")
            st.markdown(label)

            lines_html = ""
            for ws in st.session_state["working_sentences"]:
                if ws["adopted"]:
                    lines_html += (
                        f"<span style='background:#c8f7c5;border-radius:3px;"
                        f"padding:1px 2px'>{ws['current']}</span>"
                    )
                else:
                    lines_html += ws["current"]

            st.markdown(
                f"<div style='background:#f0fff4;border:1px solid #b2dfdb;border-radius:8px;"
                f"padding:14px;min-height:220px;font-size:0.92em;line-height:1.9;"
                f"color:#333;white-space:pre-wrap'>{lines_html}</div>",
                unsafe_allow_html=True,
            )

        # コピー用
        st.markdown("")
        final_text = get_working_text()
        st.text_area(
            "コピー用（改善後テキスト全文）",
            value=final_text,
            height=130,
        )
