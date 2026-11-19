    display: inline-flex; align-items: center; gap: 8px; padding: 7px 12px; margin-bottom: 15px;
    border: 1px solid rgba(103, 232, 249, 0.28); border-radius: 999px;
    background: rgba(8, 27, 51, 0.62); color: #8be9f5;
    font: 700 11px 'Space Grotesk', sans-serif; letter-spacing: 0.16em; text-transform: uppercase;
}
.hero-title {
    margin: 0; color: #f5f8ff;
    font: 800 clamp(28px, 4vw, 44px)/1.15 'Space Grotesk', 'Be Vietnam Pro', sans-serif;
    letter-spacing: -0.045em;
}
.hero-title span {
    display: block; margin-top: 6px;
    background: linear-gradient(90deg, #93c5fd 0%, #67e8f9 52%, #a5b4fc 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-description { max-width: 920px; margin: 15px 0 0 0; color: #b8c8e2; font-size: 14px; line-height: 1.8; }
.hero-tags { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 19px; }
.hero-tag {
    padding: 7px 11px; border: 1px solid rgba(135, 169, 216, 0.22); border-radius: 8px;
    background: rgba(5, 17, 35, 0.48); color: #ccdcf5; font-size: 11px; font-weight: 600;
}

[data-testid="stTextArea"] textarea {
    color: #edf4ff !important; background: rgba(8, 22, 43, 0.88) !important;
    border: 1px solid rgba(121, 160, 215, 0.34) !important; border-radius: 12px !important;
    line-height: 1.65 !important;
}
[data-testid="stTextArea"] textarea:focus { border-color: #67c9f2 !important; box-shadow: 0 0 0 1px rgba(103, 201, 242, 0.22) !important; }
[data-testid="stTextArea"] label, [data-testid="stWidgetLabel"] p { color: #dce8fb !important; font-weight: 600 !important; }
.stButton > button[kind="primary"] {
    min-height: 52px; border: 1px solid rgba(147, 197, 253, 0.5); border-radius: 12px;
    background: linear-gradient(100deg, #2563eb 0%, #0ea5e9 54%, #6366f1 100%);
    color: white; font-weight: 800; letter-spacing: 0.045em;
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.25); transition: transform 160ms ease, box-shadow 160ms ease;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 14px 34px rgba(37, 99, 235, 0.34); border-color: #93c5fd; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stCodeBlock"] { border: 1px solid var(--rl-border); border-radius: 12px; }
hr { border-color: rgba(124, 160, 211, 0.22) !important; }
small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--rl-muted) !important; }

@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
    .hero-shell { padding: 23px 20px; border-radius: 17px; }
}
"""
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <section class="hero-shell">
      <div class="hero-kicker">✦ Artificial Intelligence &amp; Law</div>
      <h1 class="hero-title">REPLAYLAW-VN
        <span>CÔNG NGHỆ AI KIẾN TẠO TƯƠNG LAI PHÁP LUẬT</span>
      </h1>
      <p class="hero-description">
        Nền tảng AI hỗ trợ hợp nhất, phân tích và xử lý văn bản pháp luật Việt Nam.
        Kết hợp mô hình ngôn ngữ ViT5-LoRA với bộ máy AST và hệ thống 5 quy tắc logic,
        hướng đến quy trình xử lý minh bạch và nhất quán.
      </p>
      <div class="hero-tags">
        <span class="hero-tag">AI LANGUAGE MODEL · ViT5-LoRA</span>
        <span class="hero-tag">AST RULE ENGINE · R1–R5</span>
        <span class="hero-tag">LEGAL DOCUMENT AI</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.caption("REPLAYLAW-VN WORKSPACE · Hệ thống xử lý và hợp nhất văn bản pháp luật")

with st.sidebar:
    st.subheader("Trạng thái hệ thống")
    st.write(f"Thiết bị: **{DEVICE.upper()}**")
    if LORA_REPO:
        st.success(f"LoRA Repository: {LORA_REPO}")
        if LORA_SUBFOLDER:
            st.caption(f"Thư mục adapter: {LORA_SUBFOLDER}")
    else:
        st.warning("Chưa cấu hình HF_LORA_REPO trong Streamlit Secrets.")
    st.caption("Model nền: VietAI/vit5-base")

left_col, right_col = st.columns([5, 7], gap="large")
with left_col:
    st.subheader("1. Văn bản pháp luật gốc (Base Document)")
    base_text = st.text_area("Văn bản gốc", value=SAMPLE_BASE, height=230, label_visibility="collapsed")
    st.subheader("2. Nội dung văn bản sửa đổi, bổ sung (Amendment)")
    amend_text = st.text_area("Nội dung sửa đổi", value=SAMPLE_AMEND_OK, height=190, label_visibility="collapsed")

with right_col:
    st.subheader("Cấu trúc cây AST phân tích được")
    ast_placeholder = st.empty()
    st.subheader("3. Tầng Neural (ViT5 → DSL) và kiểm tra 5 luật (R1–R5)")
    log_placeholder = st.empty()

run_button = st.button("HỢP NHẤT VĂN BẢN TỰ ĐỘNG", type="primary", use_container_width=True)

if run_button:
    if not base_text.strip() or not amend_text.strip():
        st.warning("Vui lòng nhập cả văn bản gốc và nội dung sửa đổi.")
    elif not LORA_REPO:
        st.error(
            "Chưa kết nối Model Repository. Vào Streamlit Cloud → Settings → Secrets và thêm:\n\n"
            'HF_LORA_REPO = "username/model-repository"\n'
            'HF_TOKEN = "hf_..."  # chỉ thêm nếu repository Private'
        )
    else:
        try:
            with st.spinner("Đang tải mô hình và xử lý các câu lệnh sửa đổi..."):
                logs, diff_html, ast_text = interactive_consolidate(base_text, amend_text)
            ast_placeholder.text_area("AST", value=ast_text, height=110, label_visibility="collapsed")
            log_placeholder.text_area("Nhật ký xử lý", value=logs, height=350, label_visibility="collapsed")
            st.subheader("4. Văn bản sau hợp nhất (Visual Diff)")
            st.markdown(diff_html, unsafe_allow_html=True)
        except Exception as exc:
            st.exception(exc)

st.divider()
st.caption("REPLAYLAW-VN | ViT5-LoRA · DSL · AST · R1–R5")
