"""
app_streamlit.py — HemoScan Streamlit Application
Sistem Deteksi Anemia berbasis AI
Database: Supabase | Hosting: Streamlit Cloud
"""

import os
import uuid
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import modul lokal
from inference import load_model_components, predict_from_pil
from database import (
    register_user, login_user, save_scan,
    get_history, get_stats, get_all_users, get_scan_trend
)

# ─── Konfigurasi Halaman ──────────────────────────────────────────
st.set_page_config(
    page_title="HemoScan — Deteksi Anemia AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS Premium ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f1a 100%);
        min-height: 100vh;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
        border-right: 1px solid rgba(220, 38, 38, 0.2);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(220, 38, 38, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s ease;
    }
    .card:hover {
        border-color: rgba(220, 38, 38, 0.4);
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(220,38,38,0.1), rgba(15,15,30,0.8));
        border: 1px solid rgba(220,38,38,0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ef4444;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Result cards */
    .result-anemia {
        background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(15,15,30,0.9));
        border: 2px solid rgba(220,38,38,0.5);
        border-radius: 20px;
        padding: 28px;
    }
    .result-normal {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(15,15,30,0.9));
        border: 2px solid rgba(34,197,94,0.5);
        border-radius: 20px;
        padding: 28px;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-anemia .result-title { color: #ef4444; }
    .result-normal .result-title { color: #22c55e; }

    /* Risk badge */
    .badge-high { background: rgba(220,38,38,0.2); color: #ef4444; border: 1px solid rgba(220,38,38,0.4); padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; display: inline-block; }
    .badge-medium { background: rgba(234,179,8,0.2); color: #eab308; border: 1px solid rgba(234,179,8,0.4); padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; display: inline-block; }
    .badge-low { background: rgba(34,197,94,0.2); color: #22c55e; border: 1px solid rgba(34,197,94,0.4); padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; display: inline-block; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(220,38,38,0.4);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: rgba(220,38,38,0.5) !important;
        box-shadow: 0 0 0 2px rgba(220,38,38,0.1) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(220,38,38,0.3);
        border-radius: 12px;
        padding: 16px;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(220,38,38,0.6);
    }

    /* Page header */
    .page-header {
        padding: 20px 0 16px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 24px;
    }
    .page-header h1 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 0;
    }
    .page-header p {
        color: #64748b;
        font-size: 0.9rem;
        margin: 4px 0 0;
    }

    /* Logo text */
    .logo-text {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ef4444, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Nav item */
    .nav-item {
        padding: 10px 16px;
        border-radius: 10px;
        cursor: pointer;
        font-weight: 500;
        transition: background 0.2s;
    }
    .nav-item:hover {
        background: rgba(220,38,38,0.1);
    }
    .nav-item.active {
        background: rgba(220,38,38,0.15);
        border-left: 3px solid #dc2626;
    }

    /* Labels */
    .stSelectbox label, .stTextInput label,
    .stNumberInput label, .stTextArea label,
    .stRadio label, .stCheckbox label,
    .stFileUploader label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(220,38,38,0.2);
        border-radius: 12px;
        padding: 16px;
    }

    /* Table */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Info/success/error boxes */
    .stAlert {
        border-radius: 12px !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #dc2626, #ef4444) !important;
        border-radius: 999px !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #dc2626 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────
def init_session():
    defaults = {
        "logged_in": False,
        "user_email": "",
        "user_role": "user",
        "display_name": "",
        "current_page": "scan",
        "auth_page": "login",
        "scan_result": None,
        "show_result": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Load Model ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model AI...")
def load_model():
    try:
        return load_model_components()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None


# ─── Helpers ─────────────────────────────────────────────────────
def get_risk_badge(risk_level: str) -> str:
    badges = {
        "high": '<span class="badge-high">🔴 Risiko Tinggi</span>',
        "medium": '<span class="badge-medium">🟡 Risiko Sedang</span>',
        "low": '<span class="badge-low">🟢 Risiko Rendah</span>',
    }
    return badges.get(risk_level, "")


def is_google_drive_link(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def gdrive_to_direct(url: str) -> str:
    """Convert Google Drive share link to direct view link."""
    if "drive.google.com/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=view&id={file_id}"
    return url


# ─── Sidebar ─────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="logo-text">🔬 HemoScan</div>', unsafe_allow_html=True)
        st.caption("Sistem Deteksi Anemia AI")
        st.divider()

        if st.session_state.logged_in:
            st.markdown(f"""
            <div style="background:rgba(220,38,38,0.08);border:1px solid rgba(220,38,38,0.2);
                 border-radius:10px;padding:12px 14px;margin-bottom:16px;">
                <div style="font-weight:600;color:#f1f5f9;">{st.session_state.display_name}</div>
                <div style="font-size:0.75rem;color:#64748b;">{st.session_state.user_email}</div>
                <div style="margin-top:6px;">
                    {'<span style="background:rgba(234,179,8,0.2);color:#eab308;border-radius:999px;padding:2px 8px;font-size:0.7rem;font-weight:600;">👑 Admin</span>' 
                     if st.session_state.user_role == "admin" 
                     else '<span style="background:rgba(99,102,241,0.2);color:#818cf8;border-radius:999px;padding:2px 8px;font-size:0.7rem;font-weight:600;">👤 User</span>'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Navigation
            pages = [
                ("🔬 Scan Baru", "scan"),
                ("📋 Riwayat Scan", "history"),
            ]
            if st.session_state.user_role == "admin":
                pages.append(("📊 Dashboard Admin", "dashboard"))
                pages.append(("👥 Manajemen User", "users"))

            for label, page in pages:
                is_active = st.session_state.current_page == page
                style = "border-left:3px solid #dc2626;background:rgba(220,38,38,0.1);" if is_active else ""
                if st.button(label, key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.session_state.show_result = False
                    st.rerun()

            st.divider()
            if st.button("🚪 Keluar", key="logout_btn", use_container_width=True):
                for key in ["logged_in", "user_email", "user_role", "display_name", "scan_result", "show_result"]:
                    st.session_state[key] = False if key == "logged_in" else (None if key == "scan_result" else "")
                st.session_state.show_result = False
                st.rerun()
        else:
            st.info("Silakan masuk untuk menggunakan HemoScan")

        st.divider()
        st.caption("HemoScan v2.0 · Powered by AI")
        st.caption("EfficientNetB0 + XGBoost + Supabase")


# ═══════════════════════════════════════════════════════════════════
# HALAMAN: LOGIN / REGISTER
# ═══════════════════════════════════════════════════════════════════
def page_auth():
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 30px;">
            <div style="font-size:3rem;">🔬</div>
            <div style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,#ef4444,#f97316);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                HemoScan
            </div>
            <div style="color:#64748b;font-size:0.9rem;margin-top:4px;">
                Sistem Deteksi Anemia Berbasis AI
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tab switch
        col_login, col_reg = st.columns(2)
        with col_login:
            if st.button("🔑 Masuk", use_container_width=True, key="tab_login"):
                st.session_state.auth_page = "login"
                st.rerun()
        with col_reg:
            if st.button("📝 Daftar", use_container_width=True, key="tab_register"):
                st.session_state.auth_page = "register"
                st.rerun()

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.session_state.auth_page == "login":
            st.markdown("### 🔑 Masuk ke HemoScan")
            email = st.text_input("Email", placeholder="nama@email.com", key="login_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")

            if st.button("Masuk →", key="login_submit"):
                if not email or not password:
                    st.error("Isi email dan password!")
                else:
                    with st.spinner("Memverifikasi..."):
                        result = login_user(email, password)
                    if result["success"]:
                        user = result["user"]
                        st.session_state.logged_in = True
                        st.session_state.user_email = user["email"]
                        st.session_state.user_role = user["role"]
                        st.session_state.display_name = user["display_name"]
                        st.session_state.current_page = "scan"
                        st.success("✅ Login berhasil!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")

            st.markdown("""
            <div style="margin-top:16px;padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;font-size:0.8rem;color:#64748b;">
            🔒 <strong style="color:#94a3b8;">Demo Admin:</strong> admin@hemoscan.com / admin123
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("### 📝 Daftar Akun Baru")
            name = st.text_input("Nama Lengkap", placeholder="John Doe", key="reg_name")
            email = st.text_input("Email", placeholder="nama@email.com", key="reg_email")
            password = st.text_input("Password", type="password", placeholder="Min. 6 karakter", key="reg_password")
            password2 = st.text_input("Konfirmasi Password", type="password", placeholder="Ulangi password", key="reg_password2")

            if st.button("Daftar Sekarang →", key="reg_submit"):
                if not all([name, email, password, password2]):
                    st.error("Semua kolom wajib diisi!")
                elif len(password) < 6:
                    st.error("Password minimal 6 karakter!")
                elif password != password2:
                    st.error("Password tidak cocok!")
                elif len(name.strip()) < 2:
                    st.error("Nama minimal 2 karakter!")
                else:
                    with st.spinner("Mendaftarkan akun..."):
                        result = register_user(email, password, name)
                    if result["success"]:
                        st.success("✅ Pendaftaran berhasil! Silakan masuk.")
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")

        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HALAMAN: SCAN BARU
# ═══════════════════════════════════════════════════════════════════
def page_scan():
    st.markdown("""
    <div class="page-header">
        <h1>🔬 Scan Anemia Baru</h1>
        <p>Upload gambar konjungtiva dan isi data pasien untuk analisis AI</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.show_result and st.session_state.scan_result:
        render_result(st.session_state.scan_result)
        st.divider()
        if st.button("🔄 Scan Baru", key="new_scan"):
            st.session_state.show_result = False
            st.session_state.scan_result = None
            st.rerun()
        return

    components = load_model()
    if components is None:
        st.error("Model AI tidak tersedia. Pastikan file model sudah ada di folder `model_output/`.")
        return

    with st.form("scan_form", clear_on_submit=False):
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown("#### 📁 Upload Gambar Konjungtiva")
            uploaded_file = st.file_uploader(
                "Pilih gambar konjungtiva",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                img = Image.open(uploaded_file)
                st.image(img, caption="Preview gambar", use_column_width=True)

            st.markdown("#### 🔗 Link Google Drive (Opsional)")
            gdrive_link = st.text_input(
                "Tempel link Google Drive untuk menyimpan referensi gambar",
                placeholder="https://drive.google.com/file/d/...",
                label_visibility="collapsed",
                help="Bagikan gambar di Google Drive dan tempel link di sini untuk disimpan ke database"
            )

        with col_right:
            st.markdown("#### 👤 Data Pasien")
            nama = st.text_input("Nama Pasien", placeholder="Nama lengkap pasien")
            umur = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=30, step=1)
            gender = st.selectbox("Jenis Kelamin", ["M", "F"], format_func=lambda x: "Laki-laki" if x == "M" else "Perempuan")

            st.markdown("#### 🩺 Gejala")
            gejala_options = [
                "Kelelahan", "Pusing", "Sesak napas", "Pucat",
                "Detak jantung cepat", "Nyeri dada", "Tangan/kaki dingin",
                "Sakit kepala", "Nafsu makan turun"
            ]
            selected_gejala = []
            cols = st.columns(2)
            for i, g in enumerate(gejala_options):
                with cols[i % 2]:
                    if st.checkbox(g, key=f"gejala_{i}"):
                        selected_gejala.append(g)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔬 Analisis Sekarang",
            use_container_width=True,
        )

    if submitted:
        if not uploaded_file:
            st.error("❌ Upload gambar konjungtiva terlebih dahulu!")
            return
        if not nama.strip():
            st.error("❌ Nama pasien wajib diisi!")
            return

        with st.spinner("⏳ Menganalisis gambar... Mohon tunggu"):
            try:
                pil_image = Image.open(uploaded_file)
                result = predict_from_pil(pil_image, nama.strip(), int(umur), gender, components)

                scan_id = f"#SCN-{uuid.uuid4().hex[:4].upper()}"
                timestamp = datetime.datetime.now().strftime("%b %d, %Y • %H:%M")
                symptoms_str = ",".join(selected_gejala) if selected_gejala else ""

                # Tentukan image_url
                image_url = gdrive_link.strip() if gdrive_link.strip() else ""

                # Ambil info preprocessing
                from inference import hitung_age_group
                pca = components.get("pca", None)
                n_pca = int(pca.n_components_) if pca is not None else 0
                config = components.get("config", {})
                age_group = hitung_age_group(int(umur))
                age_enc = config.get("age_map", {}).get(age_group, 1)
                gender_enc = config.get("gender_map", {}).get(gender, 0)

                # Simpan ke Supabase
                scan_data = {
                    "scan_id": scan_id,
                    "user_email": st.session_state.user_email,
                    "nama": nama.strip(),
                    "umur": int(umur),
                    "gender": result["gender"],
                    "age_group": result["age_group"],
                    "symptoms": symptoms_str,
                    "prediksi": result["prediksi"],
                    "probabilitas": float(result["probabilitas"]),
                    "threshold": float(result["threshold"]),
                    "risk_level": result["risk_level"],
                    "keterangan": result["keterangan"],
                    "image_url": image_url,
                    "timestamp": timestamp,
                    "img_feat_dim": 1280,
                    "pca_components": n_pca,
                    "age_enc": int(age_enc),
                    "gender_enc": int(gender_enc),
                    "age_map_val": int(age_enc),
                }
                save_result = save_scan(scan_data)

                full_result = {
                    **result,
                    "id": scan_id,
                    "timestamp": timestamp,
                    "image_url": image_url,
                    "symptoms": selected_gejala,
                    "pca_components": n_pca,
                    "saved": save_result["success"],
                }

                st.session_state.scan_result = full_result
                st.session_state.show_result = True
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saat analisis: {str(e)}")


def render_result(result: dict):
    """Tampilkan hasil prediksi dengan UI premium."""
    is_anemia = result["prediksi"] == "Anemia"
    card_class = "result-anemia" if is_anemia else "result-normal"
    icon = "🔴" if is_anemia else "🟢"
    proba_pct = result["probabilitas"] * 100

    st.markdown(f"""
    <div class="{card_class}">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
            <div style="font-size:3rem;">{icon}</div>
            <div>
                <div class="result-title">{result['prediksi']}</div>
                <div>{get_risk_badge(result['risk_level'])}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
                <div style="font-size:0.8rem;color:#94a3b8;">Scan ID</div>
                <div style="font-weight:700;color:#e2e8f0;font-size:0.95rem;">{result['id']}</div>
                <div style="font-size:0.75rem;color:#64748b;">{result['timestamp']}</div>
            </div>
        </div>
        <p style="color:#cbd5e1;font-size:0.95rem;line-height:1.6;margin-bottom:16px;">
            {result['keterangan']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probabilitas gauge
    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_pct,
            number={"suffix": "%", "font": {"color": "#ef4444" if is_anemia else "#22c55e", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569"},
                "bar": {"color": "#ef4444" if is_anemia else "#22c55e"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                    {"range": [30, 60], "color": "rgba(234,179,8,0.15)"},
                    {"range": [60, 100], "color": "rgba(220,38,38,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#f97316", "width": 3},
                    "thickness": 0.8,
                    "value": result["threshold"] * 100
                }
            },
            title={"text": "Probabilitas Anemia", "font": {"color": "#94a3b8", "size": 14}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=10),
            height=250,
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📋 Info Pasien")
        info_items = [
            ("👤 Nama", result["nama"]),
            ("🎂 Usia", f"{result['umur']} tahun"),
            ("⚧ Gender", result["gender"]),
            ("👥 Kelompok", result["age_group"]),
            ("📊 Threshold", f"{result['threshold']:.2f}"),
            ("🧬 PCA", f"{result['pca_components']} komponen"),
        ]
        for label, val in info_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                 padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
                <span style="color:#64748b;font-size:0.85rem;">{label}</span>
                <span style="color:#e2e8f0;font-weight:600;font-size:0.85rem;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        if result.get("symptoms"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🩺 Gejala dilaporkan:**")
            gejala_html = " ".join([f'<span style="background:rgba(99,102,241,0.2);color:#818cf8;padding:3px 10px;border-radius:999px;font-size:0.78rem;margin:2px;display:inline-block;">{g}</span>' for g in result["symptoms"]])
            st.markdown(gejala_html, unsafe_allow_html=True)

    # Link Google Drive
    if result.get("image_url") and is_google_drive_link(result["image_url"]):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.5rem;">📂</span>
            <div>
                <div style="font-weight:600;color:#e2e8f0;">Referensi Gambar Tersimpan</div>
                <a href="{result['image_url']}" target="_blank" style="color:#60a5fa;font-size:0.85rem;">
                    Lihat di Google Drive →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if result.get("saved"):
        st.success("✅ Hasil scan berhasil disimpan ke database Supabase")
    else:
        st.warning("⚠️ Gagal menyimpan ke database. Cek koneksi Supabase.")


# ═══════════════════════════════════════════════════════════════════
# HALAMAN: RIWAYAT
# ═══════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("""
    <div class="page-header">
        <h1>📋 Riwayat Scan</h1>
        <p>Daftar semua hasil scan yang pernah dilakukan</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Memuat riwayat..."):
        history = get_history(st.session_state.user_role, st.session_state.user_email)

    if not history:
        st.markdown("""
        <div class="card" style="text-align:center;padding:48px;">
            <div style="font-size:3rem;">🔬</div>
            <div style="font-size:1.1rem;color:#94a3b8;margin-top:12px;">Belum ada riwayat scan</div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:4px;">Lakukan scan pertama untuk melihat riwayat</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Filter
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        filter_result = st.selectbox("Filter Hasil", ["Semua", "Anemia", "Non-Anemia"])
    with col2:
        filter_risk = st.selectbox("Filter Risiko", ["Semua", "Tinggi", "Sedang", "Rendah"])
    with col3:
        search = st.text_input("🔍 Cari nama pasien...", placeholder="Ketik nama...")

    # Apply filters
    filtered = history
    if filter_result != "Semua":
        filtered = [h for h in filtered if h["prediksi"] == filter_result]
    if filter_risk != "Semua":
        risk_map = {"Tinggi": "high", "Sedang": "medium", "Rendah": "low"}
        filtered = [h for h in filtered if h["risk_level"] == risk_map[filter_risk]]
    if search:
        filtered = [h for h in filtered if search.lower() in h["nama"].lower()]

    st.markdown(f"<div style='color:#64748b;font-size:0.85rem;margin:8px 0;'>Menampilkan {len(filtered)} dari {len(history)} scan</div>", unsafe_allow_html=True)

    # Display as cards
    for item in filtered:
        is_anemia = item["prediksi"] == "Anemia"
        color = "#ef4444" if is_anemia else "#22c55e"
        border = "rgba(220,38,38,0.3)" if is_anemia else "rgba(34,197,94,0.3)"

        with st.expander(
            f"{'🔴' if is_anemia else '🟢'} {item['nama']} — {item['prediksi']} ({item['probabilitas']*100:.1f}%) · {item['timestamp']}",
            expanded=False
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Scan ID:** `{item['id']}`")
                st.markdown(f"**Nama:** {item['nama']}")
                st.markdown(f"**Usia:** {item['umur']} tahun")
                st.markdown(f"**Gender:** {item['gender']}")
            with col2:
                st.markdown(f"**Kelompok:** {item['age_group']}")
                st.markdown(f"**Probabilitas:** {item['probabilitas']*100:.2f}%")
                st.markdown(f"**Threshold:** {item['threshold']:.2f}")
                risk_labels = {"high": "🔴 Tinggi", "medium": "🟡 Sedang", "low": "🟢 Rendah"}
                st.markdown(f"**Risiko:** {risk_labels.get(item['risk_level'], item['risk_level'])}")
            with col3:
                if st.session_state.user_role == "admin":
                    st.markdown(f"**Email:** {item['user_email']}")
                st.markdown(f"**Gejala:** {', '.join(item['symptoms']) if item['symptoms'] else '-'}")
                if item.get("image_url"):
                    if is_google_drive_link(item["image_url"]):
                        st.markdown(f"**Gambar:** [Lihat di Drive]({item['image_url']})")
                    else:
                        st.markdown(f"**Gambar:** {item['image_url']}")

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border-left:3px solid {color};
                 border-radius:0 8px 8px 0;padding:12px 16px;margin-top:8px;font-size:0.88rem;color:#94a3b8;">
                {item['keterangan']}
            </div>
            """, unsafe_allow_html=True)

    # Export CSV
    st.divider()
    if filtered:
        df_export = pd.DataFrame([{
            "Scan ID": h["id"],
            "Nama": h["nama"],
            "Usia": h["umur"],
            "Gender": h["gender"],
            "Prediksi": h["prediksi"],
            "Probabilitas (%)": round(h["probabilitas"] * 100, 2),
            "Risiko": h["risk_level"],
            "Timestamp": h["timestamp"],
            "Email": h.get("user_email", ""),
            "Link Gambar": h.get("image_url", ""),
        } for h in filtered])
        csv = df_export.to_csv(index=False)
        st.download_button(
            "⬇️ Export ke CSV",
            data=csv,
            file_name=f"hemoscan_history_{datetime.date.today()}.csv",
            mime="text/csv"
        )


# ═══════════════════════════════════════════════════════════════════
# HALAMAN: DASHBOARD ADMIN
# ═══════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("""
    <div class="page-header">
        <h1>📊 Dashboard Admin</h1>
        <p>Statistik dan analitik sistem HemoScan</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Memuat statistik..."):
        stats = get_stats()
        trend_data = get_scan_trend()

    # Stat cards
    col1, col2, col3, col4 = st.columns(4)
    stat_items = [
        (col1, "🔬", stats["total_scans"], "Total Scan"),
        (col2, "🔴", stats["anemia_count"], "Terdeteksi Anemia"),
        (col3, "🟢", stats["normal_count"], "Non-Anemia"),
        (col4, "👥", stats["total_users"], "Total Pengguna"),
    ]
    for col, icon, val, label in stat_items:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size:1.8rem;">{icon}</div>
                <div class="stat-number">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if trend_data:
        df = pd.DataFrame(trend_data)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.date
        df["probabilitas"] = df["probabilitas"].astype(float) * 100

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart hasil
            pie_data = df["prediksi"].value_counts().reset_index()
            pie_data.columns = ["Hasil", "Jumlah"]
            fig_pie = px.pie(
                pie_data, names="Hasil", values="Jumlah",
                color="Hasil",
                color_discrete_map={"Anemia": "#ef4444", "Non-Anemia": "#22c55e"},
                title="Distribusi Hasil Prediksi",
                hole=0.5,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8", "family": "Inter"},
                legend={"font": {"color": "#94a3b8"}},
                title_font_color="#e2e8f0",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar gender
            gender_data = df.groupby(["gender", "prediksi"]).size().reset_index(name="count")
            fig_gender = px.bar(
                gender_data, x="gender", y="count", color="prediksi",
                color_discrete_map={"Anemia": "#ef4444", "Non-Anemia": "#22c55e"},
                title="Distribusi Berdasarkan Gender",
                barmode="group",
            )
            fig_gender.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8", "family": "Inter"},
                title_font_color="#e2e8f0",
                xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                legend={"font": {"color": "#94a3b8"}},
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        # Tren waktu
        daily = df.groupby(["created_at", "prediksi"]).size().reset_index(name="count")
        if len(daily["created_at"].unique()) > 1:
            fig_trend = px.line(
                daily, x="created_at", y="count", color="prediksi",
                color_discrete_map={"Anemia": "#ef4444", "Non-Anemia": "#22c55e"},
                title="Tren Scan per Hari",
                markers=True,
            )
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8", "family": "Inter"},
                title_font_color="#e2e8f0",
                xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Distribusi probabilitas
        fig_hist = px.histogram(
            df, x="probabilitas", color="prediksi",
            color_discrete_map={"Anemia": "#ef4444", "Non-Anemia": "#22c55e"},
            title="Distribusi Probabilitas Anemia (%)",
            nbins=20,
            opacity=0.8,
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#94a3b8", "family": "Inter"},
            title_font_color="#e2e8f0",
            xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
            yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
            bargap=0.05,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("Belum ada data scan untuk ditampilkan.")


# ═══════════════════════════════════════════════════════════════════
# HALAMAN: MANAJEMEN USER
# ═══════════════════════════════════════════════════════════════════
def page_users():
    st.markdown("""
    <div class="page-header">
        <h1>👥 Manajemen Pengguna</h1>
        <p>Daftar semua pengguna terdaftar di HemoScan</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Memuat data pengguna..."):
        users = get_all_users()

    if not users:
        st.info("Belum ada pengguna.")
        return

    df_users = pd.DataFrame(users)
    df_users.columns = ["Email", "Nama", "Role", "Terdaftar Sejak"]
    df_users["Terdaftar Sejak"] = pd.to_datetime(df_users["Terdaftar Sejak"]).dt.strftime("%d %b %Y, %H:%M")
    df_users["Role"] = df_users["Role"].map({"admin": "👑 Admin", "user": "👤 User"})

    st.dataframe(
        df_users,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Email": st.column_config.TextColumn("Email", width="medium"),
            "Nama": st.column_config.TextColumn("Nama Lengkap", width="medium"),
            "Role": st.column_config.TextColumn("Role", width="small"),
            "Terdaftar Sejak": st.column_config.TextColumn("Terdaftar Sejak", width="medium"),
        }
    )
    st.caption(f"Total {len(users)} pengguna terdaftar")


# ═══════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    if not st.session_state.logged_in:
        page_auth()
        return

    page = st.session_state.current_page

    if page == "scan":
        page_scan()
    elif page == "history":
        page_history()
    elif page == "dashboard":
        if st.session_state.user_role == "admin":
            page_dashboard()
        else:
            st.error("Akses ditolak. Hanya admin yang bisa mengakses halaman ini.")
    elif page == "users":
        if st.session_state.user_role == "admin":
            page_users()
        else:
            st.error("Akses ditolak.")


if __name__ == "__main__":
    main()
