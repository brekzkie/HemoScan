# ================================================================
#  app.py — Aplikasi Deteksi Anemia dari Konjungtiva
#  Jalankan dengan: streamlit run app.py
#
#  Struktur folder yang dibutuhkan:
#  .
#  ├── app.py
#  ├── inference.py
#  └── model_output/
#      ├── pca.pkl
#      ├── config.pkl
#      └── xgb_anemia.json
# ================================================================

import streamlit as st
import numpy as np
from PIL import Image
import time

from inference import load_model_components, predict_from_pil


# ================================================================
# KONFIGURASI HALAMAN
# ================================================================
st.set_page_config(
    page_title  = 'AnemiaDetect',
    page_icon   = '🩸',
    layout      = 'wide',
    initial_sidebar_state = 'collapsed',
)


# ================================================================
# CSS — DESAIN KLINIK MODERN
# Palet: putih bersih + merah medis + abu slate
# Font: DM Sans (body) + Playfair Display (aksen)
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Playfair+Display:ital,wght@0,600;1,500&display=swap');

/* ── Reset & base ───────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #F7F6F3;
    color: #1a1a1a;
}

/* Sembunyikan elemen bawaan Streamlit yang mengganggu */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Halaman utama ──────────────────────── */
.page-wrapper {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* ── Header ─────────────────────────────── */
.app-header {
    background: #1C1C1E;
    color: white;
    padding: 1.4rem 3rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    border-bottom: 3px solid #C0392B;
}
.app-header .logo-dot {
    width: 14px; height: 14px;
    background: #C0392B;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}
.app-header h1 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1.35rem;
    letter-spacing: -0.02em;
    margin: 0;
    color: white;
}
.app-header .subtitle {
    font-size: 0.78rem;
    color: #888;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Main content area ──────────────────── */
.main-content {
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 0;
    flex: 1;
    min-height: calc(100vh - 80px);
}

/* ── Panel kiri (form input) ────────────── */
.left-panel {
    background: white;
    border-right: 1px solid #E8E5DF;
    padding: 2.5rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.panel-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #1C1C1E;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.panel-title em {
    font-style: italic;
    color: #C0392B;
}

.panel-desc {
    font-size: 0.82rem;
    color: #888;
    line-height: 1.6;
    font-weight: 300;
}

/* ── Upload zone ─────────────────────────── */
.upload-zone {
    border: 2px dashed #D5D1CA;
    border-radius: 12px;
    background: #FAFAF8;
    padding: 2rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
    cursor: pointer;
}
.upload-zone:hover { border-color: #C0392B; }
.upload-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.upload-label { font-size: 0.85rem; color: #666; font-weight: 400; }
.upload-hint  { font-size: 0.72rem; color: #aaa; margin-top: 0.25rem; }

/* ── Form fields ─────────────────────────── */
.field-group { display: flex; flex-direction: column; gap: 0.35rem; }
.field-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
}

/* Override Streamlit inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    font-family: 'DM Sans', sans-serif !important;
    border: 1.5px solid #E0DDD7 !important;
    border-radius: 8px !important;
    padding: 0.65rem 0.85rem !important;
    font-size: 0.9rem !important;
    background: #FAFAF8 !important;
    color: #1a1a1a !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #C0392B !important;
    box-shadow: 0 0 0 3px rgba(192,57,43,0.08) !important;
    outline: none !important;
}

[data-testid="stSelectbox"] > div > div {
    font-family: 'DM Sans', sans-serif !important;
    border: 1.5px solid #E0DDD7 !important;
    border-radius: 8px !important;
    background: #FAFAF8 !important;
}

/* ── Tombol analisis ─────────────────────── */
.stButton > button {
    width: 100% !important;
    background: #C0392B !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 1.5rem !important;
    cursor: pointer !important;
    transition: background 0.18s, transform 0.1s !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    background: #A93226 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Panel kanan (hasil) ────────────────── */
.right-panel {
    background: #F7F6F3;
    padding: 2.5rem 3rem;
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

/* ── Kartu pasien ────────────────────────── */
.patient-card {
    background: white;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    border: 1px solid #E8E5DF;
    display: flex;
    align-items: flex-start;
    gap: 1.5rem;
}
.patient-avatar {
    width: 60px; height: 60px;
    border-radius: 50%;
    background: #F2F0EC;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    flex-shrink: 0;
}
.patient-info { flex: 1; }
.patient-name {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1C1C1E;
    margin-bottom: 0.15rem;
}
.patient-meta {
    font-size: 0.8rem;
    color: #888;
    font-weight: 300;
}
.patient-meta span {
    display: inline-block;
    margin-right: 1rem;
}

/* ── Hasil klasifikasi utama ─────────────── */
.result-card {
    border-radius: 16px;
    padding: 2rem 2.25rem;
    border: 1px solid transparent;
    position: relative;
    overflow: hidden;
}
.result-card.anemia {
    background: linear-gradient(135deg, #FEF2F2 0%, #FFF5F5 100%);
    border-color: #FECACA;
}
.result-card.non-anemia {
    background: linear-gradient(135deg, #F0FDF4 0%, #F6FFF8 100%);
    border-color: #BBF7D0;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 5px;
    height: 100%;
    border-radius: 4px 0 0 4px;
}
.result-card.anemia::before     { background: #C0392B; }
.result-card.non-anemia::before { background: #16A34A; }

.result-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-card.anemia     .result-label { color: #991B1B; }
.result-card.non-anemia .result-label { color: #14532D; }

.result-status {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.75rem;
}
.result-card.anemia     .result-status { color: #C0392B; }
.result-card.non-anemia .result-status { color: #16A34A; }

.result-prob {
    font-size: 0.85rem;
    color: #555;
    font-weight: 300;
    line-height: 1.5;
}

/* ── Progress bar probabilitas ──────────── */
.prob-bar-wrap {
    margin-top: 1rem;
}
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #888;
    margin-bottom: 0.3rem;
}
.prob-bar-track {
    height: 6px;
    background: #E8E5DF;
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.prob-bar-fill.anemia     { background: #C0392B; }
.prob-bar-fill.non-anemia { background: #16A34A; }

/* ── Keterangan ──────────────────────────── */
.keterangan-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    border: 1px solid #E8E5DF;
}
.keterangan-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.65rem;
}
.keterangan-text {
    font-size: 0.88rem;
    color: #444;
    line-height: 1.75;
    font-weight: 300;
}

/* ── Info detail (grid 3 kolom) ──────────── */
.detail-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
}
.detail-item {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border: 1px solid #E8E5DF;
}
.detail-item-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 0.25rem;
}
.detail-item-value {
    font-size: 0.95rem;
    font-weight: 500;
    color: #1C1C1E;
}

/* ── Disclaimer ──────────────────────────── */
.disclaimer {
    background: #FFF8E6;
    border: 1px solid #FFE0A3;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.75rem;
    color: #854D0E;
    line-height: 1.6;
    font-weight: 300;
}
.disclaimer strong { font-weight: 600; }

/* ── Empty state ─────────────────────────── */
.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 1rem;
    padding: 4rem;
    opacity: 0.5;
}
.empty-icon { font-size: 3.5rem; }
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #444;
    font-style: italic;
}
.empty-desc { font-size: 0.82rem; color: #888; max-width: 260px; line-height: 1.6; }

/* ── Preview gambar ─────────────────────── */
.img-preview {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E8E5DF;
    background: #F7F6F3;
}
.img-preview img { width: 100%; display: block; }
.img-preview-label {
    font-size: 0.7rem;
    color: #aaa;
    text-align: center;
    padding: 0.5rem;
    font-weight: 300;
}

/* ── Loading state ──────────────────────── */
.loading-text {
    font-size: 0.85rem;
    color: #C0392B;
    font-weight: 400;
    letter-spacing: 0.04em;
}

/* ── Responsive ─────────────────────────── */
@media (max-width: 900px) {
    .main-content { grid-template-columns: 1fr; }
    .left-panel   { border-right: none; border-bottom: 1px solid #E8E5DF; }
    .detail-grid  { grid-template-columns: repeat(2, 1fr); }
    .right-panel  { padding: 2rem 1.5rem; }
    .app-header   { padding: 1.2rem 1.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# LOAD MODEL (cached agar hanya load sekali)
# ================================================================
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model_components()

try:
    components = get_model()
    model_ready = True
except FileNotFoundError as e:
    model_ready = False
    model_error = str(e)


# ================================================================
# HEADER
# ================================================================
st.markdown("""
<div class="app-header">
    <div>
        <h1><span class="logo-dot"></span>AnemiaDetect</h1>
        <div class="subtitle">Sistem Deteksi Anemia dari Konjungtiva Palpebral</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# LAYOUT UTAMA — dua kolom via Streamlit
# ================================================================
col_left, col_right = st.columns([4, 6], gap='large')


# ── PANEL KIRI ──────────────────────────────────────────────────
with col_left:
    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-title">Analisis<br><em>Konjungtiva</em></div>
    <div class="panel-desc" style="margin-bottom:1.5rem">
        Upload foto konjungtiva palpebral inferior pasien, lengkapi data diri,
        dan sistem akan memberikan klasifikasi anemia secara otomatis.
    </div>
    """, unsafe_allow_html=True)

    # ── Upload gambar ──────────────────────────────────────────
    st.markdown('<div class="field-label">Foto Konjungtiva</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label        = 'Upload foto konjungtiva',
        type         = ['jpg', 'jpeg', 'png', 'webp'],
        label_visibility = 'collapsed',
        help         = 'Foto palpebral inferior terbaik dalam kondisi terang',
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption='Preview gambar', use_container_width=True)

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)

    # ── Data pasien ────────────────────────────────────────────
    st.markdown('<div class="field-label">Nama Pasien</div>', unsafe_allow_html=True)
    nama = st.text_input(
        'Nama',
        placeholder     = 'Masukkan nama lengkap pasien',
        label_visibility= 'collapsed',
    )

    col_umur, col_gender = st.columns(2)
    with col_umur:
        st.markdown('<div class="field-label">Umur</div>', unsafe_allow_html=True)
        umur = st.number_input(
            'Umur',
            min_value        = 1,
            max_value        = 120,
            value            = 25,
            step             = 1,
            label_visibility = 'collapsed',
        )
    with col_gender:
        st.markdown('<div class="field-label">Jenis Kelamin</div>', unsafe_allow_html=True)
        gender_label = st.selectbox(
            'Jenis Kelamin',
            options          = ['Laki-laki', 'Perempuan'],
            label_visibility = 'collapsed',
        )
        gender = 'M' if gender_label == 'Laki-laki' else 'F'

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)

    # ── Tombol analisis ────────────────────────────────────────
    btn_ready = uploaded_file is not None and nama.strip() != '' and model_ready
    analisis  = st.button(
        '🔍  Mulai Analisis',
        disabled = not btn_ready,
        use_container_width = True,
    )

    # Pesan error jika model belum tersedia
    if not model_ready:
        st.markdown(f"""
        <div class="disclaimer" style="margin-top:0.75rem">
            <strong>⚠ Model belum tersedia.</strong><br>
            Jalankan dulu <code>anemia_detection_final.py</code> di Kaggle
            untuk menghasilkan folder <code>model_output/</code>.
        </div>
        """, unsafe_allow_html=True)

    elif not btn_ready and not analisis:
        hint_parts = []
        if not uploaded_file: hint_parts.append('foto konjungtiva')
        if not nama.strip():  hint_parts.append('nama pasien')
        if hint_parts:
            st.markdown(f"""
            <div style="font-size:0.75rem;color:#aaa;margin-top:0.5rem;text-align:center">
                Lengkapi: {' dan '.join(hint_parts)}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # ── Disclaimer ─────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        <strong>Perhatian Medis:</strong> Hasil analisis ini bersifat
        <em>skrining awal</em> berbasis kecerdasan buatan dan
        <strong>tidak menggantikan diagnosis dokter</strong>.
        Selalu konsultasikan hasil dengan tenaga medis profesional.
    </div>
    """, unsafe_allow_html=True)


# ── PANEL KANAN ─────────────────────────────────────────────────
with col_right:
    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)

    # Inisialisasi session state untuk menyimpan hasil terakhir
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    # ── Jalankan prediksi ──────────────────────────────────────
    if analisis and btn_ready:
        with st.spinner('Menganalisis gambar konjungtiva...'):
            time.sleep(0.3)  # animasi loading terasa lebih natural
            hasil = predict_from_pil(
                pil_image  = pil_img,
                nama       = nama.strip(),
                umur       = int(umur),
                gender     = gender,
                components = components,
            )
        st.session_state.last_result = hasil

    # ── Tampilkan hasil ────────────────────────────────────────
    hasil = st.session_state.last_result

    if hasil is None:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-title">Hasil analisis<br>akan muncul di sini</div>
            <div class="empty-desc">
                Upload foto konjungtiva dan isi data pasien
                di panel kiri, lalu klik "Mulai Analisis"
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        is_anemia    = hasil['prediksi'] == 'Anemia'
        css_class    = 'anemia' if is_anemia else 'non-anemia'
        prob_pct     = hasil['probabilitas'] * 100
        prob_display = f"{prob_pct:.1f}%"
        avatar       = '♂' if hasil['gender'] == 'Laki-laki' else '♀'

        # 1. Kartu pasien
        st.markdown(f"""
        <div class="patient-card">
            <div class="patient-avatar">{avatar}</div>
            <div class="patient-info">
                <div class="patient-name">{hasil['nama']}</div>
                <div class="patient-meta">
                    <span>🗓 {hasil['umur']} tahun</span>
                    <span>⚥ {hasil['gender']}</span>
                    <span>👥 {hasil['age_group']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Hasil utama
        emoji = '⚠️' if is_anemia else '✅'
        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-label">Hasil Klasifikasi</div>
            <div class="result-status">{emoji}  {hasil['prediksi']}</div>
            <div class="result-prob">
                Probabilitas anemia: <strong>{prob_display}</strong>
                &nbsp;·&nbsp; Threshold: {hasil['threshold']:.2f}
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-label">
                    <span>0%</span><span>50%</span><span>100%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill {css_class}" style="width:{prob_pct:.1f}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Detail grid
        st.markdown(f"""
        <div class="detail-grid">
            <div class="detail-item">
                <div class="detail-item-label">Probabilitas</div>
                <div class="detail-item-value">{prob_display}</div>
            </div>
            <div class="detail-item">
                <div class="detail-item-label">Threshold</div>
                <div class="detail-item-value">{hasil['threshold']:.2f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-item-label">Risk Level</div>
                <div class="detail-item-value" style="color:{'#C0392B' if hasil['risk_level']=='high' else '#E67E22' if hasil['risk_level']=='medium' else '#16A34A'}">
                    {'Tinggi' if hasil['risk_level']=='high' else 'Sedang' if hasil['risk_level']=='medium' else 'Rendah'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4. Keterangan
        st.markdown(f"""
        <div class="keterangan-card">
            <div class="keterangan-title">Keterangan</div>
            <div class="keterangan-text">{hasil['keterangan']}</div>
        </div>
        """, unsafe_allow_html=True)

        # 5. Tombol unduh laporan teks
        laporan = f"""LAPORAN SKRINING ANEMIA — AnemiaDetect
{"="*50}
Nama           : {hasil['nama']}
Umur           : {hasil['umur']} tahun
Jenis Kelamin  : {hasil['gender']}
Kelompok Usia  : {hasil['age_group']}
{"="*50}
HASIL KLASIFIKASI : {hasil['prediksi']}
Probabilitas Anemia : {hasil['probabilitas']*100:.1f}%
Risk Level          : {hasil['risk_level'].upper()}
Threshold           : {hasil['threshold']:.2f}
{"="*50}
Keterangan:
{hasil['keterangan']}
{"="*50}
PERHATIAN: Hasil ini adalah skrining awal berbasis AI
dan tidak menggantikan diagnosis dokter.
Selalu konsultasikan dengan tenaga medis profesional.
"""
        st.download_button(
            label              = '📄  Unduh Laporan (.txt)',
            data               = laporan,
            file_name          = f"laporan_{hasil['nama'].replace(' ','_').lower()}.txt",
            mime               = 'text/plain',
            use_container_width= True,
        )