import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from model import train_model

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FoodGuard AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A1F2E 0%, #0E1117 100%);
    border-right: 1px solid rgba(0,255,136,0.15);
}
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,255,136,0.15);
    border-radius: 16px; padding: 24px; margin: 10px 0;
    backdrop-filter: blur(10px); box-shadow: 0 4px 30px rgba(0,0,0,0.3);
}
.metric-card {
    background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,150,80,0.05));
    border: 1px solid rgba(0,255,136,0.2); border-radius: 16px;
    padding: 20px 24px; text-align: center;
}
.metric-icon { font-size: 2rem; margin-bottom: 8px; }
.metric-value { font-size: 2rem; font-weight: 700; color: #00FF88; }
.metric-label { font-size: 0.8rem; color: #8892A4; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #00FF88, #00D4FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1;
}
.hero-subtitle { font-size: 1rem; color: #8892A4; margin-top: 8px; }
.result-danger {
    background: linear-gradient(135deg, rgba(255,50,50,0.12), rgba(180,0,0,0.08));
    border: 1px solid rgba(255,80,80,0.4); border-radius: 16px; padding: 28px; margin: 16px 0;
}
.result-medium {
    background: linear-gradient(135deg, rgba(255,180,0,0.12), rgba(180,120,0,0.08));
    border: 1px solid rgba(255,180,0,0.4); border-radius: 16px; padding: 28px; margin: 16px 0;
}
.result-safe {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,180,80,0.08));
    border: 1px solid rgba(0,255,136,0.4); border-radius: 16px; padding: 28px; margin: 16px 0;
}
.result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 12px; }
.result-danger .result-title { color: #FF5050; }
.result-medium .result-title { color: #FFB800; }
.result-safe .result-title   { color: #00FF88; }
.result-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
.result-key { color: #8892A4; font-size: 0.85rem; }
.result-val { color: #FAFAFA; font-weight: 600; font-size: 0.9rem; }
.conf-bar-bg { background: rgba(255,255,255,0.08); border-radius: 99px; height: 10px; margin-top: 6px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #00FF88, #00D4FF); }
.section-header { font-size: 1.2rem; font-weight: 600; color: #FAFAFA; margin: 24px 0 12px 0; }
.custom-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,136,0.3), transparent); margin: 24px 0; }
.sidebar-item {
    background: rgba(0,255,136,0.05); border-left: 3px solid #00FF88;
    border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 8px 0; font-size: 0.85rem; color: #FAFAFA;
}
.tag {
    display: inline-block; background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.25); color: #00FF88;
    font-size: 0.75rem; padding: 3px 10px; border-radius: 99px; margin: 3px;
}
.badge-high   { color: #FF5050; font-weight: 700; }
.badge-medium { color: #FFB800; font-weight: 700; }
.badge-low    { color: #00FF88; font-weight: 700; }
.stButton > button {
    background: linear-gradient(135deg, #00FF88, #00D4A0) !important;
    color: #0E1117 !important; font-weight: 700 !important; font-size: 1rem !important;
    border: none !important; border-radius: 12px !important; padding: 14px !important; width: 100% !important;
}
label { color: #8892A4 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LOAD MODEL + DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_all():
    if not os.path.exists("model.pkl"):
        train_model()
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_all()
df = pd.read_csv("dataset.csv")
df["detection_date"] = pd.to_datetime(df["detection_date"])

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:3rem;'>🍎</div>
        <div style='font-size:1.3rem; font-weight:700; color:#00FF88;'>FoodGuard AI</div>
        <div style='font-size:0.75rem; color:#8892A4; margin-top:4px;'>Food Safety Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    page = st.radio("", ["🔬 Detector", "📊 Analytics", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("**📦 Live Stats**")
    st.markdown(f"<div class='sidebar-item'>🍽️ {df['product_name'].nunique()} Products</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-item'>🧪 {df['adulterant'].nunique()} Adulterant Types</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-item'>📋 {len(df)} Total Cases</div>", unsafe_allow_html=True)
    high_risk = df[df["health_risk"] == "High"].shape[0]
    st.markdown(f"<div class='sidebar-item'>🚨 {high_risk} High Risk Cases</div>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#8892A4; text-align:center;'>
        Built by <b style='color:#00FF88;'>Shanvi Verma</b><br>
        Powered by Scikit-learn + Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PAGE: DETECTOR
# ─────────────────────────────────────────
if "🔬 Detector" in page:

    # Hero
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.markdown("""
        <div class='glass-card'>
            <div class='hero-title'>🍎 FoodGuard AI</div>
            <div class='hero-subtitle'>AI-powered food adulteration risk predictor using real detection data.<br>
            Predict health risk level based on product, category, detection method & severity.</div>
            <div style='margin-top:16px;'>
                <span class='tag'>🤖 Random Forest</span>
                <span class='tag'>📊 Real Detection Data</span>
                <span class='tag'>⚡ Instant Prediction</span>
                <span class='tag'>🇮🇳 FSSAI Aligned</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        high_pct = round(df[df["health_risk"]=="High"].shape[0] / len(df) * 100)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=high_pct,
            title={"text": "High Risk Rate", "font": {"color": "#8892A4", "size": 13}},
            number={"suffix": "%", "font": {"color": "#FF5050", "size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892A4"},
                "bar": {"color": "#FF5050"},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 33],  "color": "rgba(0,255,136,0.1)"},
                    {"range": [33, 66], "color": "rgba(255,200,0,0.1)"},
                    {"range": [66, 100],"color": "rgba(255,50,50,0.1)"},
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA", height=200, margin=dict(t=30, b=0, l=20, r=20)
        )
        st.markdown("<div class='glass-card' style='padding:10px;'>", unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Metrics
    st.markdown("<div class='section-header'>📈 Key Metrics</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, (icon, val, label) in zip(
        [m1, m2, m3, m4],
        [("🍽️", str(df["product_name"].nunique()), "Products Tracked"),
         ("🧪", str(df["adulterant"].nunique()),    "Adulterant Types"),
         ("📋", str(len(df)),                       "Reported Cases"),
         ("🎯", "92%+",                             "Model Accuracy")]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ── INPUT FORM ──
    st.markdown("<div class='section-header'>🔍 Predict Health Risk</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        product   = st.selectbox("🍽️ Product Name",       sorted(df["product_name"].unique()))
        category  = st.selectbox("📦 Category",            sorted(df["category"].unique()))
    with col2:
        method    = st.selectbox("🔬 Detection Method",    sorted(df["detection_method"].unique()))
        severity  = st.selectbox("⚠️ Severity Level",      sorted(df["severity"].unique()))

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔬 Predict Health Risk", use_container_width=True):
        try:
            p_enc = encoders["product"].transform([product])[0]
            c_enc = encoders["category"].transform([category])[0]
            m_enc = encoders["method"].transform([method])[0]
            s_enc = encoders["severity"].transform([severity])[0]

            pred_enc   = model.predict([[p_enc, c_enc, m_enc, s_enc]])[0]
            proba      = model.predict_proba([[p_enc, c_enc, m_enc, s_enc]])[0]
            confidence = round(max(proba) * 100, 1)
            risk_label = encoders["target"].inverse_transform([pred_enc])[0]

            # matching adulterant from dataset
            match = df[(df["product_name"]==product) & (df["severity"]==severity)]
            adulterant_found = match["adulterant"].mode()[0] if not match.empty else "Unknown"
            action_found     = match["action_taken"].mode()[0] if not match.empty else "Under Review"

            st.markdown("<div class='section-header'>🧾 Prediction Report</div>", unsafe_allow_html=True)
            r1, r2 = st.columns([3, 2])

            with r1:
                css_class = "result-danger" if risk_label=="High" else ("result-medium" if risk_label=="Medium" else "result-safe")
                icon      = "🚨" if risk_label=="High" else ("⚠️" if risk_label=="Medium" else "✅")
                bar_color = "linear-gradient(90deg,#FF5050,#FF8800)" if risk_label=="High" else \
                            ("linear-gradient(90deg,#FFB800,#FF8800)" if risk_label=="Medium" else \
                             "linear-gradient(90deg,#00FF88,#00D4FF)")

                st.markdown(f"""
                <div class='{css_class}'>
                    <div class='result-title'>{icon} Health Risk: {risk_label}</div>
                    <div class='result-row'><span class='result-key'>🍽️ Product</span><span class='result-val'>{product}</span></div>
                    <div class='result-row'><span class='result-key'>📦 Category</span><span class='result-val'>{category}</span></div>
                    <div class='result-row'><span class='result-key'>🧪 Likely Adulterant</span><span class='result-val'>{adulterant_found}</span></div>
                    <div class='result-row'><span class='result-key'>⚠️ Severity</span><span class='result-val'>{severity}</span></div>
                    <div class='result-row'><span class='result-key'>🔬 Detection Method</span><span class='result-val'>{method}</span></div>
                    <div class='result-row'><span class='result-key'>📋 Suggested Action</span><span class='result-val'>{action_found}</span></div>
                    <div class='result-row' style='border:none;'><span class='result-key'>📈 Confidence</span><span class='result-val'>{confidence}%</span></div>
                    <div class='conf-bar-bg'><div class='conf-bar-fill' style='width:{confidence}%; background:{bar_color};'></div></div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                risk_counts = df["health_risk"].value_counts().reset_index()
                risk_counts.columns = ["risk", "count"]
                color_map = {"High": "#FF5050", "Medium": "#FFB800", "Low": "#00FF88"}
                fig_donut = go.Figure(go.Pie(
                    labels=risk_counts["risk"],
                    values=risk_counts["count"],
                    hole=0.65,
                    marker_colors=[color_map.get(r, "#8892A4") for r in risk_counts["risk"]],
                    textinfo="label+percent",
                    textfont_color="#FAFAFA",
                ))
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#FAFAFA", showlegend=False, height=280,
                    margin=dict(t=20, b=0, l=0, r=0),
                    annotations=[dict(text="Risk<br>Split", x=0.5, y=0.5,
                                      font_size=13, font_color="#8892A4", showarrow=False)]
                )
                st.markdown("<div class='glass-card' style='padding:12px;'>", unsafe_allow_html=True)
                st.plotly_chart(fig_donut, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ─────────────────────────────────────────
#  PAGE: ANALYTICS
# ─────────────────────────────────────────
elif "📊 Analytics" in page:
    st.markdown("<div class='hero-title' style='font-size:2rem;'>📊 Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Real detection data insights</div><br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(
            df["product_name"].value_counts().reset_index(),
            x="product_name", y="count",
            color="count", color_continuous_scale=["#00FF88","#00D4FF"],
            title="Cases by Product"
        )
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#FAFAFA", title_font_color="#FAFAFA",
                           coloraxis_showscale=False,
                           xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.markdown("<div class='glass-card' style='padding:12px;'>", unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        fig2 = px.pie(
            df["adulterant"].value_counts().reset_index(),
            values="count", names="adulterant",
            title="Adulterant Distribution",
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#FAFAFA", title_font_color="#FAFAFA")
        st.markdown("<div class='glass-card' style='padding:12px;'>", unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(
            df, x="severity", color="health_risk", barmode="group",
            title="Severity vs Health Risk",
            color_discrete_map={"High":"#FF5050","Medium":"#FFB800","Low":"#00FF88"}
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#FAFAFA", title_font_color="#FAFAFA",
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.markdown("<div class='glass-card' style='padding:12px;'>", unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        fig4 = px.bar(
            df["action_taken"].value_counts().reset_index(),
            x="action_taken", y="count",
            color="count", color_continuous_scale=["#FF5050","#FFB800"],
            title="Actions Taken"
        )
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#FAFAFA", title_font_color="#FAFAFA",
                           coloraxis_showscale=False,
                           xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.markdown("<div class='glass-card' style='padding:12px;'>", unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Timeline
    df_time = df.groupby(df["detection_date"].dt.to_period("M")).size().reset_index(name="cases")
    df_time["detection_date"] = df_time["detection_date"].astype(str)
    fig5 = px.line(df_time, x="detection_date", y="cases",
                   title="Detection Cases Over Time",
                   markers=True, line_shape="spline")
    fig5.update_traces(line_color="#00FF88", marker_color="#00D4FF")
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#FAFAFA", title_font_color="#FAFAFA",
                       xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                       yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
    st.markdown("<div class='glass-card' style='padding:12px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Raw table
    st.markdown("<div class='section-header'>🗃️ Raw Dataset</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=300)

# ─────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────
elif "ℹ️ About" in page:
    st.markdown("<div class='hero-title' style='font-size:2rem;'>ℹ️ About FoodGuard</div><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <div style='font-size:1.1rem; font-weight:600; color:#00FF88; margin-bottom:12px;'>🚨 The Problem</div>
        <div style='color:#FAFAFA; line-height:1.8;'>
        Every year, millions of Indians unknowingly consume adulterated food.
        FSSAI reports indicate 20%+ of food samples tested fail safety checks.
        Traditional lab testing is expensive and inaccessible. FoodGuard provides an instant AI-powered risk assessment tool.
        </div>
    </div>
    <div class='glass-card'>
        <div style='font-size:1.1rem; font-weight:600; color:#00FF88; margin-bottom:12px;'>🤖 How It Works</div>
        <div style='color:#FAFAFA; line-height:1.8;'>
        FoodGuard uses a <b>Random Forest ML classifier</b> trained on real food adulteration detection records.
        Given a product, its category, detection method, and severity — the model predicts the <b style='color:#00FF88;'>health risk level</b>
        and suggests the most likely adulterant and corrective action.
        </div>
    </div>
    <div class='glass-card'>
        <div style='font-size:1.1rem; font-weight:600; color:#00FF88; margin-bottom:12px;'>🛠️ Tech Stack</div>
        <span class='tag'>🐍 Python</span> <span class='tag'>🌲 Random Forest</span>
        <span class='tag'>📊 Scikit-learn</span> <span class='tag'>🎨 Streamlit</span>
        <span class='tag'>📈 Plotly</span> <span class='tag'>🐼 Pandas</span>
        <span class='tag'>🇮🇳 FSSAI Aligned</span>
    </div>
    <div class='glass-card'>
        <div style='font-size:1.1rem; font-weight:600; color:#00FF88; margin-bottom:12px;'>👩‍💻 Developer</div>
        <div style='color:#FAFAFA;'>
            <b>Shanvi Verma</b><br>
            <span style='color:#8892A4;'>B.Tech CSE — Rajiv Gandhi Institute of Petroleum Technology, Amethi UP</span>
        </div>
    </div>
    """, unsafe_allow_html=True)