import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from fpdf import FPDF


st.set_page_config(page_title="Chronic Care Risk Dashboard", layout="wide")



DATA_PATH = 'dashboard_data.json'

# ---------- Helpers ----------

def load_dashboard_json(path=DATA_PATH):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def synthesize_cohort(data):
    """Create a synthetic cohort matching the cohortStats and using samplePatients as templates."""
    stats = data.get('cohortStats', {})
    total = stats.get('totalPatients', 150)
    high = stats.get('highRisk', 23)
    med = stats.get('mediumRisk', 67)
    low = stats.get('lowRisk', 60)

    templates = {p['riskLevel']: p for p in data.get('samplePatients', [])}

    rows = []
    rng = np.random.default_rng(seed=42)

    def gen_patient(idx, level):
        t = templates.get(level, None)
        if t:
            base = t.copy()
            name = f"Patient {idx:03d}"
            age = max(18, int(t['age'] + rng.integers(-5, 6)))
            score = int(np.clip(t['riskScore'] + rng.integers(-10, 11), 0, 100))
            last_visit = t.get('lastVisit', '2024-01-01')
            return {
                'id': idx,
                'name': name,
                'age': age,
                'gender': t.get('gender', 'F'),
                'condition': t.get('condition', 'Chronic Condition'),
                'riskScore': score,
                'riskLevel': level,
                'lastVisit': last_visit,
                'nextActions': t.get('nextActions', []),
                'keyFindings': t.get('keyFindings', [])
            }
        else:
            # fallback synthetic row
            lvl_map = {'High': (70, 90), 'Medium': (40, 69), 'Low': (0, 39)}
            rng_score = rng.integers(*lvl_map.get(level, (30, 70)))
            return {
                'id': idx,
                'name': f'Patient {idx:03d}',
                'age': int(rng.integers(30, 85)),
                'gender': rng.choice(['M', 'F']),
                'condition': 'Type 2 Diabetes',
                'riskScore': int(rng_score),
                'riskLevel': level,
                'lastVisit': '2024-01-01',
                'nextActions': [],
                'keyFindings': []
            }

    idx = 1
    for _ in range(high):
        rows.append(gen_patient(idx, 'High')); idx += 1
    for _ in range(med):
        rows.append(gen_patient(idx, 'Medium')); idx += 1
    for _ in range(low):
        rows.append(gen_patient(idx, 'Low')); idx += 1

    # If rounding issues cause mismatch, pad with low-risk
    while len(rows) < total:
        rows.append(gen_patient(idx, 'Low')); idx += 1

    df = pd.DataFrame(rows)
    return df


def color_for_risk(row):
    if row == 'High':
        return '🔴 High'
    if row == 'Medium':
        return '🟠 Medium'
    return '🟢 Low'


def build_shap_like_contributions(patient_row, top_factors):
    """Generate a patient-level contribution bar values using top risk factors and patient's risk score."""
    score = patient_row.get('riskScore', 50)
    base_importances = np.array([f['importance'] for f in top_factors])
    # normalize and scale by (score/100)
    if base_importances.sum() == 0:
        scaled = np.ones_like(base_importances) * (score / 100)
    else:
        scaled = (base_importances / base_importances.sum()) * (score / 100)
    # produce positive/negative direction heuristically using rank
    contributions = scaled * 100
    labels = [f['clinical_name'] for f in top_factors]
    return labels, contributions


def plot_risk_timeline(patient):
    # Simple synthetic timeline of riskScore across past 6 points
    x = pd.date_range(end=pd.to_datetime(patient.get('lastVisit', '2024-01-01')), periods=6)
    rng = np.random.default_rng(seed=patient['id'])
    base = patient['riskScore']
    y = np.clip(base + rng.integers(-8, 9, size=6).cumsum()/3, 0, 100)
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_ylim(0, 100)
    ax.set_title('Risk Score Trend (synthetic)')
    ax.set_ylabel('Risk Score')
    ax.set_xlabel('Date')
    fig.autofmt_xdate()
    return fig


def plot_shap_bar(labels, values):
    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Contribution (approx %)')
    ax.set_title('Top Feature Contributions (approx)')
    plt.tight_layout()
    return fig


def download_json_button(data, filename='dashboard_export.json'):
    b = json.dumps(data, indent=2).encode()
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">📥 Download the JSON for structured data</a>'
    st.markdown(href, unsafe_allow_html=True)


def generate_pdf_report(patient, data, filename='patient_report.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Patient Report - {patient['name']}", ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"Age: {patient['age']}    Gender: {patient['gender']}    Last Visit: {patient['lastVisit']}", ln=1)
    pdf.cell(0, 8, f"Risk Score: {patient['riskScore']}    Risk Level: {patient['riskLevel']}", ln=1)
    pdf.ln(4)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Key Findings:', ln=1)
    pdf.set_font('Arial', '', 11)
    for k in patient.get('keyFindings', []):
        pdf.multi_cell(0, 7, f"- {k}")
    pdf.ln(2)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Recommended Next Actions:', ln=1)
    pdf.set_font('Arial', '', 11)
    for act in patient.get('nextActions', []):
        pdf.multi_cell(0, 7, f"- ({act.get('priority')}) {act.get('action')}  -- {act.get('reason')}")

    # Save to buffer
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


# ---------- App ----------

data = load_dashboard_json()

st.title('🏥 Chronic Care Risk Prediction Dashboard')

# Sidebar: model summary
with st.sidebar:
    st.header('Model Summary')
    perf = data.get('modelPerformance', {})
    st.write(f"**Best model:** {perf.get('best_model', 'N/A')}")
    st.write(f"AUROC: {perf.get('auroc', 'N/A')}")
    st.write(f"AUPRC: {perf.get('auprc', 'N/A')}")
    st.write(f"Sensitivity: {perf.get('sensitivity', 'N/A')}")
    st.write(f"Specificity: {perf.get('specificity', 'N/A')}")
    st.write(f"Brier score: {perf.get('brier_score', 'N/A')}")
    st.markdown('---')
    st.header('Cohort')
    stats = data.get('cohortStats', {})
    st.write(f"Total patients: {stats.get('totalPatients', 'N/A')}")
    st.write(f"High: {stats.get('highRisk', 0)} | Medium: {stats.get('mediumRisk', 0)} | Low: {stats.get('lowRisk', 0)}")


# Build cohort dataframe (synthetic but matches counts)
cohort_df = synthesize_cohort(data)

# Main layout
tab1, tab2, tab3 = st.tabs(['Cohort View', 'Patient Detail', 'Model Performance'])

with tab1:
    st.header('Cohort View')
    c1, c2 = st.columns([3, 1])
    with c2:
        risk_filter = st.multiselect('Filter by risk level', options=['High', 'Medium', 'Low'], default=['High','Medium','Low'])
        min_age, max_age = st.slider('Age range', 18, 100, (18, 90))
        search_name = st.text_input('Search name (contains)')

    # apply filters
    df_view = cohort_df[cohort_df['riskLevel'].isin(risk_filter)]
    df_view = df_view[(df_view['age'] >= min_age) & (df_view['age'] <= max_age)]
    if search_name:
        df_view = df_view[df_view['name'].str.contains(search_name, case=False)]

    df_display = df_view[['id','name','age','gender','condition','riskScore','riskLevel','lastVisit']].copy()
    df_display['riskTag'] = df_display['riskLevel'].apply(color_for_risk)

    st.dataframe(df_display.reset_index(drop=True), use_container_width=True)

    st.markdown('### Cohort Distribution')
    dist = df_display['riskLevel'].value_counts().reindex(['High','Medium','Low']).fillna(0)
    fig, ax = plt.subplots()
    ax.pie(dist, labels=dist.index, autopct='%1.1f%%')
    ax.set_title('Risk Level Distribution (filtered cohort)')
    st.pyplot(fig)

    st.markdown('---')
    st.write('Download the current cohort JSON:')
    download_json_button(df_view.to_dict(orient='records'), filename='cohort_view.json')

with tab2:
    st.header('Patient Detail')
    sample_ids = cohort_df['id'].tolist()[:150]
    selected = st.selectbox('Select patient', options=sample_ids, format_func=lambda x: f"{x} - {cohort_df.loc[cohort_df['id']==x,'name'].values[0]}")
    patient = cohort_df[cohort_df['id']==selected].iloc[0].to_dict()

    p1, p2 = st.columns([2,1])
    with p1:
        st.subheader(f"{patient['name']} — {patient['riskLevel']} risk ({patient['riskScore']})")
        st.write(f"Age: {patient['age']}  |  Gender: {patient['gender']}  |  Last visit: {patient['lastVisit']}")

        st.markdown('**Key Findings**')
        findings = patient.get('keyFindings', [])
        if findings:
            for f in findings:
                st.write(f"- {f}")
        else:
            st.write('No structured findings available for this synthetic patient.')

        st.markdown('**Recommended Next Actions**')
        acts = patient.get('nextActions', [])
        if acts:
            for a in acts:
                st.write(f"- ({a.get('priority')}) {a.get('action')} — {a.get('reason')}")
        else:
            st.write('No recommended actions available')

    with p2:
        st.markdown('### Risk Timeline (synthetic)')
        fig_t = plot_risk_timeline(patient)
        st.pyplot(fig_t)

    st.markdown('---')
    st.markdown('### Explainability (Top Indicators)')
    top_factors = data.get('topRiskFactors', [])
    labels, values = build_shap_like_contributions(patient, top_factors[:10])
    fig_s = plot_shap_bar(labels, values)
    st.pyplot(fig_s)

    st.markdown('---')
    if st.button('Generate PDF report for this patient'):
        buf = generate_pdf_report(patient, data)
        b64 = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{patient["name"].replace(" ","_")}_report.pdf">📥 Download PDF report</a>'
        st.markdown(href, unsafe_allow_html=True)

with tab3:
    st.header('Model Performance')
    perf = data.get('modelPerformance', {})
    st.metric('Best Model', perf.get('best_model', 'N/A'))
    st.write('**Discrimination metrics**')
    cols = st.columns(3)
    cols[0].metric('AUROC', perf.get('auroc', 'N/A'))
    cols[1].metric('AUPRC', perf.get('auprc', 'N/A'))
    cols[2].metric('Brier Score', perf.get('brier_score', 'N/A'))

    st.markdown('**Confusion Matrix (from JSON)**')
    cm = np.array(data.get('confusionMatrix', [[0,0],[0,0]]))
    df_cm = pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
    st.table(df_cm)

    st.markdown('**Top Risk Factors (Population-level)**')
    tr = pd.DataFrame(data.get('topRiskFactors', []))
    if not tr.empty:
        st.dataframe(tr[['rank','clinical_name','feature','importance']].set_index('rank'))
    else:
        st.write('No top factors in data')

    st.markdown('**Clinical Recommendations**')
    for rec in data.get('clinicalRecommendations', []):
        st.write(f'- {rec}')
