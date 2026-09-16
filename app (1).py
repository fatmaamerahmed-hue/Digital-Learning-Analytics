import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# ---------------------------------------------------------
# 1. إعدادات الصفحة (يجب أن تكون في البداية تماماً)
# ---------------------------------------------------------
st.set_page_config(
    page_title="LearnFlow — Student Retention Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# 2. تحميل الموديل والـ Scaler والداتا المدربة
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    # تحميل الموديل من ملف JSON الخاص بـ XGBoost إذا كان موجوداً
    if os.path.exists("model.json"):
        model = xgb.XGBClassifier()
        model.load_model("model.json")
    else:
        model = joblib.load("model.pkl")

    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("digital_learning_analytics_100k.csv", index_col=0)

    # -------------------------------------------------
    # ترميز الأعمدة النصية (Categorical Encoding)
    # -------------------------------------------------
    # لو الـ Scaler اتدرب على أعمدة نصية (زي gender) بعد ترميزها،
    # لازم نعمل نفس الترميز هنا بنفس الطريقة، وإلا هيحصل
    # ValueError: could not convert string to float.
    #
    # ملحوظة مهمة: لو عندك ملف encoders.pkl محفوظ من وقت التدريب
    # (LabelEncoder لكل عمود نصي) لازم تحمّله بدل الحل التلقائي ده،
    # عشان تضمن إن الترميز مطابق 100% لما اتدرب عليه الموديل.
    encoders_path = "encoders.pkl"
    encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
        for col in categorical_cols:
            if col in encoders:
                df[col] = encoders[col].transform(df[col].astype(str))
    else:
        # حل احتياطي: ترميز أبجدي ثابت (LabelEncoder-style) بترتيب القيم
        # مهم: ده افتراض إن الترميز وقت التدريب كان بنفس الترتيب الأبجدي.
        # لو مش متأكد، الأفضل تحفظ الـ encoders الأصلية من كود التدريب.
        for col in categorical_cols:
            categories = sorted(df[col].dropna().unique().tolist())
            mapping = {cat: i for i, cat in enumerate(categories)}
            encoders[col] = mapping
            df[col] = df[col].map(mapping)

    return model, scaler, df, encoders, categorical_cols


model, scaler, df_students, encoders, categorical_cols = load_assets()

# ---------------------------------------------------------
# 3. الهيدر الرئيسي للمشروع
# ---------------------------------------------------------
st.title("🎓 LearnFlow — Student Success & Retention Dashboard")
st.caption("AI-Powered Early Warning System for Student Dropout Risk Prediction")
st.markdown("---")

# ---------------------------------------------------------
# 4. اختيار طالب والعرض التفاعلي
# ---------------------------------------------------------
st.subheader("👤 Real-Time Student Lookup & Sensitivity Simulator")

selected_id = st.selectbox(
    "Select Student ID from active test set:",
    options=df_students.index
)

student_row = df_students.loc[selected_id].copy()
actual_status = student_row["course_completed"]

feature_cols = [col for col in df_students.columns if col != "course_completed"]
baseline_features = student_row[feature_cols].copy()

st.write("")
col_left, col_right = st.columns([1, 1])

# الجزء الأيسر: الـ Sliders السلوكية الأكثر تأثيراً (Top 8 Impact Features)
with col_left:
    st.info(f"📌 **Baseline Status — Student #{selected_id}** | Ground Truth: {'Completed ✅' if actual_status == 1 else 'Not Completed ❌'}")
    st.markdown("#### 🎛️ Top High-Impact Behavioral Sliders")
    st.caption("Adjust student behaviors to simulate real-time model retention sensitivity:")

    top_8_features = [
        'engagement_consistency',
        'total_learning_hours',
        'video_completion_pct',
        'assignment_submission_rate',
        'daily_app_minutes',
        'session_count_weekly',
        'digital_literacy_score',
        'learning_efficiency_score'
    ]

    # نتأكد إننا بنعمل sliders بس للأعمدة الرقمية (مش النصية المُرمّزة)
    slider_features = [f for f in top_8_features if f in feature_cols and f not in categorical_cols]
    slider_vals = {}

    for feat in slider_features:
        orig_val = float(baseline_features[feat])
        min_v = float(df_students[feat].min())
        max_v = float(df_students[feat].max())

        if min_v == max_v:
            max_v += 1.0

        step_v = (max_v - min_v) / 100.0 if (max_v - min_v) != 0 else 0.1

        slider_vals[feat] = st.slider(
            label=f"Modify `{feat}`",
            min_value=min_v,
            max_value=max_v,
            value=orig_val,
            step=step_v
        )

    modified_features = baseline_features.copy()
    for feat, new_val in slider_vals.items():
        modified_features[feat] = new_val

# الجزء الأيمن: النتيجة والـ Prediction اللحظي
with col_right:
    st.markdown("### 🤖 Dynamic Model Prediction & Intervention")

    # 1. تحويل البيانات المعدلة إلى DataFrame لمعالجة الأنواع
    input_df = pd.DataFrame([modified_features])

    # 2. تحويل أي عمود نصي متبقي (احتياطي إضافي لو فيه عمود اتفوت)
    for col in input_df.columns:
        if input_df[col].dtype == object:
            if col in encoders:
                enc = encoders[col]
                if hasattr(enc, "transform"):
                    input_df[col] = enc.transform(input_df[col].astype(str))
                else:
                    input_df[col] = input_df[col].map(enc)
            else:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    # 3. تحديد الأعمدة الرقمية الخاصة بالـ Scaler حصراً
    if hasattr(scaler, "feature_names_in_"):
        input_for_scaler = input_df[scaler.feature_names_in_]
    else:
        input_for_scaler = input_df.select_dtypes(include=[np.number])

    # 4. عمل Scale للمدخلات
    input_scaled = scaler.transform(input_for_scaler)

    # 5. التنبؤ
    prediction = model.predict(input_scaled)[0]

    if hasattr(model, "predict_proba"):
        completion_prob = model.predict_proba(input_scaled)[0][1]
    else:
        completion_prob = float(prediction)

    st.write("")
    if prediction == 1:
        st.success("🎉 **Predicted Status: On Track to Complete**")
        st.metric("Completion Probability", f"{completion_prob * 100:.2f}%")
        st.write("**Action Recommended:** Student is engaged. Automated recognition badge candidate.")
    else:
        st.error("⚠️ **Predicted Status: At-Risk of Dropping Out**")
        st.metric("Dropout Risk Level", f"{(1 - completion_prob) * 100:.2f}%")
        st.write("**Action Recommended:** High intervention priority.")

        if st.button("📩 Trigger Automated Retention Nudge"):
            st.toast("Retention intervention email dispatched!", icon="🚀")
            st.markdown("""
            > **Automated Email Preview:**
            > *"Dear Student, we noticed your activity dropped recently. Check out your personalized dashboard to get back on track!"* 🌟
            """)

st.markdown("---")
st.caption("LearnFlow Engine — Powered by XGBoost Classifier")