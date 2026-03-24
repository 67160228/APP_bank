import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ตั้งชือหัวเว็บ
st.title("📊 Bank Deposit Prediction Model")
st.write("นี่คือผลการรันโมเดล Gradient Boosting สำหรับทำนายผล Bank Deposit")

# ใช้ st.cache_data เพื่อไม่ให้โหลดข้อมูลใหม่ทุกครั้งที่ขยับหน้าเว็บ
@st.cache_data
def load_data():
    return pd.read_csv('bank.csv')

# 1. โหลดข้อมูล
df = load_data()

# กำหนด X ด้วยคุณลักษณะที่เลือก และ y เป็นตัวแปรเป้าหมาย 'deposit'
X = df[['duration', 'campaign', 'pdays', 'previous']]
y = df['deposit']

# แปลงคอลัมน์ 'deposit' เป็นค่าตัวเลข (0 สำหรับ 'ไม่', 1 สำหรับ 'ใช่')
y_encoded = y.map({'no': 0, 'yes': 1})

# 2. Scaling ข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 3. สร้างและเทรนโมเดล (ใช้แค่ Gradient Boosting ตามโค้ดเดิมของคุณ)
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. วัดผล
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label=1)
cm = confusion_matrix(y_test, y_pred)

# แสดงค่า Accuracy และ F1 บนหน้าเว็บ Streamlit
st.subheader("Model Performance")
st.write(f"**Accuracy:** {acc:.4f}")
st.write(f"**F1 Score:** {f1:.4f}")

# 5. สร้างรูปกราฟ
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax)

ax.set_ylabel('Actual Class')
ax.set_xlabel('Predicted Class')
plt.tight_layout()

# ใช้ st.pyplot เพื่อส่งกราฟไปแสดงบนหน้าเว็บ Streamlit
st.pyplot(fig)
