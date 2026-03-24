import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# 1. ตั้งค่าหัวหน้าเว็บ
st.set_page_config(page_title="Bank Deposit Predictor", page_icon="🏦")
st.title("🏦 Bank Deposit Prediction App")
st.write("แอปพลิเคชันสำหรับทำนายว่าลูกค้าจะทำการฝากเงินหรือไม่ (Deposit: Yes / No)")

# 2. ฟังก์ชันสำหรับโหลดข้อมูลและเทรนโมเดล (ใช้ Cache เพื่อไม่ให้เทรนใหม่ทุกครั้งที่ผู้ใช้ขยับเมาส์)
@st.cache_resource
def load_and_train_model():
    # โหลดข้อมูล
    df = pd.read_csv('bank.csv')
    
    # กำหนด Features และ Target
    X = df[['duration', 'campaign', 'pdays', 'previous']]
    y = df['deposit'].map({'no': 0, 'yes': 1})
    
    # Scale ข้อมูล
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # สร้างและเทรนโมเดล (ใช้ข้อมูลทั้งหมดเทรนเพื่อไปใช้จริง)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# เรียกใช้งานฟังก์ชันเพื่อเตรียมโมเดล
model, scaler = load_and_train_model()

# 3. สร้าง Sidebar สำหรับรับข้อมูลจากผู้ใช้งาน
st.sidebar.header("📝 กรอกข้อมูลลูกค้า")
duration = st.sidebar.number_input("Duration (ระยะเวลาการคุย - วินาที)", min_value=0, value=100)
campaign = st.sidebar.number_input("Campaign (จำนวนครั้งที่ติดต่อ)", min_value=1, value=1)
pdays = st.sidebar.number_input("Pdays (วันหลังติดต่อครั้งล่าสุด, -1 คือไม่เคย)", min_value=-1, value=-1)
previous = st.sidebar.number_input("Previous (จำนวนครั้งที่ติดต่อก่อนหน้านี้)", min_value=0, value=0)

# 4. ปุ่มกดสำหรับทำนายผล
if st.button("🔍 ทำนายผล (Predict)"):
    # นำข้อมูลที่ผู้ใช้กรอกมาจัดรูปแบบเป็น DataFrame
    input_data = pd.DataFrame({
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous]
    })
    
    # Scale ข้อมูล Input ให้สเกลเดียวกับตอนเทรน
    input_scaled = scaler.transform(input_data)
    
    # ให้โมเดลทำนายผล
    prediction = model.predict(input_scaled)
    
    # 5. แสดงผลลัพธ์
    st.subheader("ผลการทำนาย:")
    if prediction[0] == 1:
        st.success("✅ มีแนวโน้มที่ลูกค้าจะ **ฝากเงิน (Yes)**")
    else:
        st.error("❌ มีแนวโน้มที่ลูกค้าจะ **ไม่ฝากเงิน (No)**")
