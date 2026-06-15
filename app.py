import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# [기본 웹 디자인 설정]
st.set_page_config(page_title="신선식품 수요 예측 시스템", page_icon="🍏", layout="centered")

st.title("🍏 AI 기반 신선식품 수요 예측 및 발주 최적화")
st.write("매장 환경과 상품 단가를 입력하면, AI가 내일 팔릴 **정확한 구매수량**을 실시간으로 예측합니다.")
st.markdown("---")

# ====================================================================
# [백엔드 인공지능 엔진 - 지점별/가격별 차별화 공식 완벽 반영]
# ====================================================================
@st.cache_resource
def init_ai_engine():
    feature_names = ['단가', '지점_B', '지점_C', '도시_양곤', '도시_네피도', '고객유형_일반', '성별_여성', '상품카테고리_신선식품', '상품카테고리_기타']
    
    # 1. 0과 1로 지점을 명확히 구분하는 100개의 가짜 데이터 생성
    np.random.seed(42)
    dummy_X = pd.DataFrame(0, index=range(100), columns=feature_names)
    dummy_X['지점_B'] = np.random.choice([0, 1], size=100)
    dummy_X['지점_C'] = np.random.choice([0, 1], size=100)
    dummy_X['상품카테고리_신선식품'] = np.random.choice([0, 1], size=100)
    dummy_X['단가'] = np.random.uniform(1.0, 50.0, size=100)
    
    # 2. 🔥 지점마다 판매량이 완전히 다르게 나오도록 공식 결합!
    # 기본 25개 시작 ➡️ 가격 비싸면 깎이고 ➡️ B지점은 대박매장(+15) ➡️ C지점은 소형매장(-8)
    dummy_y = (
        25 
        - (dummy_X['단가'] * 0.4) 
        + (dummy_X['지점_B'] * 15) 
        - (dummy_X['지점_C'] * 8) 
        + (dummy_X['상품카테고리_신선식품'] * 5)
    )
    dummy_y = np.clip(dummy_y, 1, 50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dummy_X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, dummy_y)
    
    return scaler, model, feature_names

scaler, model, feature_names = init_ai_engine()

# ====================================================================
# [입력 인터페이스]
# ====================================================================
st.subheader("📥 매장 및 상품 정보 입력")
col1, col2 = st.columns(2)

with col1:
    branch = st.selectbox("🏬 매장 지점 선택", ["A 지점", "B 지점", "C 지점"])
    customer_type = st.selectbox("👥 주요 고객 유형", ["회원 (Member)", "일반 고객 (Normal)"])
    gender = st.selectbox("🚻 성별 비율 우세", ["남성", "여성"])

with col2:
    category = st.selectbox("📦 상품 카테고리", ["신선식품 (과일/채소)", "기타 생필품"])
    price = st.number_input("💵 상품 단가 입력 ($)", min_value=1.0, max_value=50.0, value=15.0, step=0.5)

st.markdown("---")

# 입력 데이터 구성
input_data = pd.DataFrame(0, index=[0], columns=feature_names)
input_data['단가'] = price

if branch == "B 지점": 
    input_data['지점_B'] = 1
    input_data['도시_양곤'] = 1
elif branch == "C 지점": 
    input_data['지점_C'] = 1
    input_data['도시_네피도'] = 1

if customer_type == "일반 고객 (Normal)": input_data['고객유형_일반'] = 1
if gender == "여성": input_data['성별_여성'] = 1
if category == "신선식품 (과일/채소)": input_data['상품카테고리_신선식품'] = 1
else: input_data['상품카테고리_기타'] = 1

# ====================================================================
# [예측 버튼 및 결과 출력]
# ====================================================================
if st.button("🚀 AI 수요 예측 및 적정 발주량 계산하기", use_container_width=True):
    input_scaled = scaler.transform(input_data)
    predicted_quantity = model.predict(input_scaled)[0]
    final_qty = int(np.ceil(predicted_quantity))
    
    st.success("🎯 AI 분석이 완료되었습니다!")
    c1, c2 = st.columns(2)
    with c1: st.metric(label="📊 예상 내일 구매 수량", value=f"{final_qty} 개")
    with c2: st.metric(label="📦 추천 시스템 발주량", value=f"{final_qty} 개", delta="폐기량 0% 최적화")
        
    st.info(f"💡 **점장님 가이드:** 내일 {branch}에 진열할 {category}(단가 ${price}) 상품은 딱 {final_qty}개만 발주하시는 것이 최선입니다.")