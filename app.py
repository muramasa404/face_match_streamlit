import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os

# Streamlit 앱 제목
st.title("👽 나와 닮은꼴 찾기")
st.subheader("⭐️ 나는 박보검일까? 아님 그냥 🍔 밥버거일까?")

# 모델 파일 업로드
uploaded_model = st.file_uploader("모델 파일 업로드", type=["h5"])

if uploaded_model is not None:
    model = load_model(uploaded_model, compile=False)  # 업로드된 모델 로드

    # 라벨 로드
    class_names = open("labels.txt", "r").readlines()

    # 사용자로부터 카메라 입력 받기 (웹캠으로 사진 찍기)
    uploaded_image = st.camera_input("📸 웹캠으로 사진을 찍어주세요!")

    if uploaded_image is not None:
        # "결과 확인" 버튼 추가
        if st.button("🔍 결과 확인"):
            # 이미지 열기
            image = Image.open(uploaded_image).convert("RGB")

            # 이미지 크기 조정 (224x224)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # numpy 배열로 변환 후 정규화
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # 모델 입력 형태로 변환
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # 모델 예측
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index] * 100  # 100% 변환

            # 결과 이미지 경로
            image_path = f"images/{class_name}.jpeg"  # 상대 경로로 수정

            # 예측된 클래스 이미지가 존재하는 경우만 출력
            if os.path.exists(image_path):
                st.subheader("👤 예측 결과")
                st.write(f"**닮은꼴 연예인:** {class_name}")
                st.write(f"**닮은꼴 매칭률:** {confidence_score:.2f}%")  # 퍼센트 표시
                st.image(image_path, caption="🔍 나와 닮은꼴 연예인", use_container_width=True)
            else:
                st.error("⚠️ 해당 클래스에 대한 이미지가 없습니다!")
