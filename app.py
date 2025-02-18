import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os

# 모델 로드
model = load_model("keras_model.h5", compile=False)

# 라벨 로드
class_names = open("labels.txt", "r").readlines()

# 비밀번호 설정 (간단한 비밀번호 예시)
ADMIN_PASSWORD = "1234"

# 세션 상태를 이용한 관리자 인증 유지
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# 사용자 인증을 위한 함수
def authenticate_user():
    password = st.text_input("비밀번호를 입력하세요", type="password")
    if password == ADMIN_PASSWORD:
        st.session_state.is_admin = True
        return True
    elif password:
        st.error("비밀번호가 틀렸습니다.")
    return False

# 사용자로부터 카메라 입력 받기
uploaded_image = st.camera_input("📸 웹캠으로 사진을 찍어주세요!")

# 이미지를 저장할 디렉토리
if not os.path.exists("uploaded_images"):
    os.makedirs("uploaded_images")

# 사용자로부터 이미지가 업로드되면 처리
if uploaded_image is not None:
    # 이미지 파일 저장
    image_path = f"uploaded_images/photo_{len(os.listdir('uploaded_images')) + 1}.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # "결과 보기" 버튼 추가
    if st.button("🔍 결과 보기"):
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

        # 예측된 클래스 이미지가 존재하는 경우만 출력
        image_path = f"image/{class_name}.jpeg"
        if os.path.exists(image_path):
            st.subheader("⭐️ 나와 닮은꼴 연예인은? ⭐️")
            st.write(f"**👉🏻 닮은꼴 연예인:** {class_name}")
            st.write(f"**👉🏻 닮은꼴 신뢰도:** {confidence_score:.2f}%")  # 퍼센트 표시
            st.image(image_path, caption="🔍 나...어쩌면 닮아버린걸까...?", use_container_width=True)

        else:
            st.error("⚠️ 해당 클래스에 대한 이미지가 없습니다!")

# 관리자 로그인 버튼을 사이드바에 배치
with st.sidebar:
    if not st.session_state.is_admin:
        if st.button("관리자 로그인"):
            authenticate_user()

# 관리자만 사진 로그 보기
if st.session_state.is_admin:
    # 관리자만 볼 수 있는 부분
    if st.button("📜 촬영된 사진 로그 보기"):
        # 로그에 저장된 촬영된 이미지 목록 보여주기
        image_files = os.listdir("uploaded_images")
        if image_files:
            st.subheader("촬영된 이미지 목록")
            for image_file in image_files:
                image_path = os.path.join("uploaded_images", image_file)
                st.image(image_path, caption=image_file, use_container_width=True)
        else:
            st.write("촬영된 이미지가 없습니다.")
else:
    st.write("🔒 관리자만 사진 로그를 볼 수 있습니다.")
