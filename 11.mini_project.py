from ultralytics import YOLO
import numpy as np
import cv2
import requests
from PIL import ImageFont, ImageDraw, Image
import math

# 욜로 모델 불러오기
model = YOLO(
    "C:\euntae/06_TensorFlow_File/02_tensorflow_workspace/ch7/Object-Detection/Yolo-Weights/animals_old2.pt"
)

# 사용할 한글 폰트 경로 설정
font_path = "C:\\euntae\\06_TensorFlow_File\\02_tensorflow_workspace\\ch7\\Object-Detection\\fonts\\gulim.ttc"
# 한글 폰트 관련 변수 설정
font_size = 15
font_color = (0, 0, 255)

# 동물의 클래스 매핑
class_mapping = {
    0: "영양 (동물)",
    1: "곰",
    2: "붉은스라소니",
    3: "침팬지",
    4: "하마",
    5: "수달",
    6: "코뿔소",
}


# 위키피디아에서 정보를 가져오는 함수 정의.
# title: 검색하고자 하는 제목 (여기서는 동물의 이름)
# language: 사용할 위키피디아의 언어 (기본값은 한국어 'ko')
def get_wikipedia_info(title, language="ko"):
    # 위키피디아 API의 URL을 생성.
    base_url = "https://{}.wikipedia.org/w/api.php".format(language)

    # 위키피디아 API에 전달할 파라미터들을 딕셔너리로 설정.
    params = {
        "action": "query",  # 데이터를 조회하는 동작을 수행
        "format": "json",  # 결과 데이터 형식을 JSON으로 지정
        "prop": "extracts",  # 페이지의 내용 요약을 가져옴.
        "titles": title,  # 조회하고자하는 페이지(제목)
        "exintro": True,  # 문서의 서두 부분만 추출
        "explaintext": True,  # HTML이 아닌 순수 텍스트로 내용을 가져옴
    }

    # 위 설정 파라미터와 URL을 사용해 요청을 보냄.
    response = requests.get(base_url, params=params)

    # 응답으로 받은 데이터를 JSON 형태로 변환.
    data = response.json()

    # JSON 데이터에서 'query' -> 'pages' 항목에 접근.
    pages = data.get("query", {}).get("pages", {})

    # 'pages' 항목에서 각 페이지에 대한 정보를 순회.
    for page_id, page_info in pages.items():
        # 페이지의 요약된 내용을 반환.
        return page_info.get("extract", "")


# 이미지 파일 경로 설정
image_path = r"C:\euntae\06_TensorFlow_File\02_tensorflow_workspace\ch7\Object-Detection\Animal_Type_Detection-1\test\images\Img-6745_jpg.rf.0af3f4a607308d0c5e9785862674f002.jpg"

# 이미지 읽기
img = cv2.imread(image_path)

# 모델로 이미지 인식
results = model(img)


# 검출된 동물 이름을 저장하기 위한 집합
displayed_animals = set()
# 검출 결과에 대한 반복 처리
for r in results:
    # 결과에서 바운딩 박스 리스트 추출
    boxes = r.boxes
    for box in boxes:
        # 바운딩 박스의 꼭지점 좌표 추출 및 정수 형태로 변환
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # 바운딩 박스의 너비와 높이 계산
        w, h = x2 - x1, y2 - y1
        # 이미지에 바운딩 박스를 파란색으로 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 검출 신뢰도를 구하고, 소수점 둘째 자리에서 반올림
        conf = math.ceil((box.conf[0] * 100)) / 100
        # 바운딩 박스에 대응되는 클래스 ID 를 정수로 변환
        cls = int(box.cls[0])
        # 클래스 ID 를 기반으로 동물 이름 조회 및 "Unknown"으로 대체할 문자열 설정
        class_name = class_mapping.get(cls, "Unknown")
        # 신뢰도가 0.5 이상인 경우에만 처리
        if conf > 0.5:
            # 동물 이름이 이미 표시되지 않은 경우에 대한 처리
            if class_name not in displayed_animals:
                print("검출된 동물:", class_name)
                # 위키피디아로부터 동물에 대한 설명을 가져옴
                description = get_wikipedia_info(class_name)
                # 설명이 있는 경우에 대한 처리
                if description:

                    # PIL 이미지 객체 생성
                    img_pil = Image.fromarray(img)
                    # 이미지에 그리기 위한 객체 생성
                    draw = ImageDraw.Draw(img_pil)
                    # 폰트 설정
                    font = ImageFont.truetype(font_path, font_size)

                    # 텍스트 시작 위치
                    y_text = 0

                    # 설명을 50자 단위로 나누어 리스트 생성
                    lines = [
                        description[i : i + 50] for i in range(0, len(description), 50)
                    ]
                    # 각 줄에 대한 처리
                    for line in lines:
                        # 텍스트 그리기
                        draw.text((10, y_text), line, font=font, fill=font_color)
                        # 다음 줄의 y 위치 조정
                        y_text += 30
                    # PIL 이미지 객체를 다시 NumPy 배열로 변환
                    img = np.array(img_pil)

                # 해당 동물 이름을 표시된 동물 집합에 추가
                displayed_animals.add(class_name)


# 결과 이미지 출력
cv2.imshow("YOLO", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
