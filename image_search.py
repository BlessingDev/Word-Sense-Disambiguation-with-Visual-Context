import os
import requests
from PIL import Image
from bs4 import BeautifulSoup
from google.cloud import vision

def is_truncated(img_path):
    try:
        with Image.open(img_path) as img:
            img.load()  # This forced loading catches the truncation
        return False
    except OSError:
        return True

# API 키 설정 (환경 변수 또는 직접 입력)
# OS_ENV: GOOGLE_APPLICATION_CREDENTIALS (Vision API용 JSON 키 경로)
# genai.configure(api_key="YOUR_GEMINI_API_KEY")

def perform_google_vision_search(image_path):
    """
    Google Cloud Vision API를 사용하여 이미지와 관련된 웹사이트 URL을 추출하는 함수
    
    :param image_path: 분석할 이미지 파일의 경로
    :return: 이미지와 관련된 웹사이트 URL 리스트
    """
    client = vision.ImageAnnotatorClient()
    
    if is_truncated(image_path):
        print(f"Warning: The image at {image_path} is truncated. Skipping this image.")
        return None
    
    # 임시 이미지 파일로 downsized된 이미지를 저장 (API 토큰 제한에 걸리지 않도록)
    img = Image.open(image_path)
    
    if img.size[0] > 3000 or img.size[1] > 3000:
        resized_img = img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)), Image.LANCZOS)

        temp_image_path = "/workspace/temp_resized_image.jpg"
        resized_img.save(temp_image_path, format="JPEG", quality=100)
        image_path = temp_image_path
    
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    try:
        response = client.web_detection(image=image)
    except Exception as e:
        print(f"Error calling Vision API for {image_path}: {e}")
        return None
    annotations = response.web_detection

    # 무엇을 어떤 형태로 저장할 것인가
    # API를 한 번만 불러서 데이터셋을 구성하는 모든 이미지의 검색 정보를 얻어와 저장해두기로 하자
    # 나중에 다시 API 부를 일이 없도록 모든 정보 저장 필요
        
    web_urls = [{"title": page.page_title, "url": page.url} for page in annotations.pages_with_matching_images[:10]] # 상위 10개 추출
    best_label = "No label"
    if annotations.best_guess_labels:
        best_label = annotations.best_guess_labels[0].label
    
    entities = list()
    for entity in annotations.web_entities[:5]: # 상위 5개
        entities.append({
            "description": entity.description,
            "score": entity.score
        })
    
    result_dict = {
        "web_urls": web_urls,
        "best_label": best_label,
        "entities": entities
    }
    
    return result_dict

def get_image_context_pipeline(image_path):
    """
    이미지 경로를 입력받아 최종 컨텍스트 요약을 출력하는 파이프라인
    """
    
    # --- 1단계: Google Cloud Vision API를 통한 웹 검출 ---
    # 입력: image_path (str) - 분석할 이미지 파일 경로
    # 출력: web_urls (list) - 이미지가 포함된 관련 웹 페이지 URL 리스트
    
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    # web_detection 기능을 사용하여 인터넷상에서 유사한 이미지와 페이지를 찾음
    response = client.web_detection(image=image)
    annotations = response.web_detection

    web_urls = [page.url for page in annotations.pages_with_matching_images[:3]] # 상위 3개 추출
    print(f"[Step 1] 관련 사이트 추출 완료: {len(web_urls)}개 발견")


    # --- 2단계: 웹 사이트 정보 크롤링 및 요약 데이터 준비 ---
    # 입력: web_urls (list) - 웹 페이지 URL 리스트
    # 출력: combined_text (str) - 각 사이트에서 추출한 텍스트 데이터의 합계
    
    collected_data = []
    
    for url in web_urls:
        try:
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # 제목과 본문 텍스트 일부 추출 (너무 길면 API 토큰 제한에 걸릴 수 있음)
            title = soup.title.string if soup.title else "No Title"
            paragraphs = soup.find_all('p')
            content_snippet = " ".join([p.get_text() for p in paragraphs[:5]]) 
            
            collected_data.append(f"Site: {title}\nContent: {content_snippet}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    combined_text = "\n\n".join(collected_data)
    print(f"[Step 2] 사이트 정보 수집 완료 (텍스트 길이: {len(combined_text)})")


    # --- 3단계: Gemini API를 이용한 컨텍스트 요약 ---
    # 입력: combined_text (str) - 수집된 웹 정보 텍스트
    # 출력: summary (str) - 이미지의 최종 컨텍스트 분석 결과
    
    # 이 부분은 local llm으로 수행
    
    
    return summary

# 실행 예시
# result = get_image_context_pipeline("my_image.jpg")
# print("\n=== 이미지 분석 결과 ===\n")
# print(result)