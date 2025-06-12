"""
매트리스 AI Agent 설정 파일
모든 설정값을 한 곳에서 관리
"""

import os

# ========================
# 프로젝트 기본 설정
# ========================
PROJECT_NAME = "매트리스 구매 가이드 AI Agent"
VERSION = "1.0.0"
AUTHOR = "TEAM_10"

# ========================
# 데이터 경로 설정
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MATTRESS_DATA_PATH = os.path.join(DATA_DIR, "mattress_data.json")

# ========================
# AI 모델 설정
# ========================
# 임베딩 모델
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# LLM 모델 (Hugging Face)
LLM_MODEL = "microsoft/DialoGPT-medium"
MAX_LENGTH = 512
TEMPERATURE = 0.7

# ========================
# ChromaDB 설정
# ========================
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "mattress_collection"
SIMILARITY_THRESHOLD = 0.7

# ========================
# AI Agent 설정
# ========================
MAX_RECOMMENDATIONS = 3
MIN_RECOMMENDATIONS = 1
SEARCH_RESULTS_COUNT = 8

# Function Calling 가중치
SCORING_WEIGHTS = {
    "budget": 0.30,        # 예산 적합성 30%
    "sleep_position": 0.25, # 수면자세 25%
    "health": 0.25,        # 건강 문제 25%
    "temperature": 0.20    # 온도 선호 20%
}

# ========================
# 웹 앱 설정 (Streamlit)
# ========================
STREAMLIT_CONFIG = {
    "port": 8501,
    "host": "localhost",
    "page_title": "매트리스 구매 가이드 AI Agent",
    "page_icon": "🛏️",
    "layout": "wide"
}

# ========================
# 로깅 설정
# ========================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(BASE_DIR, "ai_agent.log")

# ========================
# 개발/테스트 설정
# ========================
DEBUG_MODE = True
TEST_DATA_SIZE = 5  # 테스트 시 사용할 데이터 개수
ENABLE_CACHE = True

# ========================
# 프롬프트 설정
# ========================
SYSTEM_PROMPTS = {
    "expert_role": "당신은 10년 경력의 매트리스 전문가입니다.",
    "analysis_steps": [
        "1. 고객 요구사항 분석",
        "2. 조건별 매트리스 필터링", 
        "3. 적합도 점수 계산",
        "4. 상위 3개 추천",
        "5. 상세 설명 제공"
    ]
}

# ========================
# 설정 검증 함수
# ========================
def validate_config():
    """설정값들이 올바른지 검증"""
    errors = []
    
    # 필수 디렉토리 확인
    if not os.path.exists(DATA_DIR):
        errors.append(f"데이터 디렉토리가 없습니다: {DATA_DIR}")
    
    # 데이터 파일 확인
    if not os.path.exists(MATTRESS_DATA_PATH):
        errors.append(f"매트리스 데이터 파일이 없습니다: {MATTRESS_DATA_PATH}")
    
    # 가중치 합계 확인
    total_weight = sum(SCORING_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"점수 가중치 합계가 1.0이 아닙니다: {total_weight}")
    
    return errors

if __name__ == "__main__":
    print(f"🛏️ {PROJECT_NAME} v{VERSION}")
    print(f"📁 프로젝트 디렉토리: {BASE_DIR}")
    
    # 설정 검증
    errors = validate_config()
    if errors:
        print("❌ 설정 오류:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 설정이 올바릅니다!")