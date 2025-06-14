"""
프로젝트 설정 파일
환경변수와 기본 설정을 관리합니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

# .env 파일 로드
load_dotenv(PROJECT_ROOT / '.env')

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')  # 선택사항

# 모델 설정
EMBEDDING_MODEL = 'jhgan/ko-sroberta-multitask'  # 한국어 지원 임베딩 모델
LLM_MODEL = 'gpt-3.5-turbo'  # 비용 효율적인 LLM
MAX_TOKENS = 500  # 토큰 제한으로 비용 절약
TEMPERATURE = 0.7

# 검색 설정
TOP_K = 5  # 상위 K개 문서 검색
MAX_ITERATIONS = 3  # ReAct 최대 반복 횟수
SIMILARITY_THRESHOLD = 0.5  # 유사도 임계값

# 데이터 경로
DATA_DIR = PROJECT_ROOT / 'data'
MATTRESS_DATA_PATH = DATA_DIR / 'mattress_data.json'

# 로그 설정
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Streamlit 설정
STREAMLIT_CONFIG = {
    'page_title': '매트리스 상담 AI',
    'page_icon': '🛏️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# 테스트 설정
TEST_DATA_SIZE = 10  # 테스트용 데이터 크기
TEST_QUERIES = [
    "10만원 이하 추천 매트리스",
    "메모리폼 소재 매트리스 특징",
    "단단한 매트리스 브랜드 추천",
    "가성비 좋은 퀸사이즈 매트리스",
    "허리 아픈 사람에게 좋은 매트리스"
]

# 검증 함수
def validate_config():
    """설정값 검증"""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    if not MATTRESS_DATA_PATH.exists():
        errors.append(f"매트리스 데이터 파일이 없습니다: {MATTRESS_DATA_PATH}")
    
    if TOP_K <= 0:
        errors.append("TOP_K는 양수여야 합니다.")
    
    if MAX_ITERATIONS <= 0:
        errors.append("MAX_ITERATIONS는 양수여야 합니다.")
    
    if errors:
        raise ConfigurationError("\n".join(errors))
    
    return True

class ConfigurationError(Exception):
    """설정 오류 예외"""
    pass

# 설정 정보 출력 함수
def print_config():
    """현재 설정 정보 출력"""
    print("🔧 현재 설정:")
    print(f"  📊 임베딩 모델: {EMBEDDING_MODEL}")
    print(f"  🤖 LLM 모델: {LLM_MODEL}")
    print(f"  🔍 검색 개수: {TOP_K}")
    print(f"  🔄 최대 반복: {MAX_ITERATIONS}")
    print(f"  📁 데이터 경로: {MATTRESS_DATA_PATH}")
    print(f"  🔑 API 키 설정: {'✅' if OPENAI_API_KEY else '❌'}")
