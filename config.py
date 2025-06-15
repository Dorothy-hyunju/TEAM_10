"""
프로젝트 설정 파일
환경변수와 기본 설정을 관리합니다.
"""

import os
from pathlib import Path

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# 데이터 설정
DEFAULT_DATA_FILE = DATA_DIR / "mattress_data.json"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# OpenAI 설정
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 700
OPENAI_TEMPERATURE = 0.7

# 한국어 임베딩 모델 설정
KOREAN_EMBEDDING_MODELS = [
    "jhgan/ko-sroberta-multitask",           # 추천: 한국어 특화 RoBERTa
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",  # 한국어 SentenceBERT
    "BM-K/KoSimCSE-roberta-multitask",      # 한국어 SimCSE
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 다국어
    "all-MiniLM-L6-v2"                      # 폴백
]

# RAG 시스템 설정
RAG_CONFIG = {
    "max_results": 5,
    "similarity_threshold": 0.3,
    "batch_size": 16,
    "cache_size": 1000
}

# AI Agent 설정
AGENT_CONFIG = {
    "max_search_terms": 8,
    "conversation_history_limit": 50,
    "personalization_boost": 0.1
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "mattress_ai.log"
}

# API 키 설정 (환경변수에서 가져옴)
def get_openai_api_key():
    return os.getenv('OPENAI_API_KEY')

# 설정 검증
def validate_config():
    """설정 유효성 검사"""
    errors = []
    
    # 디렉토리 존재 확인
    for dir_path in [DATA_DIR, SRC_DIR]:
        if not dir_path.exists():
            errors.append(f"필수 디렉토리 없음: {dir_path}")
    
    # 데이터 파일 확인
    if not DEFAULT_DATA_FILE.exists():
        errors.append(f"데이터 파일 없음: {DEFAULT_DATA_FILE}")
    
    return errors

if __name__ == "__main__":
    print("🔧 매트리스 AI 설정 정보")
    print("=" * 40)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"데이터 디렉토리: {DATA_DIR}")
    print(f"ChromaDB 경로: {CHROMA_DB_PATH}")
    print(f"OpenAI 모델: {OPENAI_MODEL}")
    print(f"한국어 모델: {KOREAN_EMBEDDING_MODELS[0]}")
    
    # 설정 검증
    errors = validate_config()
    if errors:
        print("\n❌ 설정 오류:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ 설정 검증 완료")