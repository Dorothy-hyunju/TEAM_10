# 🛏️ 매트리스 상담 AI Agent

ReAct 방식의 RAG 시스템을 활용한 매트리스 전문 상담 AI Agent입니다.

## 📋 프로젝트 개요

이 프로젝트는 매트리스 데이터를 기반으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공하는 AI 상담 시스템입니다. ReAct(Reasoning + Acting) 방식을 사용하여 단계적 추론과 검색을 통해 최적의 답변을 생성합니다.

## 🏗️ 프로젝트 구조

```
TEAM_10/
├── 📁 src/                    # 소스 코드
│   ├── 📄 generate_data.py    # Phase 0: 데이터 생성
│   ├── 📄 data_loader.py      # Phase 1: 데이터 로드
│   ├── 📄 rag_system.py       # Phase 2: RAG 시스템
│   ├── 📄 ai_agent.py         # Phase 3: AI Agent
│   └── 📄 web_app.py          # Phase 4: Streamlit 앱
├── 📁 data/                   # 데이터 파일
│   └── 📄 mattress_data.json  # 매트리스 데이터
├── 📁 tests/                  # 테스트 코드
│   ├── 📄 test_rag.py         # RAG 테스트
│   └── 📄 test_agent.py       # Agent 테스트
├── 📁 docs/                   # 문서
│   └── 📄 user_guide.md       # 사용자 가이드
├── 📄 main.py                 # 메인 실행 파일
├── 📄 requirements.txt        # 필요한 라이브러리
├── 📄 README.md               # 프로젝트 설명
├── 📄 .gitignore              # Git 무시 파일
└── 📄 config.py               # 설정 파일
```

## 🚀 주요 기능

### 🎯 Phase 2: RAG 시스템 (`src/rag_system.py`)
- **허깅페이스 임베딩**: 무료 한국어 지원 모델 (`ko-sroberta-multitask`)
- **내적 기반 유사도**: 효율적인 벡터 유사도 계산
- **OpenAI GPT-3.5**: 비용 효율적인 답변 생성
- **컨텍스트 기반 검색**: 상위 K개 문서 검색 및 포맷팅

### 🤖 Phase 3: AI Agent (`src/ai_agent.py`)
- **ReAct 방식**: Thought → Action → Observation → Answer
- **반복적 개선**: 최대 3회 반복으로 답변 품질 향상
- **대화 기록**: 상담 이력 관리
- **스마트 쿼리 개선**: 이전 답변 기반 쿼리 재구성

## 💰 비용 효율적 설계

- ✅ **허깅페이스 임베딩**: 완전 무료
- ✅ **GPT-3.5-turbo**: GPT-4 대비 약 10배 저렴
- ✅ **토큰 제한**: max_tokens=500으로 비용 절약
- ✅ **최적화된 반복**: 최대 3회로 API 호출 최소화