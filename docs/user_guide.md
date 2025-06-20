# 매트리스 AI 에이전트 사용자 가이드

> Enhanced RAG + GPT 동의어 + Few-shot 학습 강화 버전

## 📋 목차
1. [개요](#개요)
2. [시스템 요구사항](#시스템-요구사항)
3. [설치 및 설정](#설치-및-설정)
4. [사용 방법](#사용-방법)
5. [주요 기능](#주요-기능)
6. [질문 예시](#질문-예시)
7. [문제 해결](#문제-해결)
8. [기술적 특징](#기술적-특징)

---

## 개요

매트리스 AI 에이전트는 고객의 매트리스 구매 의사결정을 돕는 지능형 상담 시스템입니다. 실제 매트리스 전문가의 15년 경험을 바탕으로 한 Few-shot 학습과 GPT 기반 동의어 확장 기술을 활용하여 정확하고 개인화된 추천을 제공합니다.

### 🎯 핵심 가치
- **전문성**: 15년 경력 매트리스 전문가의 지식 학습
- **개인화**: 고객의 건강 상태, 예산, 선호도 맞춤 추천
- **신뢰성**: 실제 사용자 후기와 평점 기반 추천
- **친근함**: 친구 같은 전문가의 따뜻한 상담

---

## 시스템 요구사항

### 필수 요구사항
- **Python**: 3.8 이상
- **메모리**: 최소 4GB RAM (8GB 권장)
- **저장공간**: 최소 2GB 여유 공간

### 선택 사항
- **OpenAI API 키**: GPT 강화 기능 사용시 필요
- **인터넷 연결**: 최신 모델 다운로드 및 GPT 기능 사용

---

## 설치 및 설정

### 1. 프로젝트 구조 확인
```
매트리스_AI_프로젝트/
TEAM_10/
├── 📁 src/                          # 소스 코드 디렉토리
│   ├── 📄 generate_data.py          # Phase 0: 샘플 데이터 생성
│   ├── 📄 data_loader.py            # Phase 1: 데이터 로드 및 전처리
│   ├── 📄 rag_system.py             # Phase 2: 한국어 특화 RAG 시스템
│   ├── 📄 ai_agent.py               # Phase 3: 기본 AI Agent (OpenAI 통합)
│   └── 📄 few_shot_examples.py      # Phase 3+: Few-shot 학습 강화 모듈
├── 📁 data/                         # 데이터 파일 디렉토리
│   └── 📄 mattress_data.json        # 매트리스 데이터베이스
├── 📁 chroma_db/                    # 벡터 데이터베이스 (자동 생성)
│   ├── 📄 chroma.sqlite3            # ChromaDB 메타데이터
│   └── 📁 segments/                 # 벡터 임베딩 저장소
├── 📁 demo_db/                      # 데모 대화 저장 디렉토리
├── 📁 docs/                         # 문서 디렉토리
│   ├── 📄 user_guide.md             # 사용자 가이드
├── 📁 logs/                         # 로그 파일 (자동 생성)
│   └── 📄 mattress_ai.log           # 시스템 로그
├── 📄 main.py                       # 메인 실행 파일
├── 📄 config.py                     # 설정 파일
├── 📄 requirements.txt              # Python 의존성
├── 📄 interactive_demo.py           # 데모 대화 실행 파일
├── 📄 .gitignore                    # Git 무시 파일
├── 📄 README.md                     # 프로젝트 설명서
└── 📄 LICENSE                       # 라이선스 파일
```

### 2. 의존성 설치
```bash
pip install openai sentence-transformers chromadb torch numpy pandas
```

### 3. OpenAI API 키 설정 (선택사항)
```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Mac/Linux
export OPENAI_API_KEY="your-api-key-here"
```

### 4. 매트리스 데이터 준비
`data/mattress_data.json` 파일이 존재하는지 확인하세요. 없다면 샘플 데이터를 생성하거나 기존 데이터를 배치하세요.

---

## 사용 방법

### 기본 실행
```bash
python interactive_demo.py
```

### 실행 화면 예시
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  🛏️  매트리스 AI 상담사와 대화하기  🤖                      ║
║                                                                              ║
║              Enhanced RAG + GPT 동의어 + Few-shot 학습 적용                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

💬 매트리스에 대해 무엇이든 물어보세요!
```

### 대화 명령어
| 명령어 | 설명 |
|--------|------|
| `/help` | 도움말 및 사용법 보기 |
| `/status` | 시스템 상태 및 성능 확인 |
| `/history` | 현재 세션의 대화 기록 보기 |
| `/clear` | 화면 정리 |
| `/save` | 대화 기록을 JSON 파일로 저장 |
| `/quit` | 프로그램 종료 |

---

## 주요 기능

### 🔍 지능형 관련성 체크
매트리스와 관련된 질문만 처리하고, 무관한 질문은 친절하게 안내합니다.

**허용되는 질문**:
- 매트리스, 침대, 수면 관련
- 침구류 (베개, 이불 등)
- 수면 건강 (허리, 목 통증과 매트리스 연관)

**차단되는 질문**:
- 다른 가구 (서랍장, 소파, 책상 등)
- 가전제품, 음식, 날씨 등 무관한 주제

### 🧠 Enhanced RAG 검색
- **GPT 동의어 확장**: 질문의 의미를 정확히 파악하여 관련 동의어로 검색 확장
- **다중 전략 검색**: 3가지 검색 전략을 조합하여 최적 결과 도출
- **유사도 극대화**: 평균 0.2-0.25 포인트 유사도 향상

### 🎯 개인화 추천
- **건강 상태 분석**: 허리, 목, 어깨 등 건강 이슈 고려
- **예산 필터링**: 고객의 예산 범위 내에서 추천
- **선호도 학습**: 경도, 소재, 크기 등 개인 선호도 반영

### 📊 신뢰성 있는 정보 제공
- **실제 사용자 후기**: 기술적 스펙보다 실제 경험담 중심
- **평점 및 통계**: "평점 4.8/5점", "재구매율 92%" 등 구체적 수치
- **사용 기간별 효과**: "2-3주 내 수면 질 개선" 등 시간별 효과

---

## 질문 예시

### 건강 관련 질문
```
✅ "허리 디스크 환자용 딱딱한 매트리스 추천해주세요"
✅ "목이 아픈 사람한테 좋은 매트리스 있나요?"
✅ "어깨 결림 때문에 고생인데 어떤 매트리스가 좋을까요?"
```

### 예산 기반 질문
```
✅ "50만원 이하로 가성비 좋은 매트리스 찾고 있어요"
✅ "100만원 정도 예산으로 프리미엄 매트리스 추천해주세요"
✅ "200만원대 최고급 매트리스 어떤게 있나요?"
```

### 라이프스타일 질문
```
✅ "더위 많이 타는데 시원한 매트리스 있나요?"
✅ "신혼부부용 킹사이즈 매트리스 추천해주세요"
✅ "아이용 싱글 매트리스 찾고 있어요"
✅ "커플인데 동작 격리 잘 되는 매트리스 있나요?"
```

### 소재/기술 질문
```
✅ "메모리폼 매트리스 중에 좋은 제품 있나요?"
✅ "라텍스 매트리스 장단점 알려주세요"
✅ "스프링 매트리스 추천해주세요"
```

### 차단되는 질문 (안내 메시지 제공)
```
❌ "서랍장 추천해주세요" → 가구 관련 안내
❌ "배고파" → 매트리스 관련 질문 유도
❌ "소파 어떤게 좋아요?" → 전문 분야 안내
```

---

## 응답 예시

### 일반적인 응답 구조
```
🤖 AI 상담사 응답:

[상황 공감] 
더위를 많이 타시는군요. 여름철 수면이 정말 중요하죠 😊

[제품 추천]
'퍼플 하이브리드 프리미어 킹'을 추천드릴게요! 195만원으로 프리미엄이지만...

[사용자 후기]
실제 구매 후기를 보면 '에어컨 없이도 시원하게 잔다'는 평가가 압도적이에요.

[평점/통계]
평점 4.7/5점에 재구매율 95%를 자랑하고, 여름철 사용자의 85%가 '체감온도 3-4도 낮아진 느낌'이라고 답했어요.

[따뜻한 마무리]
시원하고 편안한 잠자리가 되시길 바라요!
```

### 성능 지표 표시
```
⚡ 처리 시간: 2.34초
🚀 적용된 강화 기능: GPT 동의어, Few-shot 학습, Enhanced RAG
🎯 평균 유사도: 87.5%

📋 추천 매트리스 상세:
1. 퍼플 하이브리드 프리미어 킹 (퍼플)
   💰 가격: 195만원
   📊 유사도: 91.2%
   🏷️ 타입: 젤그리드 하이브리드
   ✨ 특징: 젤그리드, 쿨링시스템, 동작격리
```

---

## 문제 해결

### 자주 발생하는 문제

#### 1. 프로그램 실행 오류
```bash
❌ ModuleNotFoundError: No module named 'openai'
```
**해결방법**: 의존성 재설치
```bash
pip install openai sentence-transformers chromadb torch
```

#### 2. 데이터 파일 없음
```bash
❌ 데이터 파일을 찾을 수 없습니다: data/mattress_data.json
```
**해결방법**: 
- `data` 폴더 생성
- `mattress_data.json` 파일 배치 또는 샘플 데이터 생성

#### 3. OpenAI API 키 관련
```bash
⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다
```
**해결방법**: 
- API 키 설정 (선택사항, 기본 기능은 API 키 없이도 동작)
- GPT 강화 기능 사용시에만 필요

#### 4. 메모리 부족
```bash
❌ CUDA out of memory
```
**해결방법**:
- 임베딩 모델을 CPU 모드로 전환
- 배치 크기 줄이기
- 더 작은 모델 사용

### 성능 최적화

#### GPU 사용 (권장)
- NVIDIA GPU + CUDA 설치시 자동으로 GPU 사용
- 임베딩 생성 속도 5-10배 향상

#### 메모리 관리
- 대화 기록이 많아지면 `/save` 명령어로 저장 후 새 세션 시작
- 큰 데이터셋 사용시 배치 크기 조절

---

## 기술적 특징

### Enhanced RAG 시스템
- **벡터 검색**: Sentence Transformers 기반 의미 검색
- **하이브리드 전략**: 키워드 + 의미 + GPT 동의어 조합
- **ChromaDB**: 고성능 벡터 데이터베이스 활용

### GPT 통합 기능
- **동의어 확장**: 검색 정확도 향상
- **의도 분석**: 고객 요구사항 정확한 파악
- **응답 생성**: 전문적이고 친근한 상담 제공

### Few-shot 학습
- **전문가 경험**: 15년 경력 매트리스 전문가의 상담 패턴 학습
- **상황별 대응**: 건강, 예산, 라이프스타일별 맞춤 응답
- **지속적 개선**: 대화 기록을 통한 성능 향상

### 품질 보증
- **관련성 체크**: 4단계 필터링으로 부적절한 질문 차단
- **응답 품질**: 실제 후기와 평점 기반 신뢰성 있는 정보
- **개인정보 보호**: 대화 기록 로컬 저장, 외부 전송 없음

---

## 대화 기록 관리

### 자동 저장 위치
- **폴더**: `demo_db/`
- **파일명**: `mattress_ai_conversation_YYYYMMDD_HHMMSS.json`
- **형식**: JSON (구조화된 데이터)

### 저장 내용
```json
{
  "session_start": "2025-06-15T14:30:22",
  "session_end": "2025-06-15T15:45:33",
  "total_conversations": 8,
  "conversation_history": [
    {
      "id": 1,
      "timestamp": "2025-06-15T14:31:15",
      "user_query": "허리 아픈 사람용 매트리스 추천",
      "ai_response": "허리가 아프시는군요...",
      "processing_time": 2.34,
      "avg_similarity": 0.875,
      "enhancements_used": ["GPT 동의어", "Few-shot 학습"]
    }
  ]
}
```

---

*매트리스 AI 에이전트와 함께 완벽한 수면을 찾아보세요! 🛏️✨*