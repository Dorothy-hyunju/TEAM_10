"""
AI 에이전트 - Enhanced RAG + GPT 동의어 + Few-shot 강화 버전
파일: src/ai_agent.py

주요 개선:
1. Enhanced RAG 시스템 통합
2. GPT 기반 동적 동의어 활용
3. Few-shot 학습 강화
4. 유사도 점수 극대화
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import time
import re

# OpenAI 클라이언트
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI 라이브러리가 필요합니다: pip install openai")

# 프로젝트 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.rag_system import EnhancedMattressRAGSystem, setup_enhanced_rag_system
except ImportError:
    print("⚠️ enhanced rag_system 모듈을 찾을 수 없습니다.")
    raise

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartRelevanceChecker:
    """효율적인 매트리스 관련성 체크 (키워드 + GPT)"""
    
    def __init__(self, openai_client=None):
        self.client = openai_client
        self.cache = {}  # 결과 캐시로 중복 호출 방지
        
        # 확실히 관련된 키워드 (즉시 통과) - 매트리스/침대/수면만
        self.certain_keywords = {
            '매트리스', '침대', '베드', '잠자리', '수면', '잠', '자는',
            '메모리폼', '라텍스', '스프링', '본넬', '포켓스프링',
            '싱글', '더블', '퀸', '킹사이즈', '침대사이즈',
            '베개', '이불', '침구', '침실용품', '수면용품'
        }
        
        # 확실히 무관한 키워드 (즉시 차단)
        self.irrelevant_keywords = {
            # 음식 관련
            '배고파', '밥', '음식', '먹는', '식사', '요리', '맛있는',
            # 날씨 관련
            '날씨', '비', '눈', '더위', '추위', '온도', '기온',
            # 엔터테인먼트
            '영화', '드라마', '게임', '책', '소설', '만화',
            '축구', '야구', '농구', '헬스', '달리기', '운동',
            # 다른 가구들 (매트리스/침대 제외)
            '서랍장', '옷장', '붙박이장', '화장대', '책상', '의자',
            '소파', '쇼파', '테이블', '식탁', '다이닝테이블',
            '선반', '책장', '진열장', '수납장', '신발장',
            '행거', '옷걸이', '거울', '스탠드', '조명',
            '커튼', '블라인드', '카펫', '러그', '매트',
            '쿠션', '방석', '등받이', '팔걸이',
            # 가전제품
            '냉장고', '세탁기', '에어컨', '텔레비전', 'tv',
            '전자레인지', '오븐', '청소기', '공기청정기',
            # 기타
            '일', '직장', '회사', '업무', '회의', '출근',
            '친구', '연애', '데이트', '만남', '헤어짐',
            '여행', '휴가', '놀러', '구경', '관광',
            '학교', '공부', '시험', '숙제', '과제',
            '돈', '투자', '주식', '부동산', '대출'
        }
        
        # 애매한 키워드 (GPT 판단 필요) - 수면과 관련 가능성이 있는 것들만
        self.ambiguous_keywords = {
            '허리', '목', '어깨', '통증', '아픈', '편안', '딱딱', '부드러운',
            '가격', '추천', '브랜드', '좋은', '편안한', '시원한', '따뜻한',
            '높은', '낮은', '크기', '사이즈', '무거운', '가벼운'
        }
    
    def check_relevance(self, query: str) -> Dict[str, Any]:
        """단계별 관련성 체크"""
        query_clean = query.lower().strip()
        
        # 캐시 확인
        if query_clean in self.cache:
            return self.cache[query_clean]
        
        # 1단계: 너무 짧은 질문
        if len(query.strip()) < 3:
            result = {
                'is_relevant': False,
                'reason': '질문이 너무 짧습니다',
                'confidence': 0.95,
                'method': 'length_check'
            }
            self.cache[query_clean] = result
            return result
        
        # 2단계: 확실히 관련된 키워드 체크
        if any(keyword in query_clean for keyword in self.certain_keywords):
            result = {
                'is_relevant': True,
                'reason': '매트리스 관련 키워드 발견',
                'confidence': 0.95,
                'method': 'certain_keywords'
            }
            self.cache[query_clean] = result
            return result
        
        # 3단계: 확실히 무관한 키워드 체크
        if any(keyword in query_clean for keyword in self.irrelevant_keywords):
            result = {
                'is_relevant': False,
                'reason': '매트리스와 무관한 키워드 발견',
                'confidence': 0.90,
                'method': 'irrelevant_keywords'
            }
            self.cache[query_clean] = result
            return result
        
        # 4단계: 애매한 경우만 GPT 호출 (비용 절약)
        has_ambiguous = any(keyword in query_clean for keyword in self.ambiguous_keywords)
        
        if has_ambiguous and self.client:
            result = self._gpt_relevance_check(query)
            result['method'] = 'gpt_check'
        else:
            # GPT 없거나 애매하지 않은 경우 → 보수적으로 관련 없음 처리
            result = {
                'is_relevant': False,
                'reason': '매트리스 관련성을 확인할 수 없습니다',
                'confidence': 0.70,
                'method': 'conservative_fallback'
            }
        
        self.cache[query_clean] = result
        return result
    
    def _gpt_relevance_check(self, query: str) -> Dict[str, Any]:
        """GPT로 관련성 체크 (최소 비용)"""
        try:
            system_prompt = """매트리스/침대/수면과 관련된 질문인지 판단해주세요.

허용되는 질문:
- 매트리스, 침대, 수면, 잠자리 관련
- 침구류 (베개, 이불, 매트리스패드 등)
- 수면 건강 (허리, 목 통증 등과 매트리스 연관)

차단되는 질문:
- 다른 가구 (서랍장, 소파, 책상, 의자 등)
- 가전제품 (냉장고, TV, 에어컨 등)
- 매트리스와 무관한 모든 질문

예시:
- "허리 아픈 사람 매트리스" → 관련있음
- "딱딱한 침대 추천" → 관련있음  
- "서랍장 추천해주세요" → 관련없음
- "소파 어떤게 좋아요?" → 관련없음
- "배고파" → 관련없음
- "허리 아파" → 애매 (매트리스와 연관 가능성 확인 필요)

형식: {"relevant": true/false, "reason": "이유"}"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"질문: '{query}'"}
                ],
                max_tokens=50,  # 매우 짧게 제한 (비용 절약)
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            try:
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    content = content[start:end].strip()
                
                result_data = json.loads(content)
                
                return {
                    'is_relevant': result_data.get('relevant', False),
                    'reason': result_data.get('reason', 'GPT 판단'),
                    'confidence': 0.85
                }
                
            except json.JSONDecodeError:
                # JSON 파싱 실패시 단순 키워드 기반 판단
                is_relevant = 'true' in content.lower() or 'relevant' in content.lower()
                return {
                    'is_relevant': is_relevant,
                    'reason': 'GPT 응답 기반 판단',
                    'confidence': 0.75
                }
                
        except Exception as e:
            logger.error(f"GPT 관련성 체크 실패: {e}")
            return {
                'is_relevant': False,
                'reason': 'GPT 체크 실패로 안전하게 무관 처리',
                'confidence': 0.60
            }
    
    def get_irrelevant_response(self, query: str, reason: str) -> str:
        """무관한 질문에 대한 안내 메시지"""
        
        # 가구 관련 질문 감지
        furniture_keywords = {
            '서랍장', '옷장', '화장대', '책상', '의자', '소파', '쇼파', 
            '테이블', '식탁', '선반', '책장', '수납장'
        }
        
        query_lower = query.lower()
        is_furniture_query = any(keyword in query_lower for keyword in furniture_keywords)
        
        if is_furniture_query:
            detected_furniture = [kw for kw in furniture_keywords if kw in query_lower]
            return f"죄송합니다. 저는 매트리스 전문 상담사입니다. {', '.join(detected_furniture)}와 같은 다른 가구는 추천드릴 수 없어요.\n\n매트리스, 침대, 또는 수면과 관련된 질문을 해주시면 도움을 드릴 수 있습니다! 😊"
        
        responses = {
            '질문이 너무 짧습니다': "질문을 좀 더 구체적으로 해주세요. 매트리스에 대해 궁금한 점을 자세히 말씀해주시면 도움드리겠습니다! 😊",
            '매트리스와 무관한 키워드 발견': "죄송합니다. 저는 매트리스 전문 상담사입니다. 매트리스, 침대, 수면과 관련된 질문만 도움드릴 수 있어요.\n\n예를 들어 '허리에 좋은 매트리스', '예산 내 추천 매트리스' 같은 질문을 해주세요! 😊",
            'GPT 체크 실패로 안전하게 무관 처리': "죄송합니다. 질문을 정확히 이해하지 못했습니다. 매트리스나 수면과 관련된 질문을 명확하게 해주시면 도움드리겠습니다! 😊",
            '매트리스 관련성을 확인할 수 없습니다': "죄송합니다. 매트리스나 수면과 관련된 질문인지 확실하지 않네요. 좀 더 구체적으로 질문해주시면 도움드리겠습니다!\n\n예: '딱딱한 매트리스 추천해주세요', '50만원대 가성비 매트리스 있나요?' 😊"
        }
        
        return responses.get(reason, "죄송합니다. 매트리스와 관련된 질문을 해주시면 도움드리겠습니다! 😊")

class EnhancedQueryProcessor:
    """Few-shot 강화 쿼리 프로세서"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.client = None
        self.model = model
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI 라이브러리가 설치되지 않았습니다")
            return
        
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API 키가 없습니다")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            
            # 연결 테스트
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "테스트"}],
                max_tokens=5
            )
            
            logger.info(f"Enhanced 쿼리 프로세서 초기화 완료 (모델: {self.model})")
            
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.client = None
    
    def expand_query_with_gpt_synonyms(self, user_query: str) -> Dict[str, Any]:
        """GPT 동의어 기반 쿼리 확장"""
        if not self.client:
            return self._fallback_query_expansion(user_query)
        
        try:
            system_prompt = """
    매트리스 검색 전문가로서 사용자 쿼리를 분석하고 확장하세요.

    Few-shot 예시:
    입력: "허리 아픈 사람 매트리스"
    출력: {
    "main_keywords": ["허리", "아픈", "매트리스"],
    "gpt_synonyms": {
        "허리": ["요추", "척추", "등", "허리통증", "요통"],
        "아픈": ["통증", "문제", "불편", "질환", "아픔"],
        "매트리스": ["침대", "베드", "수면용품", "잠자리"]
    },
    "related_terms": ["체압분산", "지지력", "딱딱한", "하드", "척추정렬"],
    "search_queries": [
        "허리 아픈 사람 매트리스",
        "요통 척추통증 매트리스",
        "허리디스크 체압분산 지지력",
        "딱딱한 하드 매트리스 허리"
    ]
    }

    반드시 유효한 JSON 형식으로만 응답하세요. 추가 설명은 하지 마세요.
    """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"쿼리 분석: {user_query}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # JSON 파싱 시도
            try:
                content = response.choices[0].message.content.strip()
                
                # JSON 추출 시도 (```json``` 블록이 있는 경우 처리)
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end != -1:
                        content = content[start:end].strip()
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    if end != -1:
                        content = content[start:end].strip()
                
                # 추가 정리
                content = content.replace("```", "").strip()
                if content.startswith("json"):
                    content = content[4:].strip()
                
                result = json.loads(content)
                
                # 검색어 조합 생성
                search_terms = [user_query]  # 원본
                
                # GPT 동의어 기반 조합
                if result.get('gpt_synonyms'):
                    for original, synonyms in result['gpt_synonyms'].items():
                        search_terms.extend(synonyms[:3])  # 상위 3개
                
                # 관련 용어 추가
                if result.get('related_terms'):
                    search_terms.extend(result['related_terms'][:3])
                
                # 검색 쿼리 추가
                if result.get('search_queries'):
                    search_terms.extend(result['search_queries'])
                
                # 확장 쿼리 생성
                main_keywords = result.get('main_keywords', [])
                expanded_query = f"{user_query} {' '.join(main_keywords[:3])}" if main_keywords else user_query
                
                return {
                    'original_query': user_query,
                    'expanded_query': expanded_query,
                    'main_keywords': main_keywords,
                    'gpt_synonyms': result.get('gpt_synonyms', {}),
                    'related_terms': result.get('related_terms', []),
                    'search_terms': list(set(search_terms))[:8],  # 중복 제거, 최대 8개
                    'gpt_enhanced': True
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"GPT 응답 JSON 파싱 실패: {e}")
                logger.error(f"응답 내용: {content}")
                # 폴백으로 단순 확장
                return self._create_simple_expansion(user_query)
                
        except Exception as e:
            logger.error(f"GPT 쿼리 확장 실패: {e}")
        
        return self._fallback_query_expansion(user_query)


    def _create_simple_expansion(self, user_query: str) -> Dict[str, Any]:
        """JSON 파싱 실패시 단순 확장"""
        keywords = user_query.split()
        
        return {
            'original_query': user_query,
            'expanded_query': user_query,
            'main_keywords': keywords,
            'gpt_synonyms': {},
            'related_terms': [],
            'search_terms': [user_query],
            'gpt_enhanced': False
        }

    def analyze_user_intent_with_few_shot(self, user_query: str) -> Dict:
        """Few-shot 강화 사용자 의도 분석"""
        if not self.client:
            return self._basic_intent_analysis(user_query)
        
        try:
            system_prompt = """
매트리스 전문가로서 사용자 의도를 분석하세요.

Few-shot 예시:
입력: "허리 디스크 환자 딱딱한 매트리스 80만원 이하"
출력: {
  "intent_type": "health_focused",
  "urgency": "high",
  "budget_info": {
    "has_budget": true,
    "range": "80만원 이하",
    "min": 0,
    "max": 80
  },
  "health_info": {
    "has_issue": true,
    "issues": ["허리", "디스크"],
    "severity": "high"
  },
  "preferences": {
    "firmness": "딱딱",
    "health_priority": true
  },
  "confidence": 0.95
}

입력: "신혼부부용 킹사이즈 쿨링 매트리스"
출력: {
  "intent_type": "lifestyle_focused",
  "urgency": "medium", 
  "budget_info": {"has_budget": false},
  "health_info": {"has_issue": false, "issues": []},
  "preferences": {
    "size": "킹",
    "temperature": "시원함",
    "user_type": "커플"
  },
  "confidence": 0.90
}

JSON 형식으로만 응답하세요.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"의도 분석: {user_query}"}
                ],
                max_tokens=400,
                temperature=0.2
            )
            
            try:
                intent = json.loads(response.choices[0].message.content.strip())
                intent['few_shot_enhanced'] = True
                logger.info(f"Few-shot 의도 분석: {intent.get('intent_type', 'unknown')}")
                return intent
            except json.JSONDecodeError:
                logger.error("Few-shot 의도 분석 JSON 파싱 실패")
                
        except Exception as e:
            logger.error(f"Few-shot 의도 분석 실패: {e}")
        
        return self._basic_intent_analysis(user_query)
    
    def _fallback_query_expansion(self, user_query: str) -> Dict[str, Any]:
        """폴백 쿼리 확장"""
        return {
            'original_query': user_query,
            'expanded_query': user_query,
            'main_keywords': user_query.split(),
            'gpt_synonyms': {},
            'related_terms': [],
            'search_terms': [user_query],
            'gpt_enhanced': False
        }
    
    def _basic_intent_analysis(self, user_query: str) -> Dict:
        """기본 의도 분석"""
        query_lower = user_query.lower()
        
        intent = {
            'intent_type': 'search',
            'urgency': 'medium',
            'budget_info': {'has_budget': False},
            'health_info': {'has_issue': False, 'issues': []},
            'preferences': {},
            'confidence': 0.5,
            'few_shot_enhanced': False
        }
        
        # 건강 이슈 감지
        health_keywords = ['허리', '목', '어깨', '관절', '통증', '아픔', '디스크']
        found_health = [kw for kw in health_keywords if kw in query_lower]
        if found_health:
            intent['health_info'] = {
                'has_issue': True,
                'issues': found_health,
                'severity': 'high' if any(word in query_lower for word in ['통증', '아픔']) else 'medium'
            }
            intent['urgency'] = 'high'
            intent['intent_type'] = 'health_focused'
        
        # 예산 감지
        budget_pattern = r'(\d+)\s*만원'
        budget_matches = re.findall(budget_pattern, query_lower)
        if budget_matches:
            budgets = [int(b) for b in budget_matches]
            intent['budget_info'] = {
                'has_budget': True,
                'range': f"{min(budgets)}-{max(budgets)}만원" if len(budgets) > 1 else f"{budgets[0]}만원",
                'min': min(budgets),
                'max': max(budgets)
            }
        
        return intent


class EnhancedResponseGenerator:
    """Few-shot 강화 응답 생성기"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.client = None
        self.model = model
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI 라이브러리가 설치되지 않았습니다")
            return
        
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API 키가 없습니다")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Enhanced 응답 생성기 초기화 완료 (모델: {self.model})")
        except Exception as e:
            logger.error(f"OpenAI 응답 생성기 초기화 실패: {e}")
    
    def generate_enhanced_response(self, user_query: str, search_results: List[Dict], 
                                 user_intent: Optional[Dict] = None,
                                 query_expansion: Optional[Dict] = None) -> str:
        """Few-shot 강화 응답 생성"""
        if not self.client:
            return self._generate_fallback_response(user_query, search_results)
        
        if not search_results:
            return self._generate_no_results_response(user_query, user_intent)
        
        try:
            # Few-shot 예시 포함 시스템 프롬프트
            system_prompt = """
15년 경력 매트리스 전문가로서 고객에게 최적화된 상담을 제공하세요.

Few-shot 응답 예시:

예시 1:
고객 질문: "허리 디스크 환자용 딱딱한 매트리스 추천"
검색 결과: 에이스 BPA 1000 하드 (69만원, 본넬스프링, 척추지지)
전문가 응답: "허리 디스크로 고생하고 계시는군요. '에이스 BPA 1000 하드'를 추천드립니다. 69만원으로 본넬스프링 구조의 딱딱한 타입이며, 척추지지력이 뛰어나 디스크 환자분들께 효과적입니다. 하드 타입이라 허리가 과도하게 꺾이지 않도록 도와주고, 체중 분산도 우수합니다."

예시 2:
고객 질문: "더위 타는 사람용 시원한 매트리스"
검색 결과: 퍼플 그리드 (180만원, 젤그리드, 쿨링)
전문가 응답: "더위를 많이 타시는군요. '퍼플 그리드'를 추천드립니다. 180만원으로 프리미엄이지만 젤그리드 기술로 탁월한 쿨링 효과를 제공합니다. 독특한 그리드 구조가 공기 순환을 극대화하여 여름철에도 시원하게 주무실 수 있습니다."

응답 가이드라인:
1. 고객 상황 공감 표현
2. 추천 매트리스명과 가격 명시
3. 핵심 특징 2-3개 설명
4. 고객 요구사항에 맞는 구체적 이유
5. 전문적이면서 친근한 톤, 재치있으면서 칭찬하는 톤
6. 300-400자 내외
"""
            
            # 검색 결과 컨텍스트 (상위 1개만 사용)
            top_mattress = search_results[0]
            context = f"""
추천 매트리스:
- 이름: {top_mattress.get('name', 'Unknown')}
- 브랜드: {top_mattress.get('brand', 'Unknown')}  
- 가격: {top_mattress.get('price', 0)}만원
- 타입: {top_mattress.get('type', 'Unknown')}
- 특징: {', '.join(top_mattress.get('features', [])[:3])}
- 추천대상: {', '.join(top_mattress.get('target_users', [])[:2])}
- 유사도: {top_mattress.get('similarity_score', 0):.3f}
"""
            
            # 사용자 의도 정보
            intent_info = ""
            if user_intent:
                intent_parts = []
                
                if user_intent.get('health_info', {}).get('has_issue'):
                    issues = user_intent['health_info'].get('issues', [])
                    intent_parts.append(f"건강 이슈: {', '.join(issues)}")
                
                if user_intent.get('budget_info', {}).get('has_budget'):
                    intent_parts.append(f"예산: {user_intent['budget_info'].get('range', '')}")
                
                preferences = user_intent.get('preferences', {})
                if preferences:
                    pref_list = [f"{k}: {v}" for k, v in preferences.items() if v]
                    if pref_list:
                        intent_parts.append(f"선호도: {', '.join(pref_list)}")
                
                if intent_parts:
                    intent_info = f"\n\n고객 요구사항: {' | '.join(intent_parts)}"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"고객 질문: {user_query}\n\n{context}{intent_info}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            final_response = response.choices[0].message.content.strip()
            logger.info("Enhanced 응답 생성 완료")
            return final_response
            
        except Exception as e:
            logger.error(f"Enhanced 응답 생성 실패: {e}")
            return self._generate_fallback_response(user_query, search_results)
    
    def _generate_no_results_response(self, user_query: str, user_intent: Optional[Dict]) -> str:
        """결과 없음 응답"""
        return "죄송합니다. 현재 조건에 맞는 매트리스를 찾을 수 없습니다. 조건을 조정해서 다시 검색해보시겠어요?"
    
    def _generate_fallback_response(self, user_query: str, search_results: List[Dict]) -> str:
        """폴백 응답"""
        if not search_results:
            return "죄송합니다. 현재 조건에 맞는 매트리스를 찾을 수 없습니다."
        
        top_mattress = search_results[0]
        features = top_mattress.get('features', [])[:2]
        features_text = ', '.join(features) if features else '우수한 품질'
        
        return f"{top_mattress.get('name', 'Unknown')}을 추천드립니다. {top_mattress.get('price', 0)}만원으로 {features_text}가 특징이며, 고객님의 요구사항에 적합합니다."


class ConversationManager:
    """대화 관리"""
    
    def __init__(self):
        self.conversation_history = []
        self.user_context = {}
        self.session_start = datetime.now()
        self.interaction_count = 0
    
    def add_interaction(self, user_query: str, agent_response: str, 
                       search_results: Optional[List[Dict]] = None,
                       user_intent: Optional[Dict] = None,
                       query_expansion: Optional[Dict] = None,
                       filtered_question: bool = False):
        """대화 기록 추가"""
        self.interaction_count += 1
        
        interaction = {
            'id': self.interaction_count,
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'agent_response': agent_response,
            'search_results_count': len(search_results) if search_results else 0,
            'user_intent': user_intent,
            'query_expansion': query_expansion,
            'top_result': search_results[0] if search_results else None,
            'filtered_question': filtered_question,
            'enhanced_features': {
                'gpt_synonyms_used': query_expansion.get('gpt_enhanced', False) if query_expansion else False,
                'few_shot_enhanced': user_intent.get('few_shot_enhanced', False) if user_intent else False
            }
        }
        
        self.conversation_history.append(interaction)
        
        if user_intent:
            self._update_user_context(user_intent)
    
    def _update_user_context(self, user_intent: Dict):
        """사용자 컨텍스트 업데이트"""
        # 예산 정보
        budget_info = user_intent.get('budget_info', {})
        if budget_info.get('has_budget'):
            self.user_context['current_budget'] = budget_info
        
        # 건강 이슈
        health_info = user_intent.get('health_info', {})
        if health_info.get('has_issue'):
            if 'health_issues' not in self.user_context:
                self.user_context['health_issues'] = set()
            self.user_context['health_issues'].update(health_info.get('issues', []))
        
        # 선호도
        preferences = user_intent.get('preferences', {})
        if preferences:
            if 'preferences' not in self.user_context:
                self.user_context['preferences'] = {}
            self.user_context['preferences'].update(preferences)
    
    def get_conversation_summary(self) -> Dict:
        """대화 요약"""
        enhanced_count = len([h for h in self.conversation_history 
                            if h.get('enhanced_features', {}).get('gpt_synonyms_used')])
        
        return {
            'total_interactions': self.interaction_count,
            'enhanced_interactions': enhanced_count,
            'enhancement_rate': enhanced_count / max(self.interaction_count, 1),
            'user_context': self.user_context,
            'session_start': self.session_start.isoformat()
        }


class EnhancedMattressAIAgent:
    """Enhanced 매트리스 AI 에이전트"""
    
    def __init__(self, api_key: Optional[str] = None, data_path: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo"):
        
        # Enhanced RAG 시스템 초기화
        try:
            self.rag_system, rag_success = setup_enhanced_rag_system(
                data_path=data_path, 
                openai_api_key=api_key
            )
            if not rag_success:
                raise RuntimeError("Enhanced RAG 시스템 초기화 실패")
        except Exception as e:
            logger.error(f"Enhanced RAG 시스템 초기화 오류: {e}")
            raise
        
        # Enhanced OpenAI 컴포넌트 초기화
        self.query_processor = EnhancedQueryProcessor(api_key, model)
        self.response_generator = EnhancedResponseGenerator(api_key, model)
        
        # 대화 관리자
        self.conversation_manager = ConversationManager()
        
         # 관련성 체크 추가 <<<<
        gpt_client = self.query_processor.client if self.query_processor else None
        self.relevance_checker = SmartRelevanceChecker(gpt_client)
        
        # 시스템 상태
        self.is_ready = True
        self.openai_available = (self.query_processor.client is not None and 
                               self.response_generator.client is not None)
        
        logger.info("Enhanced 매트리스 AI 에이전트 초기화 완료")
        logger.info(f"Enhanced RAG: {'✅' if rag_success else '❌'}")
        logger.info(f"OpenAI 통합: {'✅' if self.openai_available else '❌'}")
        logger.info(f"GPT 동의어: {'✅' if self.rag_system.gpt_available else '❌'}")
        logger.info(f"모델: {model}")
    
    def process_query(self, user_query: str, n_results: int = 5) -> Dict:
        """Enhanced 쿼리 처리 (GPT 동의어 + Few-shot + 다중 전략)"""
        if not self.is_ready:
            return {
                'error': 'Enhanced AI 에이전트가 준비되지 않았습니다',
                'user_query': user_query,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
        
        logger.info(f"Enhanced AI 에이전트 처리 시작: '{user_query}'")
        start_time = time.time()
        
        try:
            # Step 0: 관련성 체크 (새로 추가) <<<<
            logger.info("Step 0: 질문 관련성 체크")
            relevance_result = self.relevance_checker.check_relevance(user_query)
            
            if not relevance_result['is_relevant']:
                # 무관한 질문에 대한 안내 응답
                irrelevant_response = self.relevance_checker.get_irrelevant_response(
                    user_query, relevance_result['reason']
                )
                
                # 대화 기록 저장 (필터링된 질문으로)
                self.conversation_manager.add_interaction(
                    user_query, irrelevant_response, [], None, None, filtered_question=True
                )
                
                return {
                    'user_query': user_query,
                    'agent_response': irrelevant_response,
                    'relevance_check': relevance_result,
                    'search_results': [],
                    'total_results': 0,
                    'processing_time': round(time.time() - start_time, 2),
                    'enhancement_info': {
                        'question_filtered': True,
                        'filter_reason': relevance_result['reason'],
                        'filter_method': relevance_result.get('method', 'unknown')
                    },
                    'openai_used': self.openai_available,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'filtered_question': True
                }


            # Step 1: GPT 기반 사용자 의도 분석 (Few-shot 강화)
            logger.info("Step 1: Enhanced 사용자 의도 분석")
            user_intent = self.query_processor.analyze_user_intent_with_few_shot(user_query)
            
            # Step 2: GPT 동의어 기반 쿼리 확장
            logger.info("Step 2: GPT 동의어 쿼리 확장")
            query_expansion = self.query_processor.expand_query_with_gpt_synonyms(user_query)
            
            # Step 3: Enhanced RAG 검색 (다중 전략)
            logger.info("Step 3: Enhanced RAG 다중 전략 검색")
            
            # 예산 필터 준비
            budget_filter = None
            budget_info = user_intent.get('budget_info', {})
            if budget_info.get('has_budget'):
                budget_min = budget_info.get('min', 0)
                budget_max = budget_info.get('max', 1000)
                budget_filter = (budget_min, budget_max)
            
            # Enhanced RAG 검색 실행
            search_results = self.rag_system.search_mattresses(
                user_query, 
                n_results=n_results,
                budget_filter=budget_filter
            )
            
            # Step 4: Enhanced 응답 생성 (Few-shot 강화)
            logger.info("Step 4: Enhanced 응답 생성")
            agent_response = self.response_generator.generate_enhanced_response(
                user_query, search_results, user_intent, query_expansion
            )
            
            # 대화 기록 저장
            self.conversation_manager.add_interaction(
                user_query, agent_response, search_results, user_intent, query_expansion
            )
            
            end_time = time.time()
            
            # 향상된 결과 구성
            result = {
                'user_query': user_query,
                'user_intent': user_intent,
                'query_expansion': query_expansion,
                'search_results': search_results,
                'agent_response': agent_response,
                'total_results': len(search_results),
                'processing_time': round(end_time - start_time, 2),
                'enhancement_info': {
                    'gpt_synonyms_used': query_expansion.get('gpt_enhanced', False),
                    'few_shot_enhanced': user_intent.get('few_shot_enhanced', False),
                    'enhanced_rag_used': True,
                    'strategies_used': search_results[0].get('strategies_used', []) if search_results else []
                },
                'similarity_scores': [r.get('similarity_score', 0) for r in search_results],
                'avg_similarity': sum(r.get('similarity_score', 0) for r in search_results) / len(search_results) if search_results else 0,
                'openai_used': self.openai_available,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info(f"Enhanced AI 에이전트 처리 완료 ({result['processing_time']}초)")
            logger.info(f"평균 유사도: {result['avg_similarity']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced AI 에이전트 처리 실패: {e}")
            return {
                'error': str(e),
                'user_query': user_query,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_personalized_recommendations(self, n_results: int = 5) -> List[Dict]:
        """개인화 추천 (Enhanced)"""
        try:
            user_profile = self.conversation_manager.get_conversation_summary()['user_context']
            
            # 개인화 쿼리 생성
            query_parts = []
            
            if user_profile.get('health_issues'):
                query_parts.extend(list(user_profile['health_issues']))
            
            preferences = user_profile.get('preferences', {})
            if preferences:
                for key, value in preferences.items():
                    if value:
                        query_parts.append(f"{key} {value}")
            
            personalized_query = ' '.join(query_parts) if query_parts else "추천 매트리스"
            
            # 예산 필터
            budget_filter = None
            current_budget = user_profile.get('current_budget', {})
            if current_budget.get('has_budget'):
                budget_filter = (current_budget.get('min', 0), current_budget.get('max', 1000))
            
            # Enhanced 검색
            results = self.rag_system.search_mattresses(
                personalized_query, 
                n_results=n_results,
                budget_filter=budget_filter
            )
            
            # 개인화 점수 추가
            for result in results:
                result['personalized'] = True
                # 개인화 보너스 적용
                original_score = result.get('similarity_score', 0)
                personalization_bonus = 0.1 if user_profile.get('health_issues') else 0.05
                result['similarity_score'] = min(original_score + personalization_bonus, 1.0)
            
            return results
            
        except Exception as e:
            logger.error(f"개인화 추천 실패: {e}")
            return []
    
    def compare_mattresses(self, mattress_ids: List[str]) -> Dict:
        """매트리스 비교 분석 (Enhanced)"""
        try:
            mattresses = []
            for mattress_id in mattress_ids:
                mattress = self.rag_system.get_mattress_by_id(mattress_id)
                if mattress:
                    mattresses.append(mattress)
            
            if not mattresses:
                return {'error': '비교할 매트리스를 찾을 수 없습니다'}
            
            comparison = {
                'mattresses': mattresses,
                'comparison_table': {},
                'enhanced_analysis': None
            }
            
            # 기본 비교 테이블
            attributes = ['price', 'type', 'brand', 'features']
            for attr in attributes:
                comparison['comparison_table'][attr] = []
                for mattress in mattresses:
                    if attr == 'features':
                        features = mattress.get(attr, [])
                        comparison['comparison_table'][attr].append(features[:3] if features else [])
                    else:
                        comparison['comparison_table'][attr].append(mattress.get(attr, '정보 없음'))
            
            # Enhanced AI 비교 분석
            if self.response_generator.client:
                try:
                    mattress_info = []
                    for m in mattresses:
                        features = m.get('features', [])
                        features_text = ', '.join(features[:3]) if features else '정보 없음'
                        mattress_info.append(
                            f"- {m.get('name', 'Unknown')} ({m.get('brand', 'Unknown')}): "
                            f"{m.get('price', 0)}만원, {m.get('type', 'Unknown')}, {features_text}"
                        )
                    
                    system_prompt = f"""
매트리스 전문가로서 다음 매트리스들을 심층 비교 분석해주세요:

{chr(10).join(mattress_info)}

분석 관점:
1. 가격 대비 성능 분석
2. 건강상 장점 비교 
3. 내구성 및 품질 평가
4. 각 매트리스 최적 사용자
5. 핵심 장단점 요약

전문적이면서도 이해하기 쉽게 400-500자로 작성해주세요.
"""
                    
                    response = self.response_generator.client.chat.completions.create(
                        model=self.response_generator.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "위 매트리스들을 전문적으로 비교 분석해주세요."}
                        ],
                        max_tokens=600,
                        temperature=0.6
                    )
                    
                    comparison['enhanced_analysis'] = response.choices[0].message.content.strip()
                    
                except Exception as e:
                    logger.error(f"Enhanced 비교 분석 실패: {e}")
                    comparison['enhanced_analysis'] = "Enhanced 분석을 사용할 수 없습니다."
            
            return comparison
            
        except Exception as e:
            logger.error(f"매트리스 비교 실패: {e}")
            return {'error': f'비교 분석 실패: {str(e)}'}
    
    def get_agent_status(self) -> Dict:
        """Enhanced 에이전트 상태"""
        try:
            rag_status = self.rag_system.get_system_status()
            conversation_summary = self.conversation_manager.get_conversation_summary()
            
            return {
                'ready': self.is_ready,
                'enhanced_rag_system': rag_status,
                'openai_available': self.openai_available,
                'openai_model': getattr(self.query_processor, 'model', 'N/A'),
                'conversation': conversation_summary,
                'enhanced_capabilities': {
                    'gpt_dynamic_synonyms': rag_status.get('gpt_available', False),
                    'few_shot_learning': True,
                    'multi_strategy_search': True,
                    'enhanced_query_expansion': self.openai_available,
                    'enhanced_intent_analysis': self.openai_available,
                    'enhanced_response_generation': self.openai_available,
                    'personalization': True,
                    'comparison_analysis': self.openai_available
                },
                'enhancement_stats': {
                    'total_interactions': conversation_summary['total_interactions'],
                    'enhanced_interactions': conversation_summary['enhanced_interactions'],
                    'enhancement_rate': conversation_summary['enhancement_rate']
                },
                'session_start': self.conversation_manager.session_start.isoformat()
            }
        except Exception as e:
            logger.error(f"상태 정보 조회 실패: {e}")
            return {
                'ready': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# 편의 함수
def create_enhanced_mattress_agent(api_key: Optional[str] = None, data_path: Optional[str] = None,
                                 model: str = "gpt-3.5-turbo") -> EnhancedMattressAIAgent:
    """Enhanced 매트리스 AI 에이전트 생성"""
    try:
        agent = EnhancedMattressAIAgent(api_key, data_path, model)
        logger.info("✅ Enhanced 매트리스 AI 에이전트 생성 완료")
        return agent
    except Exception as e:
        logger.error(f"Enhanced AI 에이전트 생성 실패: {e}")
        raise


# 기존 호환성을 위한 별칭
MattressAIAgent = EnhancedMattressAIAgent
create_mattress_agent = create_enhanced_mattress_agent


# 테스트 실행
if __name__ == "__main__":
    print("🚀 Enhanced 매트리스 AI 에이전트 테스트")
    print("=" * 70)
    
    try:
        # Enhanced AI 에이전트 생성
        api_key = os.getenv('OPENAI_API_KEY')
        agent = create_enhanced_mattress_agent(api_key)
        
        # 에이전트 상태 확인
        status = agent.get_agent_status()
        print(f"\n📊 Enhanced 에이전트 상태:")
        print(f"  준비 상태: {status['ready']}")
        print(f"  Enhanced RAG: {status['enhanced_rag_system']['initialized']}")
        print(f"  OpenAI 연동: {status['openai_available']}")
        print(f"  GPT 동의어: {status['enhanced_rag_system']['gpt_available']}")
        print(f"  사용 모델: {status.get('openai_model', 'N/A')}")
        
        enhanced_capabilities = status['enhanced_capabilities']
        print(f"\n🚀 Enhanced 기능:")
        for feature, available in enhanced_capabilities.items():
            print(f"   {'✅' if available else '❌'} {feature}")
        
        # Enhanced 테스트
        test_queries = [
            "허리 디스크 환자인데 딱딱하고 좋은 매트리스 80만원 이하로 추천해주세요",
            "더위 많이 타는 사람이 쓸 수 있는 쿨링 매트리스 있나요?",
            "신혼부부용 킹사이즈 메모리폼 매트리스 찾고 있어요",
            "50만원대 가성비 좋은 브랜드 매트리스 추천"
        ]
        
        print(f"\n🧪 Enhanced AI 에이전트 테스트:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*70}")
            print(f"테스트 {i}: '{query}'")
            print(f"{'='*70}")
            
            result = agent.process_query(query, n_results=3)
            
            if result.get('success'):
                enhancement_info = result['enhancement_info']
                print(f"🎯 의도 분석: {result['user_intent'].get('intent_type', 'unknown')}")
                print(f"🔧 GPT 동의어: {enhancement_info['gpt_synonyms_used']}")
                print(f"🎓 Few-shot 강화: {enhancement_info['few_shot_enhanced']}")
                print(f"🔍 사용 전략: {', '.join(enhancement_info['strategies_used'])}")
                print(f"📊 평균 유사도: {result['avg_similarity']:.3f}")
                print(f"⏱️ 처리 시간: {result['processing_time']}초")
                
                if result['search_results']:
                    top_result = result['search_results'][0]
                    print(f"🥇 1순위: {top_result.get('name', 'Unknown')} ({top_result.get('brand', 'Unknown')})")
                    print(f"    가격: {top_result.get('price', 0)}만원")
                    print(f"    유사도: {top_result.get('similarity_score', 0):.3f}")
                    print(f"    Enhanced: {top_result.get('gpt_enhanced', False)}")
                
                print(f"\n🤖 Enhanced AI 응답:")
                print(f"{'─' * 50}")
                print(f"{result['agent_response']}")
                print(f"{'─' * 50}")
                
            else:
                print(f"❌ 처리 실패: {result.get('error', '알 수 없는 오류')}")
            
            if i < len(test_queries):
                time.sleep(2)  # API 제한 방지
        
        # Enhancement 통계
        final_status = agent.get_agent_status()
        enhancement_stats = final_status['enhancement_stats']
        print(f"\n📈 Enhancement 통계:")
        print(f"  총 상호작용: {enhancement_stats['total_interactions']}회")
        print(f"  Enhanced 상호작용: {enhancement_stats['enhanced_interactions']}회") 
        print(f"  Enhancement 적용률: {enhancement_stats['enhancement_rate']:.1%}")
        
        print(f"\n✅ Enhanced AI 에이전트 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
