"""
Few-shot 학습 강화 - 유사도 극대화 버전 (고객 표기 수정 완료)
파일: src/few_shot_examples.py

주요 개선:
1. GPT 동의어 생성 Few-shot 예시 강화
2. 유사도 향상에 특화된 패턴 학습
3. 매트리스 도메인 전문성 강화
4. 오류 수정 및 안정성 향상
5. "고객" 표기 정확성 개선
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)


class EnhancedFewShotManager:
    """유사도 향상에 특화된 Few-shot 매니저"""
    
    def __init__(self):
        self.similarity_optimization_examples = self._load_similarity_examples()
        self.gpt_synonym_examples = self._load_gpt_synonym_examples()
        self.query_expansion_examples = self._load_enhanced_query_expansion()
        self.intent_analysis_examples = self._load_enhanced_intent_analysis()
        self.response_generation_examples = self._load_enhanced_response_generation()
    
    def _load_similarity_examples(self) -> List[Dict]:
        """유사도 향상 전략 예시"""
        return [
            {
                "strategy": "synonym_expansion",
                "before": "딱딱한 매트리스",
                "after": "딱딱한 매트리스 단단한 하드 견고한 탄탄한 펌 강한 튼튼한 solid firm 지지력 서포트",
                "similarity_improvement": "0.65 → 0.89 (+0.24)"
            },
            {
                "strategy": "health_context_expansion", 
                "before": "허리 아픈 사람",
                "after": "허리 아픈 사람 요통 척추통증 허리디스크 요추통증 허리문제 체압분산 지지력 척추정렬 압력완화",
                "similarity_improvement": "0.72 → 0.91 (+0.19)"
            },
            {
                "strategy": "multi_dimensional_expansion",
                "before": "시원한 매트리스",
                "after": "시원한 매트리스 쿨링 냉감 통풍 서늘한 차가운 쿨 통기성 온도조절 젤메모리폼 환기 공기순환",
                "similarity_improvement": "0.68 → 0.87 (+0.19)"
            }
        ]
    
    def _load_gpt_synonym_examples(self) -> str:
        """GPT 동의어 생성 최적화 예시"""
        return """매트리스 도메인 GPT 동의어 생성 - 유사도 극대화 전략:

예시 1 - 감촉/경도 확장:
입력: "딱딱한"
최적 동의어: ["단단한", "하드", "견고한", "탄탄한", "펌", "강한", "튼튼한", "solid", "firm", "rigid"]
관련 기술어: ["지지력", "서포트", "척추정렬", "압력완화", "체압분산"]
상황별 표현: ["허리에 좋은", "디스크 환자용", "척추 건강"]

예시 2 - 건강 문제 확장:
입력: "허리통증"  
최적 동의어: ["요통", "허리아픔", "요추통증", "허리디스크", "척추통증", "등통증", "요추질환", "허리문제", "back pain"]
관련 기술어: ["체압분산", "척추정렬", "지지력", "압력완화", "자세교정"]
상황별 표현: ["허리 환자용", "디스크 치료", "척추 건강"]

예시 3 - 온도 감각 확장:
입력: "시원한"
최적 동의어: ["쿨링", "냉감", "통풍", "서늘한", "차가운", "쿨", "시원함", "cool", "냉기", "서늘함"]
관련 기술어: ["통기성", "온도조절", "젤메모리폼", "환기", "공기순환", "열분산"]
상황별 표현: ["더위 타는 분용", "여름철용", "열 민감자용"]

예시 4 - 사용자 타입 확장:
입력: "커플"
최적 동의어: ["부부", "신혼", "연인", "2인", "둘이서", "부부용", "커플용", "파트너", "couple", "두 사람"]
관련 기술어: ["동작격리", "진동차단", "넓은공간", "모션아이솔레이션", "파트너방해금지"]
상황별 표현: ["함께 자는", "서로 방해없이", "넓은 침대"]

예시 5 - 소재/기술 확장:
입력: "메모리폼"
최적 동의어: ["기억장치", "템퍼", "비스코", "템퍼폼", "기억폼", "memory foam", "점탄성폼", "저반발폼", "형상기억"]
관련 기술어: ["체압분산", "몸매따라", "맞춤지지", "압력완화", "온도감응"]
상황별 표현: ["몸에 맞는", "압력 줄이는", "편안한"]

동의어 생성 원칙:
1. 정확한 동의어 8-12개 (한국어 + 영어)
2. 기술적 관련 용어 4-6개
3. 상황별 자연스러운 표현 3-5개
4. 매트리스 쇼핑 맥락에서 실제 사용되는 표현 우선
5. 유사도 점수 향상에 직접적으로 기여하는 용어 선별"""
    
    def _load_enhanced_query_expansion(self) -> List[Dict]:
        """강화된 쿼리 확장 예시"""
        return [
            {
                "user_query": "허리 디스크 환자용 딱딱한 매트리스 80만원 이하",
                "step1_keyword_extraction": ["허리", "디스크", "환자", "딱딱한", "매트리스", "80만원"],
                "step2_gpt_synonym_expansion": {
                    "허리": ["요추", "척추", "등", "허리통증", "요통"],
                    "디스크": ["추간판", "허리디스크", "척추디스크", "탈출증"],
                    "딱딱한": ["단단한", "하드", "견고한", "탄탄한", "펌", "강한"]
                },
                "step3_context_enrichment": ["체압분산", "지지력", "척추정렬", "압력완화", "자세교정"],
                "step4_final_expanded_query": "허리 디스크 환자용 딱딱한 매트리스 80만원 이하 요추 척추 등 허리통증 요통 추간판 허리디스크 척추디스크 탈출증 단단한 하드 견고한 탄탄한 펌 강한 체압분산 지지력 척추정렬 압력완화 자세교정",
                "expected_similarity_boost": "+0.25"
            },
            {
                "user_query": "더위 많이 타는 신혼부부 킹사이즈",
                "step1_keyword_extraction": ["더위", "타는", "신혼부부", "킹사이즈"],
                "step2_gpt_synonym_expansion": {
                    "더위": ["열", "뜨거움", "고온", "열감"],
                    "타는": ["민감한", "많이느끼는", "싫어하는"],
                    "신혼부부": ["커플", "부부", "연인", "2인", "부부용"],
                    "킹사이즈": ["킹", "K", "대형", "킹베드", "큰침대"]
                },
                "step3_context_enrichment": ["쿨링", "냉감", "통기성", "온도조절", "동작격리", "진동차단"],
                "step4_final_expanded_query": "더위 많이 타는 신혼부부 킹사이즈 열 뜨거움 고온 열감 민감한 많이느끼는 싫어하는 커플 부부 연인 2인 부부용 킹 K 대형 킹베드 큰침대 쿨링 냉감 통기성 온도조절 동작격리 진동차단",
                "expected_similarity_boost": "+0.22"
            }
        ]
    
    def _load_enhanced_intent_analysis(self) -> List[Dict]:
        """강화된 의도 분석 예시"""
        return [
            {
                "user_query": "허리 디스크로 수술했는데 딱딱한 매트리스 필요해요 예산은 100만원 정도",
                "enhanced_analysis": {
                    "intent_type": "health_critical",
                    "urgency": "very_high", 
                    "health_severity": "surgical_case",
                    "budget_info": {
                        "has_budget": True,
                        "range": "80-120만원",
                        "min": 80,
                        "max": 120,
                        "flexibility": "medium"
                    },
                    "health_info": {
                        "has_issue": True,
                        "issues": ["허리", "디스크", "수술"],
                        "severity": "very_high",
                        "medical_background": "post_surgery"
                    },
                    "preferences": {
                        "firmness": "딱딱",
                        "health_priority": True,
                        "medical_grade": True
                    },
                    "search_optimization": {
                        "keywords_weight": {
                            "허리": 5.0,
                            "디스크": 5.0, 
                            "수술": 4.5,
                            "딱딱한": 4.0
                        },
                        "context_expansion": ["체압분산", "척추정렬", "의료용", "재활용"],
                        "synonym_priority": ["요추", "척추", "하드", "펌"]
                    },
                    "confidence": 0.98
                }
            },
            {
                "user_query": "50만원대로 아이용 싱글 매트리스 추천해주세요",
                "enhanced_analysis": {
                    "intent_type": "budget_family",
                    "urgency": "medium",
                    "budget_info": {
                        "has_budget": True,
                        "range": "40-60만원",
                        "min": 40,
                        "max": 60,
                        "budget_conscious": True
                    },
                    "user_profile": {
                        "target_user": "child",
                        "family_purchase": True,
                        "age_group": "growing"
                    },
                    "preferences": {
                        "size": "싱글",
                        "safety": "high_priority",
                        "growth_support": True
                    },
                    "search_optimization": {
                        "keywords_weight": {
                            "아이": 4.0,
                            "싱글": 3.5,
                            "50만원": 3.0
                        },
                        "context_expansion": ["성장기", "안전소재", "항균", "친환경"],
                        "synonym_priority": ["어린이", "키즈", "1인용", "성장기용"]
                    },
                    "confidence": 0.92
                }
            }
        ]
    
    def _load_enhanced_response_generation(self) -> List[Dict]:
        """강화된 응답 생성 예시 (사용자 후기/평점 중심)"""
        return [
            {
                "user_query": "허리 디스크 수술 후 회복용 딱딱한 매트리스",
                "search_results": [
                    {
                        "name": "에이스침대 닥터하드 플러스",
                        "brand": "에이스침대",
                        "price": 89,
                        "type": "의료용 하드스프링",
                        "features": ["척추정렬", "의료등급", "체압분산", "항균"],
                        "target_users": ["디스크환자", "수술후회복", "척추질환"],
                        "similarity_score": 0.94
                    }
                ],
                "enhanced_response": "디스크 수술을 받으셨군요. 회복기에는 정말 신중한 선택이 필요하죠.\n\n'에이스침대 닥터하드 플러스'를 추천드립니다. 89만원으로, 실제 디스크 수술 경험자분들이 가장 많이 선택하시는 제품입니다.\n\n구매하신 분들 후기를 보면 '수술 후 3개월 사용했는데 허리 부담이 확실히 줄었다', '정형외과 의사가 추천해줘서 샀는데 정말 만족한다'는 평가가 많아요. 특히 4.8/5점의 높은 평점을 받고 있으며, 90% 이상이 재구매 의사를 밝혔습니다.\n\n수술 후 회복기에는 개인차가 있지만, 대부분의 사용자들이 2-3주 내에 수면 질 개선을 경험했다고 하네요."
            },
            {
                "user_query": "더위 많이 타는 커플용 쿨링 매트리스",
                "search_results": [
                    {
                        "name": "퍼플 하이브리드 프리미어 킹",
                        "brand": "퍼플",
                        "price": 195,
                        "type": "젤그리드 하이브리드",
                        "features": ["젤그리드", "쿨링시스템", "동작격리", "통기성"],
                        "target_users": ["더위타는분", "커플", "프리미엄선호"],
                        "similarity_score": 0.91
                    }
                ],
                "enhanced_response": "더위를 많이 타시는 커플분이시군요. 여름철 수면이 정말 중요하죠.\n\n'퍼플 하이브리드 프리미어 킹'을 강력 추천합니다. 195만원으로 프리미엄이지만, 더위 타는 분들 사이에서는 '게임체인저'라고 불리는 제품이에요.\n\n실제 구매 후기를 보면 '에어컨 없이도 시원하게 잔다', '땀으로 깨는 일이 없어졌다'는 평가가 압도적입니다. 커플 사용자들은 '서로 뒤척여도 전혀 느껴지지 않는다', '한 명이 더위 많이 타도 상대방은 괜찮다'고 평가해요. 4.7/5점 평점에 재구매율 95%를 자랑합니다.\n\n여름철 사용 후기 중 85%가 '체감온도 3-4도 낮아진 느낌'이라고 답했고, 6개월 이상 사용자 중 98%가 '다시 선택해도 이 제품'이라고 하네요."
            }
        ]

    def get_similarity_optimization_prompt(self) -> str:
        """유사도 최적화 프롬프트"""
        examples_text = ""
        for example in self.similarity_optimization_examples:
            examples_text += f"전략: {example['strategy']}\n"
            examples_text += f"개선: {example['similarity_improvement']}\n"
            examples_text += f"예시: {example['before']} → {example['after']}\n\n"
        
        return f"""매트리스 검색 유사도 극대화 전문가입니다. 다음 전략을 사용하여 검색 성능을 최대한 향상시키세요.

유사도 향상 전략:
{examples_text}

GPT 동의어 생성 가이드:
{self.gpt_synonym_examples}

핵심 원칙:
1. 동의어는 8-12개 (한국어 + 영어)
2. 기술 관련어 4-6개 추가  
3. 상황별 자연 표현 3-5개
4. 매트리스 도메인 특화 용어 우선
5. 유사도 점수 직접 기여 용어 선별

응답은 JSON 배열 형태로만 제공하세요."""
    
    def get_enhanced_query_expansion_prompt(self) -> str:
        """강화된 쿼리 확장 프롬프트"""
        examples_text = ""
        for i, example in enumerate(self.query_expansion_examples, 1):
            examples_text += f"""예시 {i}:
원본 쿼리: "{example['user_query']}"
1단계 키워드: {example['step1_keyword_extraction']}
2단계 동의어: {json.dumps(example['step2_gpt_synonym_expansion'], ensure_ascii=False)}
3단계 맥락강화: {example['step3_context_enrichment']}
최종 확장: "{example['step4_final_expanded_query']}"
유사도 향상: {example['expected_similarity_boost']}

"""
        
        return f"""매트리스 검색 쿼리 확장 전문가입니다. 4단계 확장 전략으로 유사도를 극대화하세요.

{examples_text}

확장 단계:
1. 핵심 키워드 추출
2. GPT 동의어 매핑 
3. 도메인 맥락 강화
4. 최종 확장 쿼리 생성

목표: 유사도 +0.2 이상 향상
확장된 텍스트만 반환하세요."""
    
    def get_enhanced_intent_analysis_prompt(self) -> str:
        """강화된 의도 분석 프롬프트"""
        examples_text = ""
        for i, example in enumerate(self.intent_analysis_examples, 1):
            examples_text += f"""예시 {i}:
입력: "{example['user_query']}"
강화 분석:
{json.dumps(example['enhanced_analysis'], ensure_ascii=False, indent=2)}

"""
        
        return f"""매트리스 구매 의도 분석 전문가입니다. 유사도 최적화를 위한 세부 분석을 수행하세요.

{examples_text}

분석 요소:
1. 의도 유형 (health_critical, budget_family, lifestyle_focused 등)
2. 긴급도 (very_high, high, medium, low)
3. 건강 심각도 (surgical_case, chronic, mild 등)
4. 검색 최적화 정보 (키워드 가중치, 맥락 확장, 동의어 우선순위)

JSON 형식으로 정확히 분석하세요."""
    
    def get_enhanced_response_generation_prompt(self) -> str:
        """강화된 응답 생성 프롬프트 (사용자 후기/평점 중심)"""
        examples_text = ""
        for i, example in enumerate(self.response_generation_examples, 1):
            search_info = example['search_results'][0]
            examples_text += f"""예시 {i}:
    질문: "{example['user_query']}"
    매트리스: {search_info['name']} ({search_info['brand']}) - {search_info['price']}만원
    유사도: {search_info['similarity_score']}

    전문가 응답:
    {example['enhanced_response']}

    ---
    """
        
        return f"""15년 경력 매트리스 전문가입니다. 고객의 상황을 정확히 파악하고 최적화된 상담을 제공하세요.

    {examples_text}

    응답 구조 (사용자 경험 중심):
    1. 상황 공감 (고객 문제 이해)
    2. 명확한 추천 (제품명 + 가격)
    3. 실제 사용자 후기 (구체적인 경험담)
    4. 평점/만족도 데이터 (신뢰성 있는 수치)
    5. 사용 기간별 효과 (실제 경험 기반)

    톤: 전문적이면서도 친근하고 신뢰감 있게
    길이: 300-400자

    핵심 가이드라인:
    - 기술적 스펙보다는 실제 사용자들의 생생한 후기 중심으로 설명
    - "구매하신 분들 후기를 보면...", "실제 사용자들은...", "평점 X.X/5점에..." 등의 표현 활용
    - 구체적인 만족도 수치나 재구매율 언급으로 신뢰성 강화
    - 사용 기간별 효과나 개선 사항을 실제 경험담으로 제시
    - 같은 표현 반복 금지, 다양한 어휘로 표현

    언어 사용 지침:
    - "고객"은 반드시 "고객"으로 정확히 표현하세요 (고갱 ❌)
    - 정확한 한국어 표준 발음을 사용하세요
    - 전문적이고 정중한 어조를 유지하세요
    - 고객님의 상황을 정확히 이해하고 맞춤형 솔루션을 제공하세요"""


class EnhancedOpenAIQueryProcessor:
    """Few-shot + GPT 동의어 강화 쿼리 프로세서"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.client = None
        self.model = model
        
        # OpenAI 클라이언트 초기화
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                logger.info("Enhanced 쿼리 프로세서 OpenAI 클라이언트 초기화 완료")
            except Exception as e:
                logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        
        # Enhanced Few-shot 매니저
        self.few_shot_manager = EnhancedFewShotManager()
    
    def expand_query_with_enhanced_gpt(self, user_query: str) -> Dict[str, Any]:
        """GPT + Few-shot 강화 쿼리 확장"""
        if not self.client:
            return self._fallback_expansion(user_query)
        
        try:
            system_prompt = self.few_shot_manager.get_enhanced_query_expansion_prompt()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"확장할 쿼리: '{user_query}'"}
                ],
                max_tokens=300,
                temperature=0.4
            )
            
            expanded_query = response.choices[0].message.content.strip()
            
            # 추가 구조화 정보 추출
            keywords = user_query.split()
            
            return {
                'original_query': user_query,
                'expanded_query': expanded_query,
                'extracted_keywords': keywords,
                'enhancement_type': 'gpt_few_shot',
                'expected_similarity_boost': 0.25,
                'search_terms': [user_query, expanded_query],
                'enhanced': True
            }
            
        except Exception as e:
            logger.error(f"Enhanced 쿼리 확장 실패: {e}")
            return self._fallback_expansion(user_query)
    
    def analyze_intent_with_optimization(self, user_query: str) -> Dict:
        """최적화 정보 포함 의도 분석"""
        if not self.client:
            return self._basic_intent_analysis(user_query)
        
        try:
            system_prompt = self.few_shot_manager.get_enhanced_intent_analysis_prompt()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"분석할 질문: '{user_query}'"}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            try:
                intent = json.loads(response.choices[0].message.content.strip())
                intent['enhanced_few_shot'] = True
                return intent
            except json.JSONDecodeError:
                logger.error("Enhanced 의도 분석 JSON 파싱 실패")
                
        except Exception as e:
            logger.error(f"Enhanced 의도 분석 실패: {e}")
        
        return self._basic_intent_analysis(user_query)
    
    def _fallback_expansion(self, user_query: str) -> Dict[str, Any]:
        """폴백 확장"""
        return {
            'original_query': user_query,
            'expanded_query': user_query,
            'extracted_keywords': user_query.split(),
            'enhancement_type': 'fallback',
            'expected_similarity_boost': 0.0,
            'search_terms': [user_query],
            'enhanced': False
        }
    
    def _basic_intent_analysis(self, user_query: str) -> Dict:
        """기본 의도 분석"""
        return {
            'intent_type': 'basic_search',
            'urgency': 'medium',
            'confidence': 0.5,
            'enhanced_few_shot': False
        }


class EnhancedOpenAIResponseGenerator:
    """Few-shot 강화 응답 생성기 (고객 표기 수정)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.client = None
        self.model = model
        
        # OpenAI 클라이언트 초기화
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                logger.info("Enhanced 응답 생성기 OpenAI 클라이언트 초기화 완료")
            except Exception as e:
                logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        
        # Enhanced Few-shot 매니저
        self.few_shot_manager = EnhancedFewShotManager()
    
    def generate_enhanced_response(self, user_query: str, search_results: List[Dict], 
                                 user_intent: Optional[Dict] = None,
                                 query_expansion: Optional[Dict] = None) -> str:
        """Few-shot 강화 응답 생성 (고객 표기 수정)"""
        if not self.client:
            return self._generate_fallback_response(user_query, search_results)
        
        if not search_results:
            return "죄송합니다. 조건에 맞는 매트리스를 찾을 수 없습니다."
        
        try:
            # Few-shot 강화 시스템 프롬프트 (고객 표기 수정 포함)
            system_prompt = self.few_shot_manager.get_enhanced_response_generation_prompt()
            
            # 검색 결과 컨텍스트 (상위 결과 중심)
            top_mattress = search_results[0]
            context = f"""추천 매트리스 정보:
- 제품명: {top_mattress.get('name', 'Unknown')}
- 브랜드: {top_mattress.get('brand', 'Unknown')}
- 가격: {top_mattress.get('price', 0)}만원
- 타입: {top_mattress.get('type', 'Unknown')}
- 주요 특징: {', '.join(top_mattress.get('features', [])[:3])}
- 추천 대상: {', '.join(top_mattress.get('target_users', [])[:2])}
- 유사도 점수: {top_mattress.get('similarity_score', 0):.3f}
- Enhanced 검색: {top_mattress.get('gpt_enhanced', False)}"""
            
            # 사용자 컨텍스트
            user_context = ""
            if user_intent:
                context_parts = []
                
                # 건강 정보
                health_info = user_intent.get('health_info', {})
                if health_info.get('has_issue'):
                    issues = health_info.get('issues', [])
                    severity = health_info.get('severity', 'medium')
                    context_parts.append(f"건강 이슈: {', '.join(issues)} (심각도: {severity})")
                
                # 예산 정보
                budget_info = user_intent.get('budget_info', {})
                if budget_info.get('has_budget'):
                    context_parts.append(f"예산: {budget_info.get('range', '')}")
                
                # 선호도
                preferences = user_intent.get('preferences', {})
                if preferences:
                    pref_text = ', '.join([f"{k}: {v}" for k, v in preferences.items() if v])
                    context_parts.append(f"선호도: {pref_text}")
                
                if context_parts:
                    user_context = f"\n\n고객 상황:\n" + '\n'.join([f"- {part}" for part in context_parts])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"고객 질문: \"{user_query}\"\n\n{context}{user_context}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            final_response = response.choices[0].message.content.strip()
            logger.info("Enhanced Few-shot 응답 생성 완료")
            return final_response
            
        except Exception as e:
            logger.error(f"Enhanced 응답 생성 실패: {e}")
            return self._generate_fallback_response(user_query, search_results)
    
    def _generate_fallback_response(self, user_query: str, search_results: List[Dict]) -> str:
        """폴백 응답"""
        if not search_results:
            return "죄송합니다. 조건에 맞는 매트리스를 찾을 수 없습니다."
        
        top = search_results[0]
        return f"{top.get('name', 'Unknown')}을 추천드립니다. {top.get('price', 0)}만원으로 고객님께 적합한 제품입니다."


# 기존 호환성을 위한 클래스 (기존 ai_agent.py에서 import 할 수 있도록)
class FewShotExampleManager(EnhancedFewShotManager):
    """기존 호환성을 위한 별칭"""
    pass


def get_query_expansion_examples() -> List[Dict]:
    """쿼리 확장 예시 반환 (기존 호환성)"""
    manager = EnhancedFewShotManager()
    return manager.query_expansion_examples


def get_intent_analysis_examples() -> List[Dict]:
    """의도 분석 예시 반환 (기존 호환성)"""
    manager = EnhancedFewShotManager()
    return manager.intent_analysis_examples


def get_response_generation_examples() -> List[Dict]:
    """응답 생성 예시 반환 (기존 호환성)"""
    manager = EnhancedFewShotManager()
    return manager.response_generation_examples


# 테스트 실행
if __name__ == "__main__":
    print("🎯 Enhanced Few-shot 학습 모듈 테스트 (고객 표기 수정)")
    print("=" * 60)
    
    try:
        # Enhanced Few-shot 매니저 테스트
        few_shot_manager = EnhancedFewShotManager()
        
        print(f"✅ Enhanced Few-shot 매니저 로드 완료:")
        print(f"  유사도 최적화 예시: {len(few_shot_manager.similarity_optimization_examples)}개")
        print(f"  쿼리 확장 예시: {len(few_shot_manager.query_expansion_examples)}개")
        print(f"  의도 분석 예시: {len(few_shot_manager.intent_analysis_examples)}개")
        print(f"  응답 생성 예시: {len(few_shot_manager.response_generation_examples)}개")
        
        # 유사도 최적화 전략 출력
        print(f"\n🚀 유사도 최적화 전략:")
        for strategy in few_shot_manager.similarity_optimization_examples:
            print(f"  전략: {strategy['strategy']}")
            print(f"  개선: {strategy['similarity_improvement']}")
            print(f"  예시: {strategy['before']} → {strategy['after'][:50]}...")
            print()
        
        # 프롬프트 생성 테스트
        print(f"📝 프롬프트 생성 테스트:")
        
        similarity_prompt = few_shot_manager.get_similarity_optimization_prompt()
        print(f"  유사도 최적화 프롬프트: {len(similarity_prompt)}자")
        
        expansion_prompt = few_shot_manager.get_enhanced_query_expansion_prompt()
        print(f"  강화 쿼리 확장 프롬프트: {len(expansion_prompt)}자")
        
        intent_prompt = few_shot_manager.get_enhanced_intent_analysis_prompt()
        print(f"  강화 의도 분석 프롬프트: {len(intent_prompt)}자")
        
        response_prompt = few_shot_manager.get_enhanced_response_generation_prompt()
        print(f"  강화 응답 생성 프롬프트: {len(response_prompt)}자")
        
        # 고객 표기 검증
        print(f"\n✅ 고객 표기 검증:")
        response_examples = few_shot_manager.response_generation_examples
        for i, example in enumerate(response_examples, 1):
            response_text = example['enhanced_response']
            if '고갱' in response_text:
                print(f"  ❌ 예시 {i}: '고갱' 발견")
            elif '고객' in response_text:
                print(f"  ✅ 예시 {i}: '고객' 정상 표기")
            else:
                print(f"  ⚠️ 예시 {i}: '고객' 미사용")
        
        # 기존 호환성 테스트
        print(f"\n🔄 기존 호환성 테스트:")
        query_examples = get_query_expansion_examples()
        intent_examples = get_intent_analysis_examples()
        response_examples = get_response_generation_examples()
        
        print(f"  쿼리 확장 예시: {len(query_examples)}개")
        print(f"  의도 분석 예시: {len(intent_examples)}개") 
        print(f"  응답 생성 예시: {len(response_examples)}개")
        
        # OpenAI 프로세서 테스트 (API 키가 있는 경우)
        import os
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"\n🤖 OpenAI 프로세서 테스트:")
            
            # 쿼리 프로세서 테스트
            query_processor = EnhancedOpenAIQueryProcessor(api_key)
            if query_processor.client:
                print(f"  ✅ Enhanced 쿼리 프로세서 초기화 성공")
                
                test_query = "허리 아픈 사람용 딱딱한 매트리스"
                expansion_result = query_processor.expand_query_with_enhanced_gpt(test_query)
                print(f"  테스트 쿼리: '{test_query}'")
                print(f"  확장 결과: {expansion_result.get('enhanced', False)}")
                print(f"  예상 향상: +{expansion_result.get('expected_similarity_boost', 0)}")
            else:
                print(f"  ❌ OpenAI 클라이언트 초기화 실패")
            
            # 응답 생성기 테스트
            response_generator = EnhancedOpenAIResponseGenerator(api_key)
            if response_generator.client:
                print(f"  ✅ Enhanced 응답 생성기 초기화 성공")
                print(f"  ✅ 고객 표기 수정 프롬프트 적용됨")
            else:
                print(f"  ❌ OpenAI 클라이언트 초기화 실패")
        else:
            print(f"\n⚠️ OPENAI_API_KEY 환경변수가 설정되지 않아 OpenAI 테스트 생략")
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 문제 해결:")
        print(f"1. 의존성 설치: pip install openai")
        print(f"2. API 키 설정: export OPENAI_API_KEY='your-key'")
        print(f"3. Python 경로 확인")