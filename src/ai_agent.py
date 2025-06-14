"""
AI 에이전트 - OpenAI 텍스트 생성 및 ReAct 패턴
파일: src/ai_agent.py

역할:
- 사용자 쿼리 분석 및 개선
- RAG 결과 기반 응답 생성
- ReAct 패턴 구현 (추론-행동-관찰)
- 대화 관리 및 컨텍스트 유지
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import openai
from openai import OpenAI
from datetime import datetime
import time
import re

# 프로젝트 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.rag_system import MattressRAGSystem, setup_rag_system

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAITextManager:
    """OpenAI 텍스트 생성 전용 매니저"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        OpenAI 텍스트 매니저 초기화
        
        Args:
            api_key: OpenAI API 키 (없으면 환경변수에서 가져옴)
        """
        # API 키 설정
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                logger.warning(
                    "OpenAI API 키가 없습니다. "
                    "텍스트 개선 및 응답 생성 기능을 사용할 수 없습니다."
                )
                self.client = None
                return
        
        # OpenAI 클라이언트 초기화
        try:
            self.client = OpenAI(api_key=self.api_key)
            # 간단한 연결 테스트
            test_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "테스트"}],
                max_tokens=5
            )
            logger.info("OpenAI 텍스트 매니저 초기화 완료")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.client = None
    
    def enhance_query(self, user_query: str) -> str:
        """
        사용자 쿼리를 OpenAI로 개선
        
        Args:
            user_query: 원본 사용자 쿼리
            
        Returns:
            str: 개선된 쿼리
        """
        if not self.client:
            logger.warning("OpenAI 클라이언트가 없어 쿼리 개선을 건너뜁니다")
            return user_query
        
        try:
            system_prompt = """
당신은 매트리스 검색 전문가입니다. 사용자의 질문을 분석하여 매트리스 검색에 최적화된 키워드와 문장으로 변환해주세요.

사용자 질문에서 다음 요소들을 추출하고 강화하세요:
- 건강 문제 (허리통증, 목통증, 관절염 등)
- 수면 자세 (옆으로, 등으로, 엎드려 등)
- 선호도 (딱딱함, 부드러움, 시원함 등)  
- 예산 범위
- 특별 요구사항 (커플용, 어린이용 등)
- 브랜드 선호도

원본 질문의 의도를 유지하면서 검색에 유리한 형태로 변환하되, 한국어로 답변해주세요.
간결하고 핵심적인 키워드 중심으로 작성해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"다음 질문을 매트리스 검색용으로 개선해주세요: {user_query}"}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            enhanced_query = response.choices[0].message.content.strip()
            logger.info(f"쿼리 개선: '{user_query}' → '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"쿼리 개선 실패: {e}")
            return user_query  # 실패 시 원본 반환
    
    def analyze_user_intent(self, user_query: str) -> Dict:
        """
        사용자 의도 분석
        
        Args:
            user_query: 사용자 질문
            
        Returns:
            Dict: 분석된 사용자 의도
        """
        if not self.client:
            # 기본 의도 분석 (규칙 기반)
            intent = {
                'intent_type': 'search',
                'budget_range': None,
                'health_issues': [],
                'preferences': [],
                'confidence': 0.5
            }
            
            # 간단한 키워드 매칭
            query_lower = user_query.lower()
            if any(word in query_lower for word in ['허리', '목', '관절']):
                intent['health_issues'].append('통증')
            if any(word in query_lower for word in ['예산', '만원', '원']):
                intent['intent_type'] = 'budget_search'
            
            return intent
        
        try:
            system_prompt = """
사용자의 매트리스 관련 질문을 분석하여 다음 JSON 형태로 의도를 파악해주세요:

{
  "intent_type": "search|compare|recommend|info|budget_search",
  "budget_range": "예산 범위 (예: 50-100만원)",
  "health_issues": ["허리통증", "목통증", "관절염" 등],
  "sleep_position": "옆으로|등으로|엎드려|혼합",
  "preferences": ["딱딱함", "부드러움", "시원함", "따뜻함" 등],
  "size_preference": "싱글|퀸|킹|슈퍼킹",
  "special_requirements": ["커플용", "어린이용", "노인용" 등],
  "urgency": "high|medium|low",
  "confidence": 0.0-1.0
}

정확하고 간결하게 분석해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"다음 질문을 분석해주세요: {user_query}"}
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            # JSON 파싱 시도
            try:
                intent = json.loads(response.choices[0].message.content.strip())
                logger.info(f"사용자 의도 분석 완료: {intent.get('intent_type', 'unknown')}")
                return intent
            except json.JSONDecodeError:
                logger.error("의도 분석 결과 JSON 파싱 실패")
                return {'intent_type': 'search', 'confidence': 0.3}
                
        except Exception as e:
            logger.error(f"사용자 의도 분석 실패: {e}")
            return {'intent_type': 'search', 'confidence': 0.1}
    
    def generate_response(self, user_query: str, search_results: List[Dict], 
                         user_intent: Optional[Dict] = None) -> str:
        """
        검색 결과를 바탕으로 상황에 맞는 응답 생성
        
        Args:
            user_query: 사용자 원본 질문
            search_results: RAG 검색 결과
            user_intent: 분석된 사용자 의도
            
        Returns:
            str: 상황에 맞는 응답
        """
        if not self.client:
            # OpenAI 없이 기본 응답 생성
            return self._generate_fallback_response(user_query, search_results)
        
        if not search_results:
            return "죄송합니다. 현재 조건에 맞는 매트리스를 찾을 수 없습니다. 다른 조건으로 검색해보시겠어요?"
        
        try:
            # 컨텍스트 구성 (상위 3개 결과만 사용)
            context_parts = []
            for mattress in search_results[:3]:
                features_text = ', '.join(mattress['features'][:3]) if mattress['features'] else '정보 없음'
                target_users_text = ', '.join(mattress['target_users'][:2]) if mattress['target_users'] else ''
                
                context_parts.append(
                    f"- {mattress['name']} ({mattress['brand']}): "
                    f"{mattress['type']} 타입, {mattress['price']}만원, "
                    f"특징: {features_text}"
                    f"{', 추천: ' + target_users_text if target_users_text else ''}"
                )
            
            context = "\n".join(context_parts)
            
            # 사용자 의도 정보 추가
            intent_info = ""
            if user_intent:
                intent_parts = []
                if user_intent.get('health_issues'):
                    intent_parts.append(f"건강 이슈: {', '.join(user_intent['health_issues'])}")
                if user_intent.get('budget_range'):
                    intent_parts.append(f"예산: {user_intent['budget_range']}")
                if user_intent.get('preferences'):
                    intent_parts.append(f"선호도: {', '.join(user_intent['preferences'])}")
                
                if intent_parts:
                    intent_info = f"\n\n고객 정보: {' | '.join(intent_parts)}"
            
            system_prompt = f"""
당신은 10년 경력의 매트리스 전문 상담사입니다. 고객의 질문에 대해 검색된 매트리스 정보를 바탕으로 전문적이고 친근한 상담을 제공해주세요.

검색된 매트리스 정보:
{context}{intent_info}

상담 가이드라인:
1. 고객의 요구사항을 정확히 파악했음을 보여주세요
2. 가장 적합한 매트리스 1-2개를 구체적으로 추천하세요
3. 추천하는 이유를 명확히 설명하세요
4. 가격 대비 효과를 언급하세요
5. 추가 고려사항이나 팁을 제공하세요
6. 친근하고 전문적인 톤을 유지하세요
7. 한국어로 답변하세요
8. 300-400자 내외로 답변하세요
9. 필요시 추가 질문을 유도하세요
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            # 폴백 응답
            return self._generate_fallback_response(user_query, search_results)
    
    def _generate_fallback_response(self, user_query: str, search_results: List[Dict]) -> str:
        """OpenAI 없이 기본 응답 생성"""
        if not search_results:
            return "죄송합니다. 현재 조건에 맞는 매트리스를 찾을 수 없습니다."
        
        top_mattress = search_results[0]
        features_text = ', '.join(top_mattress['features'][:2]) if top_mattress['features'] else ''
        
        response = f"검색 결과, '{top_mattress['name']}'을(를) 추천드립니다. "
        response += f"{top_mattress['brand']} 브랜드의 {top_mattress['type']} 타입으로 "
        response += f"{top_mattress['price']}만원입니다."
        
        if features_text:
            response += f" 주요 특징으로는 {features_text} 등이 있습니다."
        
        if len(search_results) > 1:
            response += f" 다른 옵션도 {len(search_results)-1}개 더 있으니 참고해보세요."
        
        return response

class ConversationManager:
    """대화 관리 클래스"""
    
    def __init__(self):
        """대화 관리자 초기화"""
        self.conversation_history = []
        self.user_context = {}
        self.session_start = datetime.now()
    
    def add_interaction(self, user_query: str, agent_response: str, 
                       search_results: Optional[List[Dict]] = None,
                       user_intent: Optional[Dict] = None):
        """대화 기록 추가"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'agent_response': agent_response,
            'search_results_count': len(search_results) if search_results else 0,
            'user_intent': user_intent
        }
        
        self.conversation_history.append(interaction)
        
        # 사용자 컨텍스트 업데이트
        if user_intent:
            self._update_user_context(user_intent)
    
    def _update_user_context(self, user_intent: Dict):
        """사용자 컨텍스트 업데이트"""
        if 'budget_range' in user_intent and user_intent['budget_range']:
            self.user_context['budget_range'] = user_intent['budget_range']
        
        if 'health_issues' in user_intent and user_intent['health_issues']:
            if 'health_issues' not in self.user_context:
                self.user_context['health_issues'] = set()
            self.user_context['health_issues'].update(user_intent['health_issues'])
        
        if 'preferences' in user_intent and user_intent['preferences']:
            if 'preferences' not in self.user_context:
                self.user_context['preferences'] = set()
            self.user_context['preferences'].update(user_intent['preferences'])
    
    def get_conversation_summary(self) -> Dict:
        """대화 요약 정보 반환"""
        return {
            'session_duration': str(datetime.now() - self.session_start),
            'total_interactions': len(self.conversation_history),
            'user_context': {k: list(v) if isinstance(v, set) else v 
                           for k, v in self.user_context.items()},
            'last_query': self.conversation_history[-1]['user_query'] if self.conversation_history else None
        }

class MattressAIAgent:
    """매트리스 AI 에이전트 - ReAct 패턴 구현"""
    
    def __init__(self, api_key: Optional[str] = None, data_path: Optional[str] = None):
        """
        매트리스 AI 에이전트 초기화
        
        Args:
            api_key: OpenAI API 키
            data_path: 매트리스 데이터 파일 경로
        """
        # RAG 시스템 초기화
        self.rag_system, rag_success = setup_rag_system(data_path)
        if not rag_success:
            raise RuntimeError("RAG 시스템 초기화 실패")
        
        # 텍스트 생성 매니저 초기화
        self.text_manager = OpenAITextManager(api_key)
        
        # 대화 관리자 초기화
        self.conversation_manager = ConversationManager()
        
        # 시스템 상태
        self.is_ready = True
        
        logger.info("매트리스 AI 에이전트 초기화 완료")
        logger.info(f"RAG 시스템: {'✅' if rag_success else '❌'}")
        logger.info(f"OpenAI 텍스트: {'✅' if self.text_manager.client else '❌'}")
    
    def process_query(self, user_query: str, n_results: int = 5) -> Dict:
        """
        ReAct 패턴으로 사용자 쿼리 처리
        
        Args:
            user_query: 사용자 질문
            n_results: 검색할 결과 수
            
        Returns:
            Dict: 처리 결과
        """
        if not self.is_ready:
            return {
                'error': 'AI 에이전트가 준비되지 않았습니다',
                'user_query': user_query,
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"AI 에이전트 쿼리 처리 시작: '{user_query}'")
        start_time = time.time()
        
        try:
            # Step 1: Thought - 사용자 의도 분석
            logger.info("Step 1: 사용자 의도 분석")
            user_intent = self.text_manager.analyze_user_intent(user_query)
            
            # Step 2: Action - 쿼리 개선 및 검색 실행
            logger.info("Step 2: 쿼리 개선 및 검색")
            enhanced_query = self.text_manager.enhance_query(user_query)
            search_results = self.rag_system.search_mattresses(enhanced_query, n_results)
            
            # Step 3: Observation - 검색 결과 분석
            logger.info("Step 3: 검색 결과 분석")
            if not search_results:
                logger.warning("검색 결과가 없습니다")
            
            # Step 4: Thought - 응답 전략 결정 및 생성
            logger.info("Step 4: 응답 생성")
            agent_response = self.text_manager.generate_response(
                user_query, search_results, user_intent
            )
            
            # 대화 기록 저장
            self.conversation_manager.add_interaction(
                user_query, agent_response, search_results, user_intent
            )
            
            end_time = time.time()
            
            result = {
                'user_query': user_query,
                'enhanced_query': enhanced_query,
                'user_intent': user_intent,
                'search_results': search_results,
                'agent_response': agent_response,
                'total_results': len(search_results),
                'processing_time': round(end_time - start_time, 2),
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info(f"AI 에이전트 처리 완료 ({result['processing_time']}초)")
            return result
            
        except Exception as e:
            logger.error(f"AI 에이전트 처리 실패: {e}")
            return {
                'error': str(e),
                'user_query': user_query,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_recommendations(self, criteria: Dict) -> List[Dict]:
        """
        특정 조건에 맞는 매트리스 추천
        
        Args:
            criteria: 검색 조건 딕셔너리
            
        Returns:
            List[Dict]: 추천 매트리스 목록
        """
        # 조건을 자연어 쿼리로 변환
        query_parts = []
        
        if criteria.get('health_issues'):
            query_parts.append(f"건강 문제: {', '.join(criteria['health_issues'])}")
        
        if criteria.get('budget_max'):
            query_parts.append(f"예산 {criteria['budget_max']}만원 이하")
        
        if criteria.get('preferences'):
            query_parts.append(f"선호도: {', '.join(criteria['preferences'])}")
        
        if criteria.get('size'):
            query_parts.append(f"{criteria['size']} 사이즈")
        
        search_query = ' '.join(query_parts) if query_parts else "좋은 매트리스 추천"
        
        # 검색 실행
        results = self.rag_system.search_mattresses(search_query, n_results=10)
        
        # 조건에 맞는 필터링
        filtered_results = []
        for mattress in results:
            # 예산 필터
            if criteria.get('budget_max') and mattress['price'] > criteria['budget_max']:
                continue
            
            # 예산 최소값 필터
            if criteria.get('budget_min') and mattress['price'] < criteria['budget_min']:
                continue
            
            filtered_results.append(mattress)
        
        return filtered_results[:5]  # 상위 5개만 반환
    
    def compare_mattresses(self, mattress_ids: List[str]) -> Dict:
        """매트리스 비교 분석"""
        mattresses = []
        for mattress_id in mattress_ids:
            mattress = self.rag_system.get_mattress_by_id(mattress_id)
            if mattress:
                mattresses.append(mattress)
        
        if not mattresses:
            return {'error': '비교할 매트리스를 찾을 수 없습니다'}
        
        # 비교 테이블 생성
        comparison = {
            'mattresses': mattresses,
            'comparison_table': {},
            'summary': {}
        }
        
        # 주요 속성별 비교
        attributes = ['price', 'type', 'brand', 'firmness']
        for attr in attributes:
            comparison['comparison_table'][attr] = [
                mattress.get(attr, '정보 없음') for mattress in mattresses
            ]
        
        return comparison
    
    def get_agent_status(self) -> Dict:
        """에이전트 상태 정보 반환"""
        rag_status = self.rag_system.get_system_status()
        conversation_summary = self.conversation_manager.get_conversation_summary()
        
        return {
            'ready': self.is_ready,
            'rag_system': rag_status,
            'openai_available': self.text_manager.client is not None,
            'conversation': conversation_summary,
            'session_start': self.conversation_manager.session_start.isoformat()
        }

# 편의 함수
def create_mattress_agent(api_key: Optional[str] = None, data_path: Optional[str] = None) -> MattressAIAgent:
    """
    매트리스 AI 에이전트 생성 편의 함수
    
    Args:
        api_key: OpenAI API 키
        data_path: 매트리스 데이터 파일 경로
        
    Returns:
        MattressAIAgent: 초기화된 AI 에이전트
    """
    try:
        agent = MattressAIAgent(api_key, data_path)
        logger.info("✅ 매트리스 AI 에이전트 생성 완료")
        return agent
    except Exception as e:
        logger.error(f"AI 에이전트 생성 실패: {e}")
        raise

# 테스트 실행
if __name__ == "__main__":
    print("🤖 매트리스 AI 에이전트 테스트")
    print("=" * 50)
    
    try:
        # AI 에이전트 생성
        api_key = os.getenv('OPENAI_API_KEY')  # 선택사항
        agent = create_mattress_agent(api_key)
        
        # 에이전트 상태 확인
        status = agent.get_agent_status()
        print(f"\n📊 에이전트 상태:")
        print(f"  준비 상태: {status['ready']}")
        print(f"  RAG 시스템: {status['rag_system']['initialized']}")
        print(f"  OpenAI 사용: {status['openai_available']}")
        print(f"  저장된 매트리스: {status['rag_system']['chroma_collection'].get('count', 0)}개")
        
        # ReAct 패턴 테스트
        test_queries = [
            "허리 통증이 심해서 딱딱한 매트리스 찾고 있어요. 예산은 80만원 정도입니다.",
            "더위를 많이 타는 편이라 시원한 매트리스가 필요해요.",
            "신혼부부용으로 킹사이즈 매트리스 추천해주세요.",
            "100만원 이하로 좋은 브랜드 매트리스 있나요?"
        ]
        
        print(f"\n🧪 ReAct 패턴 테스트:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"테스트 {i}: '{query}'")
            print(f"{'='*60}")
            
            result = agent.process_query(query, n_results=3)
            
            if result.get('success'):
                print(f"🎯 사용자 의도: {result['user_intent'].get('intent_type', 'unknown')}")
                print(f"🔍 개선된 쿼리: '{result['enhanced_query']}'")
                print(f"📊 검색 결과: {result['total_results']}개")
                print(f"⏱️ 처리 시간: {result['processing_time']}초")
                
                if result['search_results']:
                    top_result = result['search_results'][0]
                    print(f"🥇 1순위: {top_result['name']} (유사도: {top_result['similarity_score']:.3f})")
                
                print(f"\n🤖 AI 응답:")
                print(f"   {result['agent_response']}")
                
            else:
                print(f"❌ 처리 실패: {result.get('error', '알 수 없는 오류')}")
            
            # API 호출 제한 방지
            if i < len(test_queries):
                time.sleep(1)
        
        # 대화 요약
        conversation_summary = agent.conversation_manager.get_conversation_summary()
        print(f"\n📈 대화 요약:")
        print(f"  총 상호작용: {conversation_summary['total_interactions']}회")
        print(f"  세션 시간: {conversation_summary['session_duration']}")
        print(f"  사용자 컨텍스트: {conversation_summary['user_context']}")
        
        print(f"\n✅ AI 에이전트 테스트 완료!")
        print(f"🚀 구현된 기능:")
        print(f"   ✅ ReAct 패턴 (Reason-Act-Observe)")
        print(f"   ✅ 사용자 의도 분석")
        print(f"   ✅ 쿼리 개선")
        print(f"   ✅ RAG 검색")
        print(f"   ✅ 상황별 응답 생성")
        print(f"   ✅ 대화 관리")
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 문제 해결:")
        print(f"1. RAG 시스템: pip install sentence-transformers torch")
        print(f"2. OpenAI API: export OPENAI_API_KEY='your-key' (선택사항)")
        print(f"3. 데이터 파일: data/mattress_data.json 확인")