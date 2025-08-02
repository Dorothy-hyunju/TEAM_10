"""
매트리스 쇼핑 가이드 AI Agent - 메인 실행 파일
파일: main.py

역할:
- 전체 시스템 통합 실행
- 사용자 인터페이스 제공
- 시스템 상태 모니터링
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
import argparse
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 프로젝트 모듈 임포트
from src.ai_agent import create_mattress_agent, MattressAIAgent
from src.rag_system import setup_korean_rag_system

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mattress_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MattressAISystem:
    """매트리스 AI 시스템 메인 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        매트리스 AI 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.agent = None
        self.is_initialized = False
        
        print("🛏️ 매트리스 쇼핑 가이드 AI Agent")
        print("=" * 50)
        print("📍 프로젝트: TEAM_10")
        print("🧠 기술: OpenAI + RAG + 한국어 특화")
        print("⚡ 패턴: ReAct (Reason-Act-Observe)")
        print("=" * 50)
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """설정 파일 로드"""
        default_config = {
            "data_path": "./data/mattress_data.json",
            "openai_model": "gpt-3.5-turbo",
            "max_results": 5,
            "reset_db": False,
            "korean_model": "jhgan/ko-sroberta-multitask",
            "log_level": "INFO"
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"설정 파일 로드: {config_path}")
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}, 기본 설정 사용")
        
        return default_config
    
    def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            logger.info("매트리스 AI 시스템 초기화 시작")
            
            # OpenAI API 키 확인
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API 키가 설정되지 않았습니다. 기본 기능만 사용 가능합니다.")
                print("⚠️ OpenAI API 키가 없습니다. 다음 명령으로 설정하세요:")
                print("   export OPENAI_API_KEY='your-api-key'")
                print("   또는 .env 파일에 저장하세요")
                
                user_input = input("\n계속 진행하시겠습니까? (y/n): ").lower()
                if user_input != 'y':
                    return False
            
            # 데이터 파일 확인
            data_path = self.config['data_path']
            if not Path(data_path).exists():
                logger.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
                print(f"❌ 데이터 파일 없음: {data_path}")
                print("💡 generate_data.py를 먼저 실행하여 데이터를 생성하세요")
                return False
            
            # AI 에이전트 초기화
            self.agent = create_mattress_agent(
                api_key=api_key,
                data_path=data_path,
                model=self.config['openai_model']
            )
            
            # 시스템 상태 확인
            status = self.agent.get_agent_status()
            
            print(f"\n✅ 시스템 초기화 완료!")
            print(f"📊 시스템 상태:")
            print(f"   RAG 시스템: {'✅' if status['rag_system']['initialized'] else '❌'}")
            print(f"   OpenAI 연동: {'✅' if status['openai_available'] else '❌'}")
            print(f"   한국어 최적화: {'✅' if status['rag_system']['korean_optimized'] else '❌'}")
            print(f"   저장된 매트리스: {status['rag_system']['chroma_collection'].get('count', 0)}개")
            print(f"   사용 모델: {status.get('openai_model', 'N/A')}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            print(f"❌ 초기화 실패: {e}")
            return False
    
    def run_interactive_mode(self):
        """대화형 모드 실행"""
        if not self.is_initialized:
            print("❌ 시스템이 초기화되지 않았습니다")
            return
        
        print(f"\n🤖 매트리스 AI 상담사가 준비되었습니다!")
        print("💬 궁금한 매트리스에 대해 무엇이든 물어보세요")
        print("📝 명령어: 'quit' (종료), 'status' (상태), 'help' (도움말)")
        print("🎯 개인화 추천: 'recommend' (맞춤 추천)")
        print("=" * 60)
        
        interaction_count = 0
        
        while True:
            try:
                # 사용자 입력
                user_input = input(f"\n[{interaction_count + 1}] 질문: ").strip()
                
                if not user_input:
                    continue
                
                # 명령어 처리
                if user_input.lower() == 'quit':
                    self._show_session_summary()
                    print("👋 매트리스 쇼핑에 도움이 되었기를 바랍니다!")
                    break
                
                elif user_input.lower() == 'status':
                    self._show_system_status()
                    continue
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'recommend':
                    self._show_personalized_recommendations()
                    continue
                
                # 쿼리 처리
                print("🔍 검색 중...")
                start_time = datetime.now()
                
                result = self.agent.process_query(
                    user_input, 
                    n_results=self.config['max_results']
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                if result.get('success'):
                    interaction_count += 1
                    
                    # 결과 표시
                    print(f"\n🤖 AI 상담사 답변:")
                    print("─" * 50)
                    print(result['agent_response'])
                    print("─" * 50)
                    
                    # 검색 결과 요약
                    if result['search_results']:
                        print(f"\n📋 추천 매트리스 목록:")
                        for i, mattress in enumerate(result['search_results'][:3], 1):
                            print(f"   {i}. {mattress['name']} ({mattress['brand']})")
                            print(f"      💰 {mattress['price']}만원 | 📊 유사도: {mattress['similarity_score']:.2f}")
                            if mattress['features'][:2]:
                                print(f"      🏷️ {', '.join(mattress['features'][:2])}")
                    
                    # 처리 정보
                    print(f"\n📈 처리 정보: {processing_time:.1f}초 | "
                          f"OpenAI: {'✅' if result['openai_used'] else '❌'} | "
                          f"검색결과: {result['total_results']}개")
                
                else:
                    print(f"❌ 처리 실패: {result.get('error', '알 수 없는 오류')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다...")
                self._show_session_summary()
                break
            except Exception as e:
                logger.error(f"대화형 모드 오류: {e}")
                print(f"❌ 오류 발생: {e}")
    
    def run_batch_mode(self, queries_file: str):
        """배치 모드 실행"""
        if not self.is_initialized:
            print("❌ 시스템이 초기화되지 않았습니다")
            return
        
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            print(f"📂 배치 처리 시작: {len(queries)}개 쿼리")
            results = []
            
            for i, query in enumerate(queries, 1):
                print(f"\n[{i}/{len(queries)}] 처리 중: {query[:50]}...")
                
                result = self.agent.process_query(query)
                results.append({
                    'query': query,
                    'success': result.get('success', False),
                    'response': result.get('agent_response', ''),
                    'results_count': result.get('total_results', 0),
                    'processing_time': result.get('processing_time', 0)
                })
            
            # 결과 저장
            output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 통계 출력
            successful = len([r for r in results if r['success']])
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            
            print(f"\n✅ 배치 처리 완료!")
            print(f"📊 성공률: {successful}/{len(queries)} ({successful/len(queries)*100:.1f}%)")
            print(f"⏱️ 평균 처리시간: {avg_time:.2f}초")
            print(f"💾 결과 저장: {output_file}")
            
        except Exception as e:
            logger.error(f"배치 모드 실행 실패: {e}")
            print(f"❌ 배치 모드 실패: {e}")
    
    def _show_system_status(self):
        """시스템 상태 표시"""
        status = self.agent.get_agent_status()
        
        print(f"\n📊 시스템 상태 정보:")
        print(f"   준비 상태: {'✅ 준비됨' if status['ready'] else '❌ 미준비'}")
        print(f"   RAG 시스템: {'✅ 정상' if status['rag_system']['initialized'] else '❌ 오류'}")
        print(f"   OpenAI 연동: {'✅ 연결됨' if status['openai_available'] else '❌ 미연결'}")
        print(f"   한국어 최적화: {'✅ 활성화' if status['rag_system']['korean_optimized'] else '❌ 비활성화'}")
        
        print(f"\n📈 사용 통계:")
        conv_summary = status['conversation']
        print(f"   총 상호작용: {conv_summary['total_interactions']}회")
        print(f"   성공한 검색: {conv_summary['successful_searches']}회")
        print(f"   성공률: {conv_summary['success_rate']:.1%}")
        print(f"   사용자 참여도: {conv_summary['user_profile']['engagement_level']}")
        
        print(f"\n🔧 기술 정보:")
        print(f"   임베딩 모델: {status['rag_system']['embedding_model']}")
        print(f"   OpenAI 모델: {status.get('openai_model', 'N/A')}")
        print(f"   저장된 매트리스: {status['rag_system']['chroma_collection'].get('count', 0)}개")
    
    def _show_help(self):
        """도움말 표시"""
        print(f"\n📖 매트리스 AI 도움말:")
        print(f"   🔍 자연어 질문: '허리 아픈 사람한테 좋은 매트리스 추천해주세요'")
        print(f"   💰 예산 포함: '50만원 이하 시원한 매트리스 있나요?'")
        print(f"   🏥 건강 고려: '목 디스크 환자용 딱딱한 매트리스'")
        print(f"   👥 사용자별: '신혼부부용 킹사이즈 메모리폼'")
        print(f"   🏷️ 브랜드별: '시몬스 매트리스 중에 추천'")
        
        print(f"\n⌨️ 명령어:")
        print(f"   quit - 프로그램 종료")
        print(f"   status - 시스템 상태 확인")
        print(f"   recommend - 개인화 추천")
        print(f"   help - 이 도움말")
        
        print(f"\n💡 팁:")
        print(f"   • 구체적으로 질문할수록 더 정확한 추천을 받을 수 있습니다")
        print(f"   • 건강 문제, 예산, 선호도를 함께 말씀해주세요")
        print(f"   • 여러 번 질문하시면 개인화된 추천이 개선됩니다")
    
    def _show_personalized_recommendations(self):
        """개인화 추천 표시"""
        try:
            recommendations = self.agent.get_personalized_recommendations(n_results=5)
            
            if not recommendations:
                print("💭 아직 개인화 추천에 충분한 정보가 없습니다.")
                print("   몇 가지 질문을 더 해주시면 맞춤 추천이 가능합니다.")
                return
            
            print(f"\n🎯 개인화 매트리스 추천:")
            print("   (지금까지의 대화를 바탕으로 한 맞춤 추천)")
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n   {i}. {rec['name']} ({rec['brand']})")
                print(f"      💰 가격: {rec['price']}만원")
                print(f"      📊 적합도: {rec['similarity_score']:.2f}")
                print(f"      🏷️ 타입: {rec['type']}")
                if rec['features'][:2]:
                    print(f"      ✨ 특징: {', '.join(rec['features'][:2])}")
                if rec['target_users'][:2]:
                    print(f"      👥 추천대상: {', '.join(rec['target_users'][:2])}")
            
            # 사용자 프로필 정보
            user_profile = self.agent.conversation_manager.get_user_profile()
            if user_profile['primary_concerns']:
                print(f"\n📋 파악된 요구사항:")
                print(f"   건강 고려사항: {', '.join(user_profile['primary_concerns'])}")
                if user_profile.get('budget_range'):
                    print(f"   예산 범위: {user_profile['budget_range']}")
                if user_profile['preferences']:
                    prefs = [f"{k}: {v}" for k, v in user_profile['preferences'].items()]
                    print(f"   선호도: {', '.join(prefs)}")
            
        except Exception as e:
            logger.error(f"개인화 추천 실패: {e}")
            print(f"❌ 개인화 추천 생성 실패: {e}")
    
    def _show_session_summary(self):
        """세션 요약 표시"""
        try:
            summary = self.agent.conversation_manager.get_conversation_summary()
            
            print(f"\n📊 세션 요약:")
            print(f"   총 상호작용: {summary['total_interactions']}회")
            print(f"   성공한 검색: {summary['successful_searches']}회")
            print(f"   성공률: {summary['success_rate']:.1%}")
            
            if summary['recent_queries']:
                print(f"   최근 질문들:")
                for query in summary['recent_queries']:
                    print(f"     • {query[:50]}...")
            
            user_profile = summary['user_profile']
            if user_profile['primary_concerns']:
                print(f"   파악된 관심사: {', '.join(user_profile['primary_concerns'])}")
            
            print(f"   참여도: {user_profile['engagement_level']}")
            
        except Exception as e:
            logger.error(f"세션 요약 생성 실패: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="매트리스 쇼핑 가이드 AI Agent")
    parser.add_argument("--config", help="설정 파일 경로")
    parser.add_argument("--batch", help="배치 처리용 쿼리 파일")
    parser.add_argument("--reset-db", action="store_true", help="데이터베이스 리셋")
    parser.add_argument("--test", action="store_true", help="시스템 테스트 실행")
    
    args = parser.parse_args()
    
    try:
        # 시스템 초기화
        system = MattressAISystem(args.config)
        
        if not system.initialize():
            sys.exit(1)
        
        # 실행 모드 선택
        if args.test:
            print("🧪 시스템 테스트 모드")
            system._show_system_status()
            
        elif args.batch:
            print(f"📂 배치 모드: {args.batch}")
            system.run_batch_mode(args.batch)
            
        else:
            print("💬 대화형 모드")
            system.run_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 프로그램이 중단되었습니다")
    except Exception as e:
        logger.error(f"메인 실행 오류: {e}")
        print(f"❌ 실행 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()