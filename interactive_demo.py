"""
매트리스 AI 에이전트 - 인터랙티브 데모
파일: interactive_demo.py

사용법:
python interactive_demo.py

기능:
- 실시간 질문/답변
- Enhanced 기능 시연
- 성능 모니터링
- 대화 기록 저장
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 매트리스 AI 에이전트 모듈 임포트
try:
    from src.ai_agent import create_enhanced_mattress_agent
    AI_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"❌ AI 에이전트 모듈 임포트 실패: {e}")
    print("💡 다음을 확인해주세요:")
    print("   1. src/ai_agent.py 파일이 존재하는지")
    print("   2. 필요한 의존성이 설치되었는지 (pip install -r requirements.txt)")
    AI_AGENT_AVAILABLE = False

# 로깅 설정 (데모에서는 에러만 표시)
logging.basicConfig(level=logging.ERROR)


class InteractiveMattressDemo:
    """인터랙티브 매트리스 AI 데모"""
    
    def __init__(self):
        self.agent = None
        self.conversation_history = []
        self.session_start = datetime.now()
        
        # 터미널 색상
        self.colors = {
            'header': '\033[95m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'white': '\033[97m',
            'end': '\033[0m',
            'bold': '\033[1m'
        }
    
    def colored_print(self, text: str, color: str = 'white', bold: bool = False):
        """색상 텍스트 출력"""
        color_code = self.colors.get(color, self.colors['white'])
        bold_code = self.colors['bold'] if bold else ''
        end_code = self.colors['end']
        print(f"{bold_code}{color_code}{text}{end_code}")
    
    def print_welcome(self):
        """환영 메시지"""
        welcome = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  🛏️  매트리스 AI 상담사와 대화하기  🤖                      ║
║                                                                              ║
║              Enhanced RAG + GPT 동의어 + Few-shot 학습 적용                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

💬 매트리스에 대해 무엇이든 물어보세요!

예시 질문들:
• "허리가 아픈 사람한테 좋은 매트리스 추천해주세요"
• "더위 많이 타는데 시원한 매트리스 있나요?"
• "50만원 정도 예산으로 가성비 좋은 매트리스 찾고 있어요"
• "신혼부부용 킹사이즈 매트리스 추천해주세요"

명령어:
• '/help' - 도움말 보기
• '/status' - 시스템 상태 확인
• '/history' - 대화 기록 보기
• '/clear' - 화면 정리
• '/save' - 대화 기록 저장
• '/quit' - 프로그램 종료

"""
        self.colored_print(welcome, 'cyan')
    
    def initialize_agent(self) -> bool:
        """AI 에이전트 초기화"""
        if not AI_AGENT_AVAILABLE:
            self.colored_print("❌ AI 에이전트를 사용할 수 없습니다.", 'red')
            return False
        
        self.colored_print("🚀 매트리스 AI 에이전트 초기화 중...", 'yellow')
        
        try:
            # API 키 확인
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.colored_print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다.", 'yellow')
                self.colored_print("   OpenAI GPT 기능이 제한될 수 있습니다.", 'yellow')
                print()
            
            # 에이전트 생성
            self.agent = create_enhanced_mattress_agent(api_key)
            
            # 상태 확인
            status = self.agent.get_agent_status()
            
            self.colored_print("✅ AI 에이전트 초기화 완료!", 'green', bold=True)
            
            # 시스템 상태 요약
            print()
            self.colored_print("📊 시스템 상태:", 'blue', bold=True)
            print(f"   • Enhanced RAG: {'✅' if status['enhanced_rag_system']['initialized'] else '❌'}")
            print(f"   • OpenAI 연동: {'✅' if status['openai_available'] else '❌'}")
            print(f"   • GPT 동의어: {'✅' if status['enhanced_rag_system']['gpt_available'] else '❌'}")
            print(f"   • 매트리스 데이터: {status['enhanced_rag_system']['chroma_collection'].get('count', 0)}개")
            
            enhanced_features = status['enhanced_capabilities']
            active_features = [name for name, enabled in enhanced_features.items() if enabled]
            print(f"   • 활성 기능: {len(active_features)}개")
            
            print()
            return True
            
        except Exception as e:
            self.colored_print(f"❌ 에이전트 초기화 실패: {e}", 'red')
            return False
    
    def process_user_query(self, user_input: str) -> bool:
        """사용자 질문 처리"""
        # 명령어 처리
        if user_input.startswith('/'):
            return self.handle_command(user_input)
        
        if not self.agent:
            self.colored_print("❌ AI 에이전트가 초기화되지 않았습니다.", 'red')
            return True
        
        try:
            self.colored_print("\n🤔 분석 중...", 'yellow')
            start_time = time.time()
            
            # AI 에이전트로 질문 처리
            result = self.agent.process_query(user_input, n_results=3)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result.get('success'):
                # AI 응답 출력
                self.colored_print("\n🤖 AI 상담사:", 'green', bold=True)
                self.colored_print("─" * 60, 'green')
                print(result['agent_response'])
                self.colored_print("─" * 60, 'green')
                
                # 추천 매트리스 상세 정보
                if result.get('search_results'):
                    self.colored_print("\n📋 추천 매트리스 상세:", 'blue', bold=True)
                    
                    for i, mattress in enumerate(result['search_results'][:3], 1):
                        print(f"\n{i}. {mattress.get('name', 'Unknown')} ({mattress.get('brand', 'Unknown')})")
                        print(f"   💰 가격: {int(mattress.get('price', 0))}만원")
                        print(f"   📊 유사도: {mattress.get('similarity_score', 0):.1%}")
                        print(f"   🏷️  타입: {mattress.get('type', 'Unknown')}")
                        
                        features = mattress.get('features', [])
                        if features:
                            print(f"   ✨ 특징: {', '.join(features[:3])}")
                        
                        target_users = mattress.get('target_users', [])
                        if target_users:
                            print(f"   👥 추천 대상: {', '.join(target_users[:2])}")
                
                # 성능 정보
                self.colored_print(f"\n⚡ 처리 시간: {processing_time:.2f}초", 'yellow')
                
                enhancement_info = result.get('enhancement_info', {})
                enhancements = []
                if enhancement_info.get('gpt_synonyms_used'):
                    enhancements.append("GPT 동의어")
                if enhancement_info.get('few_shot_enhanced'):
                    enhancements.append("Few-shot 학습")
                if enhancement_info.get('enhanced_rag_used'):
                    enhancements.append("Enhanced RAG")
                
                if enhancements:
                    self.colored_print(f"🚀 적용된 강화 기능: {', '.join(enhancements)}", 'cyan')
                
                avg_similarity = result.get('avg_similarity', 0)
                self.colored_print(f"🎯 평균 유사도: {avg_similarity:.1%}", 'cyan')
                
                # 대화 기록 저장
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user_query': user_input,
                    'ai_response': result['agent_response'],
                    'processing_time': processing_time,
                    'search_results_count': len(result.get('search_results', [])),
                    'avg_similarity': avg_similarity,
                    'enhancements_used': enhancements
                })
                
            else:
                self.colored_print(f"\n❌ 처리 실패: {result.get('error', '알 수 없는 오류')}", 'red')
            
        except Exception as e:
            self.colored_print(f"\n❌ 오류 발생: {e}", 'red')
        
        return True
    
    def handle_command(self, command: str) -> bool:
        """명령어 처리"""
        command = command.lower().strip()
        
        if command == '/help':
            self.show_help()
        elif command == '/status':
            self.show_status()
        elif command == '/history':
            self.show_history()
        elif command == '/clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            self.print_welcome()
        elif command == '/save':
            self.save_conversation()
        elif command in ['/quit', '/exit', '/q']:
            return False
        else:
            self.colored_print(f"❌ 알 수 없는 명령어: {command}", 'red')
            self.colored_print("'/help'를 입력하여 사용 가능한 명령어를 확인하세요.", 'yellow')
        
        return True
    
    def show_help(self):
        """도움말 표시"""
        help_text = """
💡 사용 가능한 명령어:

/help     - 이 도움말 표시
/status   - AI 에이전트 상태 확인
/history  - 지금까지의 대화 기록 보기
/clear    - 화면 정리
/save     - 대화 기록을 파일로 저장
/quit     - 프로그램 종료

📝 질문 예시:
• "허리 디스크 환자용 매트리스 추천해주세요"
• "예산 80만원으로 가성비 좋은 매트리스 있나요?"
• "더위 타는 사람용 시원한 매트리스 찾고 있어요"
• "신혼부부용 킹사이즈 메모리폼 매트리스 추천해주세요"

💡 팁: 구체적인 조건(예산, 건강 문제, 선호도)을 함께 말씀해주시면 더 정확한 추천을 받을 수 있습니다!
"""
        self.colored_print(help_text, 'cyan')
    
    def show_status(self):
        """시스템 상태 표시"""
        if not self.agent:
            self.colored_print("❌ AI 에이전트가 초기화되지 않았습니다.", 'red')
            return
        
        try:
            status = self.agent.get_agent_status()
            
            self.colored_print("\n📊 시스템 상태 상세:", 'blue', bold=True)
            print()
            
            # 기본 상태
            print(f"🤖 에이전트 준비: {'✅ 준비됨' if status['ready'] else '❌ 준비 안됨'}")
            print(f"🧠 OpenAI 연동: {'✅ 연결됨' if status['openai_available'] else '❌ 연결 안됨'}")
            
            if status['openai_available']:
                print(f"🤖 OpenAI 모델: {status.get('openai_model', 'N/A')}")
            
            # RAG 시스템
            rag_status = status['enhanced_rag_system']
            print(f"🔍 Enhanced RAG: {'✅ 활성' if rag_status['initialized'] else '❌ 비활성'}")
            print(f"📚 매트리스 데이터: {rag_status['chroma_collection'].get('count', 0)}개")
            print(f"🧮 임베딩 모델: {rag_status.get('embedding_model', 'N/A')}")
            print(f"💪 GPT 동의어: {'✅ 사용 가능' if rag_status['gpt_available'] else '❌ 사용 불가'}")
            
            # Enhanced 기능들
            enhanced_capabilities = status['enhanced_capabilities']
            print(f"\n🚀 Enhanced 기능:")
            for feature, enabled in enhanced_capabilities.items():
                status_icon = "✅" if enabled else "❌"
                feature_name = feature.replace('_', ' ').title()
                print(f"   {status_icon} {feature_name}")
            
            # 세션 통계
            conversation_stats = status.get('conversation', {})
            if conversation_stats:
                print(f"\n📈 세션 통계:")
                print(f"   • 총 상호작용: {conversation_stats.get('total_interactions', 0)}회")
                print(f"   • Enhanced 상호작용: {conversation_stats.get('enhanced_interactions', 0)}회")
                print(f"   • Enhancement 적용률: {conversation_stats.get('enhancement_rate', 0):.1%}")
            
            # 세션 정보
            session_duration = datetime.now() - self.session_start
            print(f"\n⏱️  세션 정보:")
            print(f"   • 시작 시간: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   • 진행 시간: {str(session_duration).split('.')[0]}")
            print(f"   • 대화 기록: {len(self.conversation_history)}개")
            
        except Exception as e:
            self.colored_print(f"❌ 상태 조회 실패: {e}", 'red')
    
    def show_history(self):
        """대화 기록 표시"""
        if not self.conversation_history:
            self.colored_print("📝 아직 대화 기록이 없습니다.", 'yellow')
            return
        
        self.colored_print(f"\n📝 대화 기록 ({len(self.conversation_history)}개):", 'blue', bold=True)
        
        for i, record in enumerate(self.conversation_history, 1):
            timestamp = datetime.fromisoformat(record['timestamp']).strftime('%H:%M:%S')
            
            print(f"\n{i}. [{timestamp}] 처리시간: {record['processing_time']:.2f}초")
            self.colored_print(f"   👤 질문: {record['user_query']}", 'cyan')
            self.colored_print(f"   🤖 답변: {record['ai_response'][:100]}{'...' if len(record['ai_response']) > 100 else ''}", 'green')
            
            if record.get('enhancements_used'):
                print(f"   🚀 적용 기능: {', '.join(record['enhancements_used'])}")
            
            print(f"   📊 유사도: {record.get('avg_similarity', 0):.1%} | 결과: {record.get('search_results_count', 0)}개")
    
    def save_conversation(self):
        """대화 기록 저장"""
        if not self.conversation_history:
            self.colored_print("💾 저장할 대화 기록이 없습니다.", 'yellow')
            return
        
        try:
            # demo_db 폴더 생성
            demo_db_path = Path("demo_db")
            demo_db_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = demo_db_path / f"mattress_ai_conversation_{timestamp}.json"
            
            save_data = {
                'session_start': self.session_start.isoformat(),
                'session_end': datetime.now().isoformat(),
                'total_conversations': len(self.conversation_history),
                'conversation_history': self.conversation_history
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.colored_print(f"💾 대화 기록이 저장되었습니다: {filename}", 'green')
            
        except Exception as e:
            self.colored_print(f"❌ 저장 실패: {e}", 'red')
    
    def run(self):
        """메인 실행 루프"""
        # 환영 메시지
        self.print_welcome()
        
        # 에이전트 초기화
        if not self.initialize_agent():
            self.colored_print("\n프로그램을 종료합니다.", 'red')
            return
        
        self.colored_print("\n💬 이제 질문해주세요! (종료하려면 '/quit' 입력)", 'green', bold=True)
        
        # 메인 대화 루프
        while True:
            try:
                print()
                self.colored_print("─" * 80, 'white')
                user_input = input("👤 질문: ").strip()
                
                if not user_input:
                    self.colored_print("❓ 질문을 입력해주세요.", 'yellow')
                    continue
                
                # 질문 처리
                if not self.process_user_query(user_input):
                    break
                
            except KeyboardInterrupt:
                print()
                self.colored_print("\n👋 프로그램을 종료합니다.", 'yellow')
                break
            except Exception as e:
                self.colored_print(f"\n❌ 예상치 못한 오류: {e}", 'red')
        
        # 종료 메시지
        if self.conversation_history:
            self.colored_print(f"\n📊 세션 요약: {len(self.conversation_history)}번의 대화", 'blue')
            
            save_choice = input("💾 대화 기록을 저장하시겠습니까? (y/N): ").strip().lower()
            if save_choice in ['y', 'yes', '네', 'ㅇ']:
                self.save_conversation()
        
        self.colored_print("\n🙏 매트리스 AI 상담사를 이용해주셔서 감사합니다!", 'cyan', bold=True)


def main():
    """메인 함수"""
    try:
        demo = InteractiveMattressDemo()
        demo.run()
    except Exception as e:
        print(f"\n❌ 프로그램 실행 오류: {e}")
        print("\n💡 문제 해결:")
        print("1. 필요한 모듈이 설치되었는지 확인: pip install -r requirements.txt")
        print("2. data/mattress_data.json 파일이 존재하는지 확인")
        print("3. OPENAI_API_KEY 환경변수 설정 (선택사항)")


if __name__ == "__main__":
    main()