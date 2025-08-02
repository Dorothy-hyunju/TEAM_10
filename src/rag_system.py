"""
RAG 시스템 - GPT 동적 동의어 + Few-shot 강화 버전
파일: src/rag_system.py

주요 개선:
1. GPT 기반 동적 동의어 생성
2. Few-shot 학습 적용
3. 유사도 점수 극대화
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np
from datetime import datetime
import time
import re

from dotenv import load_dotenv
load_dotenv()

# 한국어 특화 임베딩
try:
    from sentence_transformers import SentenceTransformer
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# OpenAI 클라이언트
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 프로젝트 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import MattressDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPTSynonymGenerator:
    """GPT 기반 동적 동의어 생성기"""
    
    def __init__(self, openai_client=None):
        self.client = openai_client
        self.synonym_cache = {}
        self.few_shot_examples = self._get_few_shot_examples()
    
    def _get_few_shot_examples(self) -> str:
        """Few-shot 학습용 동의어 생성 예시"""
        return """
매트리스 관련 동의어 생성 예시들:

입력: "딱딱한"
출력: ["단단한", "하드", "견고한", "탄탄한", "펌", "강한", "튼튼한", "solid", "firm"]

입력: "허리통증"  
출력: ["요통", "허리아픔", "요추통증", "허리디스크", "척추통증", "등통증", "요추질환", "허리문제"]

입력: "시원한"
출력: ["쿨링", "냉감", "통풍", "서늘한", "차가운", "쿨", "시원함", "cool", "냉기"]

입력: "메모리폼"
출력: ["기억장치", "템퍼", "비스코", "템퍼폼", "기억폼", "memory foam", "점탄성폼", "저반발폼"]

입력: "커플"
출력: ["부부", "신혼", "연인", "2인", "둘이서", "부부용", "커플용", "파트너", "couple"]

입력: "아이"
출력: ["어린이", "아기", "유아", "학생", "성장기", "아동", "어린아이", "키즈", "child"]
"""
    
    def generate_synonyms(self, keyword: str) -> List[str]:
        """GPT로 동적 동의어 생성"""
        if not self.client:
            return []
        
        if keyword in self.synonym_cache:
            return self.synonym_cache[keyword]
        
        try:
            system_prompt = f"""
당신은 매트리스 도메인 전문가입니다. 주어진 키워드의 동의어와 유사어를 생성하세요.

{self.few_shot_examples}

규칙:
1. 매트리스 쇼핑 맥락에서 실제 사용되는 표현
2. 정확한 동의어 8-10개 생성
3. 한국어와 영어 모두 포함
4. JSON 배열 형태로만 응답
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"키워드: '{keyword}'"}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            synonyms = json.loads(content)
            
            if isinstance(synonyms, list):
                self.synonym_cache[keyword] = synonyms
                logger.debug(f"GPT 동의어 생성: {keyword} → {len(synonyms)}개")
                return synonyms
                
        except Exception as e:
            logger.error(f"GPT 동의어 생성 실패: {keyword}, {e}")
        
        return []


class EnhancedKoreanTextPreprocessor:
    """GPT 강화 한국어 텍스트 전처리기"""
    
    def __init__(self, gpt_synonym_generator=None):
        self.gpt_synonym_generator = gpt_synonym_generator
        
        # 기본 불용어
        self.stopwords = {
            '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '로', '으로',
            '입니다', '습니다', '있습니다', '합니다', '됩니다', '있어요', '해요',
            '그리고', '하지만', '그런데', '또한', '또', '그래서', '따라서',
            '매우', '정말', '아주', '너무', '조금', '약간', '좀', '많이'
        }
        
        # 키워드 중요도 가중치
        self.keyword_weights = {
            # 건강 관련 (최고 중요도)
            '허리': 5.0, '목': 5.0, '통증': 5.0, '디스크': 5.0,
            '요추': 4.5, '척추': 4.5, '경추': 4.5,
            
            # 소재/타입 관련 (높은 중요도)  
            '메모리폼': 4.0, '라텍스': 4.0, '스프링': 4.0,
            '템퍼': 3.5, '코일': 3.5,
            
            # 감촉 관련 (높은 중요도)
            '딱딱': 3.5, '부드러': 3.5, '시원': 3.5,
            '하드': 3.0, '소프트': 3.0, '쿨링': 3.0
        }
    
    def extract_weighted_keywords(self, text: str) -> List[Tuple[str, float]]:
        """가중치 기반 키워드 추출"""
        words = self.normalize_text(text).split()
        
        weighted_keywords = []
        for word in words:
            if len(word) > 1 and word not in self.stopwords:
                weight = self.keyword_weights.get(word, 1.0)
                weighted_keywords.append((word, weight))
        
        # 가중치 기준 정렬
        return sorted(weighted_keywords, key=lambda x: x[1], reverse=True)
    
    def normalize_text(self, text: str) -> str:
        """한국어 텍스트 정규화"""
        if not text:
            return ""
        
        text = text.strip()
        text = re.sub(r'[^\w\s가-힣a-zA-Z0-9.,!?%-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\d+)\s*만\s*원', r'\1만원', text)
        
        return text.strip()
    
    def create_gpt_enhanced_text(self, text: str) -> str:
        """GPT 동의어를 활용한 텍스트 강화"""
        normalized = self.normalize_text(text)
        enhanced_parts = [normalized]
        
        # 가중치 기반 키워드 추출
        weighted_keywords = self.extract_weighted_keywords(normalized)
        
        # GPT 동의어 생성 (상위 키워드만)
        if self.gpt_synonym_generator:
            for keyword, weight in weighted_keywords[:5]:  # 상위 5개만
                synonyms = self.gpt_synonym_generator.generate_synonyms(keyword)
                
                if synonyms:
                    # 가중치에 따라 반복 횟수 결정
                    repeat_count = min(int(weight), 3)
                    selected_synonyms = synonyms[:6]  # 상위 6개 동의어
                    
                    for _ in range(repeat_count):
                        enhanced_parts.extend(selected_synonyms)
        
        # 중요 키워드 강조
        priority_keywords = [kw for kw, weight in weighted_keywords if weight >= 3.0]
        enhanced_parts.extend(priority_keywords * 2)  # 2번 반복
        
        return ' '.join(enhanced_parts)


class FewShotEnhancedEmbeddingManager:
    """Few-shot 학습 강화 임베딩 매니저"""
    
    def __init__(self, model_name: str = None, gpt_synonym_generator=None):
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("sentence-transformers가 설치되지 않았습니다")
        
        # 한국어 모델 우선순위
        korean_models = [
            "jhgan/ko-sroberta-multitask",
            "snunlp/KR-SBERT-V40K-klueNLI-augSTS", 
            "BM-K/KoSimCSE-roberta-multitask",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "all-MiniLM-L6-v2"
        ]
        
        if model_name:
            korean_models.insert(0, model_name)
        
        # 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_name = None
        
        for model in korean_models:
            try:
                logger.info(f"모델 로드 시도: {model}")
                self.model = SentenceTransformer(model, device=self.device)
                self.model_name = model
                logger.info(f"✅ 모델 로드 성공: {model}")
                break
            except Exception as e:
                logger.warning(f"모델 {model} 로드 실패: {e}")
                continue
        
        if not self.model:
            raise RuntimeError("사용 가능한 임베딩 모델이 없습니다")
        
        # 전처리기 및 캐시 초기화
        self.preprocessor = EnhancedKoreanTextPreprocessor(gpt_synonym_generator)
        self.embedding_cache = {}
        self.few_shot_examples = self._get_few_shot_examples()
        
        logger.info(f"Few-shot 강화 임베딩 매니저 초기화 완료")
        logger.info(f"모델: {self.model_name}, 디바이스: {self.device}")
    
    def _get_few_shot_examples(self) -> str:
        """Few-shot 학습용 쿼리 확장 예시"""
        return """
매트리스 검색 쿼리 확장 방법:

원본: "허리 아픈 사람 매트리스"
확장: "허리 아픈 사람 매트리스 요통 척추통증 허리디스크 체압분산 지지력 딱딱한 하드 펌 척추정렬"

원본: "더위 타는 사람용"
확장: "더위 타는 사람용 시원한 쿨링 냉감 통기성 젤메모리폼 온도조절 통풍 환기"

원본: "신혼부부 킹사이즈"
확장: "신혼부부 킹사이즈 커플 부부 연인 2인 동작격리 진동차단 넓은공간"

원본: "50만원 가성비"
확장: "50만원 가성비 예산 경제적 합리적 저렴한 중간가격대 적당한가격"
"""
    
    def generate_few_shot_expansion(self, query: str) -> str:
        """Few-shot 학습 기반 쿼리 확장"""
        if not self.preprocessor.gpt_synonym_generator or not self.preprocessor.gpt_synonym_generator.client:
            return query
        
        try:
            system_prompt = f"""
매트리스 검색 전문가로서 쿼리를 확장하여 검색 성능을 향상시키세요.

{self.few_shot_examples}

규칙:
1. 원본 쿼리 + 동의어 + 관련 용어
2. 매트리스 도메인 특화 용어 사용
3. 검색 의도 정확히 파악
4. 확장된 텍스트만 반환
"""
            
            response = self.preprocessor.gpt_synonym_generator.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"확장할 쿼리: '{query}'"}
                ],
                max_tokens=150,
                temperature=0.4
            )
            
            expanded = response.choices[0].message.content.strip()
            logger.debug(f"Few-shot 확장: {query} → {expanded[:50]}...")
            return expanded
            
        except Exception as e:
            logger.error(f"Few-shot 확장 실패: {e}")
            return query
    
    def generate_embedding(self, text: str, use_enhancement: bool = True) -> List[float]:
        """향상된 임베딩 생성"""
        cache_key = f"{text}_{use_enhancement}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            if not text or not text.strip():
                text = "빈 텍스트"
            
            if use_enhancement:
                # 1. Few-shot 확장
                expanded_text = self.generate_few_shot_expansion(text)
                
                # 2. GPT 동의어 강화
                enhanced_text = self.preprocessor.create_gpt_enhanced_text(expanded_text)
            else:
                enhanced_text = self.preprocessor.normalize_text(text)
            
            # 임베딩 생성
            embedding = self.model.encode(
                enhanced_text,
                convert_to_tensor=False,
                normalize_embeddings=True,  # 정규화로 유사도 최적화
                show_progress_bar=False
            )
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            default_dim = 768 if 'roberta' in self.model_name.lower() else 384
            return [0.0] * default_dim
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 16, 
                                 use_enhancement: bool = True) -> List[List[float]]:
        """배치 임베딩 생성"""
        if not texts:
            return []
        
        embeddings = []
        
        # 텍스트 전처리
        processed_texts = []
        for text in texts:
            if use_enhancement:
                expanded = self.generate_few_shot_expansion(text)
                enhanced = self.preprocessor.create_gpt_enhanced_text(expanded)
            else:
                enhanced = self.preprocessor.normalize_text(text)
            processed_texts.append(enhanced)
        
        # 배치 처리
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            
            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    batch_size=min(len(batch_texts), batch_size),
                    show_progress_bar=False
                )
                
                if isinstance(batch_embeddings, np.ndarray):
                    if len(batch_embeddings.shape) == 1:
                        batch_embeddings = [batch_embeddings.tolist()]
                    else:
                        batch_embeddings = batch_embeddings.tolist()
                
                embeddings.extend(batch_embeddings)
                logger.info(f"배치 임베딩 완료: {i + len(batch_texts)}/{len(texts)}")
                
            except Exception as e:
                logger.error(f"배치 임베딩 실패: {e}")
                # 개별 생성으로 폴백
                for text in batch_texts:
                    embedding = self.generate_embedding(text, use_enhancement)
                    embeddings.append(embedding)
        
        return embeddings


class ChromaDBManager:
    """ChromaDB 벡터 데이터베이스 관리"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection_name = "enhanced_mattress_collection_v4"
        self.collection = None
        
        logger.info(f"ChromaDB 매니저 초기화: {self.persist_directory}")
    
    def create_collection(self, reset: bool = False) -> bool:
        """컬렉션 생성"""
        try:
            if reset:
                try:
                    self.client.delete_collection(self.collection_name)
                    logger.info("기존 컬렉션 삭제")
                except:
                    pass
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "GPT + Few-shot 강화 매트리스 벡터 DB"}
            )
            
            logger.info(f"컬렉션 '{self.collection_name}' 준비 완료")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {e}")
            return False
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadatas: List[Dict], ids: List[str]) -> bool:
        """문서 추가"""
        if not self.collection:
            return False
        
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"문서 {len(documents)}개 추가 완료")
            return True
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """유사 문서 검색"""
        if not self.collection:
            return {}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return {}
    
    def get_collection_info(self) -> Dict:
        """컬렉션 정보"""
        if not self.collection:
            return {"error": "컬렉션 없음"}
        
        try:
            return {
                "name": self.collection_name,
                "count": self.collection.count(),
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            return {"error": str(e)}


class EnhancedMattressRAGSystem:
    """GPT + Few-shot 강화 매트리스 RAG 시스템"""
    
    def __init__(self, persist_directory: str = "./chroma_db", 
                 model_name: str = None, openai_api_key: str = None):
        
        # GPT 클라이언트 초기화
        gpt_client = None
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                gpt_client = OpenAI(api_key=openai_api_key)
                logger.info("✅ OpenAI 클라이언트 초기화 성공")
            except Exception as e:
                logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        
        # GPT 동의어 생성기
        self.gpt_synonym_generator = GPTSynonymGenerator(gpt_client)
        
        # 컴포넌트 초기화
        self.embedding_manager = FewShotEnhancedEmbeddingManager(
            model_name, self.gpt_synonym_generator
        )
        self.chroma_manager = ChromaDBManager(persist_directory)
        self.data_loader = None
        
        self.is_initialized = False
        self.gpt_available = gpt_client is not None
        
        logger.info("Enhanced RAG 시스템 초기화 완료")
        logger.info(f"GPT 동의어 생성: {'✅' if self.gpt_available else '❌'}")
    
    def initialize_with_data(self, data_loader: MattressDataLoader, reset_db: bool = False) -> bool:
        """데이터로 시스템 초기화"""
        try:
            logger.info("Enhanced RAG 시스템 데이터 초기화 시작")
            
            self.data_loader = data_loader
            
            if not self.chroma_manager.create_collection(reset=reset_db):
                return False
            
            # 기존 데이터 확인
            collection_info = self.chroma_manager.get_collection_info()
            if collection_info.get("count", 0) > 0 and not reset_db:
                logger.info(f"기존 데이터 사용: {collection_info['count']}개")
                self.is_initialized = True
                return True
            
            # RAG 데이터 전처리
            rag_data = data_loader.preprocess_for_rag()
            if not rag_data:
                return False
            
            documents = [item['search_text'] for item in rag_data]
            metadatas = [item['metadata'] for item in rag_data]
            ids = [item['id'] for item in rag_data]
            
            logger.info(f"강화된 임베딩 생성 시작: {len(documents)}개")
            
            # 강화된 임베딩 생성
            embeddings = self.embedding_manager.generate_embeddings_batch(
                documents, use_enhancement=True
            )
            
            # ChromaDB 저장
            if self.chroma_manager.add_documents(documents, embeddings, metadatas, ids):
                self.is_initialized = True
                logger.info("✅ Enhanced RAG 시스템 초기화 완료")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Enhanced RAG 초기화 실패: {e}")
            return False
    
    def search_mattresses(self, query: str, n_results: int = 5, 
                         budget_filter: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """다중 전략 강화 검색"""
        if not self.is_initialized:
            return []
        
        try:
            logger.info(f"Enhanced 검색 시작: '{query}'")
            
            all_results = {}
            
            # 전략 1: GPT Few-shot + 동의어 강화 검색 (최고 가중치)
            enhanced_results = self._search_with_full_enhancement(query, n_results * 2)
            self._add_weighted_results(all_results, enhanced_results, 1.0, 'enhanced')
            
            # 전략 2: Few-shot만 적용 검색
            few_shot_results = self._search_with_few_shot_only(query, n_results * 2)
            self._add_weighted_results(all_results, few_shot_results, 0.8, 'few_shot')
            
            # 전략 3: 원본 쿼리 검색
            original_results = self._search_with_original(query, n_results)
            self._add_weighted_results(all_results, original_results, 0.6, 'original')
            
            # 최종 결과 계산
            final_results = self._calculate_final_results(
                all_results, budget_filter, n_results
            )
            
            logger.info(f"Enhanced 검색 완료: {len(final_results)}개")
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced 검색 실패: {e}")
            return []
    
    def _search_with_full_enhancement(self, query: str, n_results: int) -> Dict:
        """완전 강화 검색 (Few-shot + GPT 동의어)"""
        embedding = self.embedding_manager.generate_embedding(query, use_enhancement=True)
        return self.chroma_manager.search_similar(embedding, n_results)
    
    def _search_with_few_shot_only(self, query: str, n_results: int) -> Dict:
        """Few-shot만 적용 검색"""
        expanded_query = self.embedding_manager.generate_few_shot_expansion(query)
        embedding = self.embedding_manager.generate_embedding(expanded_query, use_enhancement=False)
        return self.chroma_manager.search_similar(embedding, n_results)
    
    def _search_with_original(self, query: str, n_results: int) -> Dict:
        """원본 쿼리 검색"""
        embedding = self.embedding_manager.generate_embedding(query, use_enhancement=False)
        return self.chroma_manager.search_similar(embedding, n_results)
    
    def _add_weighted_results(self, all_results: Dict, search_results: Dict, 
                            weight: float, strategy: str):
        """가중치 적용 결과 추가"""
        if not search_results.get('documents'):
            return
        
        for i in range(len(search_results['documents'][0])):
            doc_id = search_results['ids'][0][i]
            distance = search_results['distances'][0][i]
            weighted_score = (1 - distance) * weight
            
            if doc_id not in all_results:
                all_results[doc_id] = {
                    'metadata': search_results['metadatas'][0][i],
                    'document': search_results['documents'][0][i],
                    'scores': {},
                    'total_score': 0,
                    'strategy_count': 0
                }
            
            all_results[doc_id]['scores'][strategy] = weighted_score
            all_results[doc_id]['total_score'] += weighted_score
            all_results[doc_id]['strategy_count'] += 1
    
    def _calculate_final_results(self, all_results: Dict, 
                               budget_filter: Optional[Tuple[int, int]], 
                               n_results: int) -> List[Dict]:
        """최종 결과 계산"""
        final_results = []
        
        for doc_id, data in all_results.items():
            metadata = data['metadata']
            price = metadata.get('price', 0)
            
            # 예산 필터
            if budget_filter:
                min_budget, max_budget = budget_filter
                if price < min_budget or price > max_budget:
                    continue
            
            # 최종 점수 계산
            avg_score = data['total_score'] / data['strategy_count']
            
            # 강화 전략 보너스
            enhancement_bonus = 0
            if 'enhanced' in data['scores']:
                enhancement_bonus += 0.15  # GPT 강화 보너스
            if 'few_shot' in data['scores']:
                enhancement_bonus += 0.1   # Few-shot 보너스
            
            # 다중 전략 보너스
            multi_bonus = min(data['strategy_count'] * 0.05, 0.15)
            
            final_score = min(avg_score + enhancement_bonus + multi_bonus, 1.0)
            
            # 결과 포맷팅
            mattress_info = self._format_result(doc_id, data, final_score)
            final_results.append(mattress_info)
        
        # 점수 기준 정렬
        final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return final_results[:n_results]
    
    def _format_result(self, doc_id: str, data: Dict, final_score: float) -> Dict:
        """결과 포맷팅"""
        metadata = data['metadata']
        
        # features와 target_users 복원
        features_text = metadata.get('features_text', '')
        features = [f.strip() for f in features_text.split(',') if f.strip()] if features_text else []
        
        target_users_text = metadata.get('target_users_text', '')
        target_users = [t.strip() for t in target_users_text.split(',') if t.strip()] if target_users_text else []
        
        return {
            'id': doc_id,
            'name': metadata.get('name', ''),
            'brand': metadata.get('brand', ''),
            'type': metadata.get('type', ''),
            'price': int(round(metadata.get('price', 0))),  # 정수로 반올림
            'price_won': metadata.get('price_won', metadata.get('price', 0) * 10000),
            'similarity_score': final_score,
            'search_text': data['document'],
            'features': features,
            'target_users': target_users,
            'features_count': len(features),
            'target_users_count': len(target_users),
            'strategies_used': list(data['scores'].keys()),
            'gpt_enhanced': self.gpt_available,
            'enhanced_system': True
        }
    
    def get_mattress_by_id(self, mattress_id: str) -> Optional[Dict]:
        """ID로 매트리스 조회"""
        if not self.data_loader:
            return None
        return self.data_loader.get_mattress_by_id(mattress_id)
    
    def get_system_status(self) -> Dict:
        """시스템 상태"""
        chroma_info = self.chroma_manager.get_collection_info()
        
        return {
            'initialized': self.is_initialized,
            'embedding_model': self.embedding_manager.model_name,
            'embedding_device': self.embedding_manager.device,
            'gpt_available': self.gpt_available,
            'chroma_collection': chroma_info,
            'enhancement_features': {
                'gpt_dynamic_synonyms': self.gpt_available,
                'few_shot_expansion': True,
                'multi_strategy_search': True,
                'weighted_scoring': True
            }
        }


# 편의 함수
def setup_enhanced_rag_system(data_path: Optional[str] = None, reset_db: bool = False,
                             model_name: str = None, openai_api_key: str = None) -> Tuple[EnhancedMattressRAGSystem, bool]:
    """Enhanced RAG 시스템 설정"""
    try:
        # 데이터 로더
        data_loader = MattressDataLoader(data_path)
        if not data_loader.load_mattress_data():
            return None, False
        
        # Enhanced RAG 시스템
        rag_system = EnhancedMattressRAGSystem(
            model_name=model_name,
            openai_api_key=openai_api_key
        )
        
        if not rag_system.initialize_with_data(data_loader, reset_db):
            return rag_system, False
        
        logger.info("✅ Enhanced RAG 시스템 설정 완료")
        return rag_system, True
        
    except Exception as e:
        logger.error(f"Enhanced RAG 설정 실패: {e}")
        return None, False


# 기존 호환성을 위한 별칭
KoreanMattressRAGSystem = EnhancedMattressRAGSystem
setup_korean_rag_system = setup_enhanced_rag_system


if __name__ == "__main__":
    print("🚀 Enhanced 매트리스 RAG 시스템 테스트")
    print("=" * 60)
    
    try:
        import os
        openai_key = os.getenv('OPENAI_API_KEY')
        
        rag_system, success = setup_enhanced_rag_system(
            openai_api_key=openai_key,
            reset_db=False
        )
        
        if success:
            status = rag_system.get_system_status()
            print(f"\n📊 시스템 상태:")
            print(f"  초기화: {status['initialized']}")
            print(f"  임베딩 모델: {status['embedding_model']}")
            print(f"  GPT 사용: {status['gpt_available']}")
            print(f"  저장된 문서: {status['chroma_collection'].get('count', 0)}개")
            
            enhancement_features = status['enhancement_features']
            print(f"\n🚀 강화 기능:")
            for feature, enabled in enhancement_features.items():
                print(f"   {'✅' if enabled else '❌'} {feature}")
            
            # 테스트 검색
            test_queries = [
                "허리 디스크 환자용 딱딱한 매트리스",
                "더위 많이 타는 사람용 시원한 매트리스", 
                "신혼부부 킹사이즈 메모리폼",
                "50만원대 가성비 좋은 매트리스"
            ]
            
            print(f"\n🔍 Enhanced 검색 테스트:")
            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. '{query}'")
                
                results = rag_system.search_mattresses(query, n_results=3)
                
                if results:
                    for j, result in enumerate(results, 1):
                        strategies = result.get('strategies_used', [])
                        print(f"   {j}. {result['name']} ({result['brand']})")
                        print(f"      유사도: {result['similarity_score']:.3f}")
                        print(f"      전략: {', '.join(strategies)}")
                        print(f"      GPT강화: {result.get('gpt_enhanced', False)}")
                else:
                    print("   결과 없음")
            
            print(f"\n✅ Enhanced RAG 시스템 테스트 완료!")
            
        else:
            print("❌ Enhanced RAG 시스템 설정 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()