"""
RAG 시스템 - 허깅페이스 임베딩 + ChromaDB
파일: src/rag_system.py

역할:
- 허깅페이스 임베딩 생성
- ChromaDB 벡터 검색
- 매트리스 데이터 전처리 및 검색
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

# 허깅페이스 임베딩
try:
    from sentence_transformers import SentenceTransformer
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("⚠️ 허깅페이스 라이브러리가 필요합니다:")
    print("pip install sentence-transformers torch")

# 프로젝트 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import MattressDataLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceEmbeddingManager:
    """허깅페이스 임베딩 관리 클래스"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        허깅페이스 임베딩 매니저 초기화
        
        Args:
            model_name: 사용할 허깅페이스 모델명 (한국어 지원)
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("sentence-transformers가 설치되지 않았습니다")
        
        try:
            # GPU 사용 가능 여부 확인
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 모델 로드 (한국어 지원 모델 우선)
            try:
                self.model = SentenceTransformer(model_name, device=self.device)
                self.model_name = model_name
            except Exception:
                # 폴백 모델
                logger.warning("기본 모델 로드 실패, 폴백 모델 사용")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self.model_name = 'all-MiniLM-L6-v2'
            
            # 임베딩 캐시 (성능 향상)
            self.embedding_cache = {}
            
            logger.info(f"허깅페이스 임베딩 매니저 초기화 완료")
            logger.info(f"모델: {self.model_name}")
            logger.info(f"디바이스: {self.device}")
            
        except Exception as e:
            logger.error(f"허깅페이스 모델 로드 실패: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트의 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            List[float]: 임베딩 벡터
        """
        # 캐시 확인
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # 텍스트 전처리
            if not text or not text.strip():
                text = "빈 텍스트"
            
            # 임베딩 생성
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # numpy array를 list로 변환
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # 캐시 저장
            self.embedding_cache[text] = embedding
            
            logger.debug(f"임베딩 생성 완료: {text[:50]}... (차원: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            # 기본 차원의 더미 임베딩 반환
            default_dim = 384  # 대부분의 sentence-transformers 모델 기본 차원
            return [0.0] * default_dim
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        여러 텍스트의 임베딩을 배치로 생성
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            List[List[float]]: 임베딩 벡터들
        """
        if not texts:
            return []
        
        embeddings = []
        
        # 빈 텍스트 처리
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("빈 텍스트")
            else:
                processed_texts.append(text)
        
        try:
            # 배치 단위로 처리
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i + batch_size]
                
                # 캐시 확인 및 새 텍스트 분리
                batch_embeddings = []
                new_texts = []
                cache_map = {}
                
                for j, text in enumerate(batch_texts):
                    if text in self.embedding_cache:
                        batch_embeddings.append(self.embedding_cache[text])
                    else:
                        new_texts.append(text)
                        cache_map[len(new_texts) - 1] = j
                        batch_embeddings.append(None)  # 자리 표시
                
                # 새로운 텍스트들 임베딩 생성
                if new_texts:
                    try:
                        new_embeddings = self.model.encode(
                            new_texts, 
                            convert_to_tensor=False, 
                            batch_size=min(len(new_texts), batch_size)
                        )
                        
                        # numpy array를 list로 변환
                        if isinstance(new_embeddings, np.ndarray):
                            if len(new_embeddings.shape) == 1:  # 단일 임베딩인 경우
                                new_embeddings = [new_embeddings.tolist()]
                            else:
                                new_embeddings = new_embeddings.tolist()
                        
                        # 캐시 저장 및 결과 업데이트
                        for new_idx, (text, embedding) in enumerate(zip(new_texts, new_embeddings)):
                            self.embedding_cache[text] = embedding
                            batch_idx = cache_map[new_idx]
                            batch_embeddings[batch_idx] = embedding
                            
                    except Exception as e:
                        logger.error(f"배치 임베딩 생성 실패: {e}")
                        # 개별 생성으로 폴백
                        for new_idx, text in enumerate(new_texts):
                            embedding = self.generate_embedding(text)
                            batch_idx = cache_map[new_idx]
                            batch_embeddings[batch_idx] = embedding
                
                embeddings.extend(batch_embeddings)
                logger.info(f"배치 임베딩 완료: {i + len(batch_texts)}/{len(texts)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {e}")
            # 완전 폴백: 개별 생성
            return [self.generate_embedding(text) for text in processed_texts]

class ChromaDBManager:
    """ChromaDB 벡터 데이터베이스 관리 클래스"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        ChromaDB 매니저 초기화
        
        Args:
            persist_directory: ChromaDB 저장 디렉토리
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # 컬렉션 이름
        self.collection_name = "mattress_collection_v2"
        self.collection = None
        
        logger.info(f"ChromaDB 매니저 초기화 완료 (저장 위치: {self.persist_directory})")
    
    def create_collection(self, reset: bool = False) -> bool:
        """
        매트리스 컬렉션 생성
        
        Args:
            reset: 기존 컬렉션 삭제 후 재생성 여부
            
        Returns:
            bool: 생성 성공 여부
        """
        try:
            # 기존 컬렉션 삭제 (reset=True인 경우)
            if reset:
                try:
                    self.client.delete_collection(self.collection_name)
                    logger.info("기존 컬렉션 삭제 완료")
                except:
                    pass
            
            # 컬렉션 생성 또는 가져오기
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "매트리스 정보 벡터 데이터베이스 (허깅페이스 임베딩)"}
            )
            
            logger.info(f"컬렉션 '{self.collection_name}' 준비 완료")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {e}")
            return False
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadatas: List[Dict], ids: List[str]) -> bool:
        """
        문서들을 벡터 데이터베이스에 추가
        
        Args:
            documents: 문서 텍스트들
            embeddings: 임베딩 벡터들
            metadatas: 메타데이터들
            ids: 문서 ID들
            
        Returns:
            bool: 추가 성공 여부
        """
        if not self.collection:
            logger.error("컬렉션이 초기화되지 않았습니다")
            return False
        
        try:
            # 데이터 유효성 검사
            if len(documents) != len(embeddings) or len(documents) != len(metadatas) or len(documents) != len(ids):
                logger.error("문서, 임베딩, 메타데이터, ID 개수가 일치하지 않습니다")
                return False
            
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
        """
        유사한 문서 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            n_results: 반환할 결과 수
            
        Returns:
            Dict: 검색 결과
        """
        if not self.collection:
            logger.error("컬렉션이 초기화되지 않았습니다")
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
        """컬렉션 정보 반환"""
        if not self.collection:
            return {"error": "컬렉션이 초기화되지 않았습니다"}
        
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            return {"error": str(e)}

class MattressRAGSystem:
    """매트리스 RAG 검색 시스템 (임베딩 + 벡터 검색 전담)"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        매트리스 RAG 시스템 초기화
        
        Args:
            persist_directory: ChromaDB 저장 디렉토리
        """
        # 컴포넌트 초기화
        self.embedding_manager = HuggingFaceEmbeddingManager()
        self.chroma_manager = ChromaDBManager(persist_directory)
        self.data_loader = None
        
        # 시스템 상태
        self.is_initialized = False
        
        logger.info("매트리스 RAG 시스템 초기화 완료")
        logger.info(f"임베딩: 허깅페이스 ({self.embedding_manager.model_name})")
    
    def initialize_with_data(self, data_loader: MattressDataLoader, reset_db: bool = False) -> bool:
        """
        매트리스 데이터로 RAG 시스템 초기화
        
        Args:
            data_loader: 매트리스 데이터 로더
            reset_db: 데이터베이스 리셋 여부
            
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            logger.info("RAG 시스템 데이터 초기화 시작")
            
            # 데이터 로더 설정
            self.data_loader = data_loader
            
            # ChromaDB 컬렉션 생성
            if not self.chroma_manager.create_collection(reset=reset_db):
                return False
            
            # 기존 데이터 확인
            collection_info = self.chroma_manager.get_collection_info()
            if collection_info.get("count", 0) > 0 and not reset_db:
                logger.info(f"기존 데이터 발견: {collection_info['count']}개 문서")
                self.is_initialized = True
                return True
            
            # RAG용 데이터 전처리
            rag_data = data_loader.preprocess_for_rag()
            if not rag_data:
                logger.error("RAG용 데이터가 없습니다")
                return False
            
            # 텍스트와 메타데이터 추출
            documents = [item['search_text'] for item in rag_data]
            metadatas = [item['metadata'] for item in rag_data]
            ids = [item['id'] for item in rag_data]
            
            logger.info(f"허깅페이스 임베딩 생성 시작: {len(documents)}개 문서")
            
            # 허깅페이스로 임베딩 생성
            embeddings = self.embedding_manager.generate_embeddings_batch(documents)
            
            # ChromaDB에 저장
            if self.chroma_manager.add_documents(documents, embeddings, metadatas, ids):
                self.is_initialized = True
                logger.info("✅ RAG 시스템 초기화 완료")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 실패: {e}")
            return False
    
    def search_mattresses(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        매트리스 검색 (허깅페이스 임베딩 사용)
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색된 매트리스 정보들
        """
        if not self.is_initialized:
            logger.error("RAG 시스템이 초기화되지 않았습니다")
            return []
        
        try:
            # 허깅페이스로 쿼리 임베딩 생성
            query_embedding = self.embedding_manager.generate_embedding(query)
            
            # ChromaDB에서 유사한 매트리스 검색
            search_results = self.chroma_manager.search_similar(query_embedding, n_results)
            
            if not search_results or not search_results.get('documents'):
                logger.warning("검색 결과가 없습니다")
                return []
            
            # 결과 포맷팅
            mattresses = []
            for i in range(len(search_results['documents'][0])):
                # 메타데이터에서 안전하게 값 추출
                metadata = search_results['metadatas'][0][i]
                
                # 문자열로 저장된 features와 target_users를 다시 리스트로 변환
                features_text = metadata.get('features_text', '')
                features = [f.strip() for f in features_text.split(',') if f.strip()] if features_text else []
                
                target_users_text = metadata.get('target_users_text', '')
                target_users = [t.strip() for t in target_users_text.split(',') if t.strip()] if target_users_text else []
                
                mattress_info = {
                    'id': search_results['ids'][0][i],
                    'name': metadata.get('name', ''),
                    'brand': metadata.get('brand', ''),
                    'type': metadata.get('type', ''),
                    'price': metadata.get('price', 0),
                    'similarity_score': 1 - search_results['distances'][0][i],  # 거리를 유사도로 변환
                    'search_text': search_results['documents'][0][i],
                    'features': features,
                    'target_users': target_users,
                    'features_count': metadata.get('features_count', 0),
                    'target_users_count': metadata.get('target_users_count', 0)
                }
                mattresses.append(mattress_info)
            
            logger.info(f"검색 완료: {len(mattresses)}개 매트리스 발견")
            return mattresses
            
        except Exception as e:
            logger.error(f"매트리스 검색 실패: {e}")
            return []
    
    def get_mattress_by_id(self, mattress_id: str) -> Optional[Dict]:
        """ID로 특정 매트리스 상세 정보 조회"""
        if not self.data_loader:
            return None
        return self.data_loader.get_mattress_by_id(mattress_id)
    
    def get_system_status(self) -> Dict:
        """시스템 상태 정보 반환"""
        chroma_info = self.chroma_manager.get_collection_info()
        
        return {
            'initialized': self.is_initialized,
            'embedding_model': self.embedding_manager.model_name,
            'embedding_device': self.embedding_manager.device,
            'embedding_cache_size': len(self.embedding_manager.embedding_cache),
            'chroma_collection': chroma_info
        }

# 편의 함수
def setup_rag_system(data_path: Optional[str] = None, reset_db: bool = False) -> Tuple[MattressRAGSystem, bool]:
    """
    RAG 시스템 설정 편의 함수
    
    Args:
        data_path: 매트리스 데이터 파일 경로
        reset_db: 데이터베이스 리셋 여부
        
    Returns:
        Tuple[MattressRAGSystem, bool]: (RAG 시스템, 성공 여부)
    """
    try:
        # 허깅페이스 라이브러리 확인
        if not HUGGINGFACE_AVAILABLE:
            logger.error("허깅페이스 라이브러리가 설치되지 않았습니다")
            print("필요한 라이브러리를 설치하세요:")
            print("pip install sentence-transformers torch")
            return None, False
        
        # 데이터 로더 초기화
        data_loader = MattressDataLoader(data_path)
        if not data_loader.load_mattress_data():
            logger.error("매트리스 데이터 로드 실패")
            return None, False
        
        # RAG 시스템 초기화
        rag_system = MattressRAGSystem()
        if not rag_system.initialize_with_data(data_loader, reset_db):
            logger.error("RAG 시스템 초기화 실패")
            return rag_system, False
        
        logger.info("✅ RAG 시스템 설정 완료")
        return rag_system, True
        
    except Exception as e:
        logger.error(f"RAG 시스템 설정 실패: {e}")
        return None, False

# 테스트 실행
if __name__ == "__main__":
    print("🔍 매트리스 RAG 시스템 테스트")
    print("=" * 50)
    
    try:
        # RAG 시스템 설정
        rag_system, success = setup_rag_system(reset_db=False)
        
        if success:
            # 시스템 상태 확인
            status = rag_system.get_system_status()
            print(f"\n📊 시스템 상태:")
            print(f"  초기화: {status['initialized']}")
            print(f"  임베딩 모델: {status['embedding_model']}")
            print(f"  디바이스: {status['embedding_device']}")
            print(f"  저장된 문서: {status['chroma_collection'].get('count', 0)}개")
            
            # 테스트 검색
            test_queries = [
                "허리 통증을 위한 매트리스",
                "시원한 매트리스 추천",
                "50만원 이하 매트리스"
            ]
            
            print(f"\n🔍 검색 테스트:")
            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. 쿼리: '{query}'")
                
                results = rag_system.search_mattresses(query, n_results=3)
                
                if results:
                    print(f"   검색 결과: {len(results)}개")
                    for j, result in enumerate(results, 1):
                        print(f"   {j}. {result['name']} (유사도: {result['similarity_score']:.3f})")
                else:
                    print("   검색 결과 없음")
            
            print(f"\n✅ RAG 시스템 테스트 완료!")
            
        else:
            print(f"\n❌ RAG 시스템 설정 실패")
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")
        print(f"💡 필요한 설치: pip install sentence-transformers torch")