"""
완전한 매트리스 데이터 로더 (Phase 1)
파일: src/data_loader.py
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MattressDataLoader:
    """매트리스 데이터 로더 클래스"""
    
    def __init__(self, data_path: Optional[str] = None):
        """데이터 로더 초기화"""
        # 프로젝트 루트 경로 설정
        self.project_root = Path(__file__).parent.parent
        
        # 데이터 파일 경로 설정
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = self.project_root / "data" / "mattress_data.json"
        
        # 데이터 저장 변수들 초기화
        self.raw_data: Optional[Dict] = None
        self.mattresses: List[Dict] = []
        self.buying_guide: Dict = {}
        self.metadata: Dict = {}
        
        # 상태 변수들 초기화
        self.is_loaded = False
        self.is_validated = False
        
        logger.info(f"데이터 로더 초기화 완료. 경로: {self.data_path}")
    
    def load_mattress_data(self) -> bool:
        """매트리스 데이터 JSON 파일 로드"""
        try:
            logger.info(f"데이터 로드 시작: {self.data_path}")
            
            # 파일 존재 확인
            if not self.data_path.exists():
                logger.error(f"파일이 존재하지 않습니다: {self.data_path}")
                return False
            
            # JSON 파일 로드
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            # 데이터 구조 분석
            self._parse_data_structure()
            
            self.is_loaded = True
            logger.info(f"✅ 데이터 로드 완료: {len(self.mattresses)}개")
            return True
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return False
    
    def _parse_data_structure(self):
        """로드된 데이터 구조 분석 및 파싱"""
        if not self.raw_data:
            raise ValueError("로드된 데이터가 없습니다")
        
        # 매트리스 데이터 추출 (여러 키 이름 시도)
        possible_keys = ['mattresses', 'products', 'mattress_list', 'data', 'items']
        
        self.mattresses = []
        for key in possible_keys:
            if key in self.raw_data:
                self.mattresses = self.raw_data[key]
                logger.info(f"'{key}' 키에서 매트리스 데이터 발견")
                break
        
        # 최상위가 배열인 경우
        if not self.mattresses and isinstance(self.raw_data, list):
            self.mattresses = self.raw_data
            logger.info("최상위 배열을 매트리스 데이터로 사용")
        
        # 여전히 없으면 경고
        if not self.mattresses:
            logger.warning("매트리스 데이터를 찾을 수 없습니다")
            logger.info(f"사용 가능한 키들: {list(self.raw_data.keys()) if isinstance(self.raw_data, dict) else '배열 구조'}")
            self.mattresses = []
        
        # 구매 가이드 및 메타데이터 추출
        self.buying_guide = self.raw_data.get('buying_guide', {}) if isinstance(self.raw_data, dict) else {}
        self.metadata = self.raw_data.get('metadata', {}) if isinstance(self.raw_data, dict) else {}
        
        logger.info(f"데이터 구조 분석 완료: 매트리스 {len(self.mattresses)}개")
    
    def validate_data(self) -> Tuple[bool, List[str]]:
        """데이터 유효성 검증"""
        if not self.is_loaded:
            return False, ["데이터가 로드되지 않았습니다"]
        
        errors = []
        
        # 기본 검증
        if not self.mattresses:
            errors.append("매트리스 데이터가 비어있습니다")
        else:
            # 각 매트리스 검증
            for i, mattress in enumerate(self.mattresses):
                if not isinstance(mattress, dict):
                    errors.append(f"매트리스 {i+1}: 올바르지 않은 데이터 형식")
                    continue
                
                # 필수 필드 중 최소 하나는 있어야 함
                essential_fields = ['name', 'id', 'brand', 'type']
                if not any(mattress.get(field) for field in essential_fields):
                    errors.append(f"매트리스 {i+1}: 필수 정보가 부족합니다")
        
        self.is_validated = len(errors) == 0
        
        if self.is_validated:
            logger.info("✅ 데이터 검증 완료")
        else:
            logger.warning(f"⚠️ 데이터 검증 실패: {len(errors)}개 오류")
        
        return self.is_validated, errors
    
    def get_mattress_by_id(self, mattress_id: str) -> Optional[Dict]:
        """ID로 매트리스 조회"""
        if not self.is_loaded:
            return None
        
        for mattress in self.mattresses:
            if mattress.get('id') == mattress_id:
                return mattress.copy()
        return None
    
    def get_all_mattresses(self) -> List[Dict]:
        """모든 매트리스 데이터 반환"""
        if not self.is_loaded:
            return []
        return self.mattresses.copy()
    
    def get_buying_guide(self) -> Dict:
        """구매 가이드 데이터 반환"""
        if not self.is_loaded:
            return {}
        return self.buying_guide.copy()
    
    def preprocess_for_rag(self) -> List[Dict]:
        """RAG 시스템용 텍스트 전처리 (ChromaDB 호환)"""
        if not self.is_loaded:
            logger.warning("데이터가 로드되지 않았습니다")
            return []
        
        rag_data = []
        
        for i, mattress in enumerate(self.mattresses):
            # 안전한 필드 접근
            def safe_get(key, default=''):
                value = mattress.get(key, default)
                return str(value) if value is not None else default
            
            def safe_join(key, default_list=None):
                if default_list is None:
                    default_list = []
                value = mattress.get(key, default_list)
                if isinstance(value, list):
                    return ', '.join(str(item) for item in value if item)
                return str(value) if value else ''
            
            # 검색 텍스트 구성 요소들
            search_parts = []
            
            # 기본 정보
            name = safe_get('name') or safe_get('mattress_name') or safe_get('product_name') or f"매트리스 {i+1}"
            brand = safe_get('brand') or safe_get('company') or safe_get('manufacturer')
            mattress_type = safe_get('type') or safe_get('material_category') or safe_get('category')
            
            # 가격 정보
            price = 0.0
            for price_key in ['base_price', 'price', 'final_price', 'cost']:
                if price_key in mattress:
                    try:
                        price = float(mattress[price_key])
                        break
                    except (ValueError, TypeError):
                        continue
            
            # 검색 텍스트 생성
            search_parts.append(f"매트리스명: {name}")
            if brand:
                search_parts.append(f"브랜드: {brand}")
            if mattress_type:
                search_parts.append(f"타입: {mattress_type}")
            if price > 0:
                search_parts.append(f"가격: {price}만원")
            
            # 특징들 (리스트 처리)
            features = []
            for feature_key in ['features', 'characteristics', 'benefits', 'pros']:
                if feature_key in mattress:
                    feature_value = mattress[feature_key]
                    if isinstance(feature_value, list):
                        features.extend([str(f) for f in feature_value if f])
                    elif feature_value:
                        features.append(str(feature_value))
            
            if features:
                search_parts.append(f"특징: {', '.join(features[:5])}")  # 최대 5개
            
            # 추천 대상 (리스트 처리)
            targets = []
            for target_key in ['target_users', 'recommended_for', 'suitable_for']:
                if target_key in mattress:
                    target_value = mattress[target_key]
                    if isinstance(target_value, list):
                        targets.extend([str(t) for t in target_value if t])
                    elif target_value:
                        targets.append(str(target_value))
            
            if targets:
                search_parts.append(f"추천: {', '.join(targets[:3])}")  # 최대 3개
            
            # 기타 정보
            for key, label in [
                ('firmness', '단단함'), ('firmness_options', '단단함'),
                ('sizes', '사이즈'), ('size', '사이즈'),
                ('thickness', '두께'), ('material', '소재')
            ]:
                if key in mattress and mattress[key]:
                    value = safe_join(key) if isinstance(mattress[key], list) else str(mattress[key])
                    if value and len(value) < 50:  # 너무 긴 텍스트 제외
                        search_parts.append(f"{label}: {value}")
            
            # 최종 검색 텍스트
            search_text = ' | '.join(search_parts)
            
            # ChromaDB 호환 메타데이터 생성 (리스트를 문자열로 변환)
            rag_item = {
                'id': safe_get('id') or f"mattress_{i}",
                'name': name,
                'search_text': search_text,
                'metadata': {
                    'name': name,
                    'brand': brand,
                    'type': mattress_type,
                    'price': float(price) if price else 0.0,  # float 타입 명시
                    'features_text': ', '.join(features[:5]) if features else '',  # 리스트를 문자열로
                    'target_users_text': ', '.join(targets[:3]) if targets else '',  # 리스트를 문자열로
                    'features_count': len(features),  # 개수는 정수로
                    'target_users_count': len(targets),  # 개수는 정수로
                    'has_features': len(features) > 0,  # 불린 값
                    'has_targets': len(targets) > 0  # 불린 값
                },
                'original_data': mattress.copy(),
                'features_list': features[:5],  # 원본 리스트는 별도 보관
                'target_users_list': targets[:3]  # 원본 리스트는 별도 보관
            }
            
            rag_data.append(rag_item)
        
        logger.info(f"RAG 데이터 전처리 완료: {len(rag_data)}개 (ChromaDB 호환)")
        return rag_data
    
    def get_data_summary(self) -> Dict:
        """데이터 요약 정보 반환"""
        if not self.is_loaded:
            return {"error": "데이터가 로드되지 않았습니다"}
        
        summary = {
            'loaded': self.is_loaded,
            'validated': self.is_validated,
            'total_mattresses': len(self.mattresses),
            'has_buying_guide': bool(self.buying_guide),
            'data_path': str(self.data_path),
            'timestamp': datetime.now().isoformat()
        }
        
        # 브랜드 통계
        if self.mattresses:
            brands = [m.get('brand', '') for m in self.mattresses if m.get('brand')]
            summary['unique_brands'] = len(set(brands))
            
            # 타입 통계
            types = [m.get('type', '') for m in self.mattresses if m.get('type')]
            summary['types'] = list(set(types))
        
        return summary

# 편의 함수
def load_and_validate_data(data_path: Optional[str] = None) -> Tuple[MattressDataLoader, bool]:
    """데이터 로더 생성, 로드, 검증 편의 함수"""
    loader = MattressDataLoader(data_path)
    
    if not loader.load_mattress_data():
        return loader, False
    
    is_valid, errors = loader.validate_data()
    if not is_valid:
        logger.warning(f"검증 오류: {errors}")
    
    return loader, True

# 테스트 실행
if __name__ == "__main__":
    print("🛏️ 매트리스 데이터 로더 테스트")
    print("=" * 50)
    
    loader, success = load_and_validate_data()
    
    if success:
        summary = loader.get_data_summary()
        print(f"✅ 로드 성공: {summary['total_mattresses']}개 매트리스")
        
        # RAG 테스트
        rag_data = loader.preprocess_for_rag()
        if rag_data:
            print(f"🔄 RAG 전처리: {len(rag_data)}개 항목")
            print(f"샘플: {rag_data[0]['search_text'][:100]}...")
        
        print("✅ 모든 테스트 통과!")
    else:
        print("❌ 데이터 로드 실패")