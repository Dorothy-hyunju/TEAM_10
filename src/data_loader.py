"""
매트리스 데이터 로더 - ChromaDB 호환성 수정 버전
파일: src/data_loader.py

주요 수정:
1. ChromaDB ID 형식 정규화
2. 메타데이터 타입 검증
3. 중복 ID 처리
4. 오류 처리 강화
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MattressDataLoader:
    """매트리스 데이터 로더 클래스 (ChromaDB 호환성 강화)"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        데이터 로더 초기화
        
        Args:
            data_path: 매트리스 데이터 파일 경로
        """
        # 기본 경로 설정
        if data_path:
            self.data_file = Path(data_path)
        else:
            # 프로젝트 루트에서 data 폴더 찾기
            current_dir = Path(__file__).parent
            project_root = current_dir.parent  # src의 부모 디렉토리
            self.data_file = project_root / "data" / "mattress_data.json"
        
        # 데이터 저장용
        self.mattresses = []
        
        logger.info(f"데이터 로더 초기화 완료. 경로: {self.data_file}")
    
    def _sanitize_id(self, raw_id: str) -> str:
        """
        ChromaDB 호환 ID로 정규화
        
        Args:
            raw_id: 원본 ID
            
        Returns:
            str: 정규화된 ID
        """
        if not raw_id:
            return "unknown_mattress"
        
        # 1. 기본 정리
        sanitized = str(raw_id).strip()
        
        # 2. 한글 및 특수문자를 안전한 문자로 변환
        # 한글은 유니코드로, 특수문자는 언더스코어로
        sanitized = re.sub(r'[^\w가-힣]', '_', sanitized)
        
        # 3. 연속된 언더스코어 제거
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # 4. 시작/끝 언더스코어 제거
        sanitized = sanitized.strip('_')
        
        # 5. 길이 제한 (ChromaDB는 보통 ID 길이 제한이 있음)
        if len(sanitized) > 100:
            # 해시를 사용해서 고유성 보장
            hash_suffix = hashlib.md5(sanitized.encode()).hexdigest()[:8]
            sanitized = sanitized[:80] + "_" + hash_suffix
        
        # 6. 빈 문자열 처리
        if not sanitized:
            sanitized = "mattress_unknown"
        
        # 7. 숫자로만 시작하는 경우 방지
        if sanitized[0].isdigit():
            sanitized = "mattress_" + sanitized
        
        return sanitized
    
    def _generate_unique_id(self, mattress: Dict, existing_ids: set) -> str:
        """
        고유한 ID 생성
        
        Args:
            mattress: 매트리스 데이터
            existing_ids: 기존 ID 집합
            
        Returns:
            str: 고유한 ID
        """
        # 기본 ID 생성
        name = mattress.get('name', 'Unknown').strip()
        brand = mattress.get('brand', 'Unknown').strip()
        
        # 기본 ID 패턴
        base_id = f"mattress_{brand}_{name}"
        sanitized_id = self._sanitize_id(base_id)
        
        # 중복 확인 및 처리
        final_id = sanitized_id
        counter = 1
        
        while final_id in existing_ids:
            final_id = f"{sanitized_id}_{counter}"
            counter += 1
            
            # 무한 루프 방지
            if counter > 1000:
                # 해시 사용
                unique_hash = hashlib.md5(f"{base_id}_{counter}".encode()).hexdigest()[:8]
                final_id = f"mattress_{unique_hash}"
                break
        
        return final_id
    
    def _convert_price_to_manwon(self, price_won: Union[int, float, str]) -> float:
        """
        원 단위 가격을 만원 단위로 변환
        
        Args:
            price_won: 원 단위 가격 (예: 500000)
            
        Returns:
            float: 만원 단위 가격 (예: 50.0)
        """
        try:
            if isinstance(price_won, str):
                # 문자열에서 숫자만 추출
                price_won = re.sub(r'[^\d.]', '', price_won)
                price_won = float(price_won) if price_won else 0
            
            price_value = float(price_won)
            
            # 이미 만원 단위인지 확인 (1000 이하면 이미 만원 단위일 가능성)
            if price_value <= 1000:  # 1000만원 이하면 이미 만원 단위일 수 있음
                return price_value
            else:
                # 원 단위에서 만원 단위로 변환
                return price_value / 10000
                
        except (ValueError, TypeError):
            logger.warning(f"가격 변환 실패: {price_won}")
            return 0.0

    def _validate_metadata(self, metadata: Dict) -> Dict:
        """
        ChromaDB 메타데이터 검증 및 정리
        
        Args:
            metadata: 원본 메타데이터
            
        Returns:
            Dict: 검증된 메타데이터
        """
        validated = {}
        
        for key, value in metadata.items():
            # 키 정리
            clean_key = str(key).strip()
            if not clean_key:
                continue
            
            # 값 검증 및 변환
            if value is None:
                validated[clean_key] = ""
            elif isinstance(value, (int, float)):
                validated[clean_key] = value
            elif isinstance(value, bool):
                validated[clean_key] = value
            elif isinstance(value, (list, dict)):
                # 복잡한 타입은 문자열로 변환
                validated[clean_key] = str(value)
            else:
                # 문자열로 변환
                validated[clean_key] = str(value).strip()
        
        return validated
    
    def _normalize_mattress_prices(self, mattresses: List[Dict]) -> List[Dict]:
        """
        매트리스 데이터의 가격을 만원 단위로 정규화
        
        Args:
            mattresses: 원본 매트리스 데이터 리스트
            
        Returns:
            List[Dict]: 가격이 정규화된 매트리스 데이터
        """
        normalized_mattresses = []
        
        for mattress in mattresses:
            try:
                normalized_mattress = mattress.copy()
                
                # 원본 가격 (원 단위)
                original_price = mattress.get('price', 0)
                
                # 만원 단위로 변환
                price_manwon = self._convert_price_to_manwon(original_price)
                normalized_mattress['price'] = price_manwon
                
                # 원본 가격도 보관 (필요시 사용)
                normalized_mattress['price_won'] = int(float(original_price)) if original_price else 0
                
                # 표시용 가격 문자열
                if price_manwon >= 100:
                    normalized_mattress['price_display'] = f"{int(price_manwon)}만원"
                else:
                    normalized_mattress['price_display'] = f"{int(round(price_manwon))}만원"
                
                # 필수 필드 보장
                if 'name' not in normalized_mattress or not normalized_mattress['name']:
                    normalized_mattress['name'] = 'Unknown Mattress'
                
                if 'brand' not in normalized_mattress or not normalized_mattress['brand']:
                    normalized_mattress['brand'] = 'Unknown Brand'
                
                if 'type' not in normalized_mattress or not normalized_mattress['type']:
                    normalized_mattress['type'] = 'Unknown Type'
                
                normalized_mattresses.append(normalized_mattress)
                
            except Exception as e:
                logger.error(f"매트리스 데이터 정규화 실패: {e}")
                # 기본값으로 매트리스 추가
                default_mattress = {
                    'name': 'Unknown Mattress',
                    'brand': 'Unknown Brand',
                    'type': 'Unknown Type',
                    'price': 0.0,
                    'price_won': 0,
                    'price_display': '0만원'
                }
                normalized_mattresses.append(default_mattress)
        
        return normalized_mattresses

    def load_mattress_data(self) -> bool:
        """
        매트리스 데이터 로드 (ChromaDB 호환성 강화)
        """
        try:
            if not self.data_file.exists():
                logger.error(f"데이터 파일을 찾을 수 없습니다: {self.data_file}")
                return False
            
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터 구조에 따라 매트리스 리스트 추출
            if isinstance(data, dict):
                if 'mattresses' in data:
                    mattress_list = data['mattresses']
                elif 'data' in data:
                    mattress_list = data['data']
                else:
                    # 딕셔너리 자체가 하나의 매트리스인 경우
                    mattress_list = [data]
            elif isinstance(data, list):
                mattress_list = data
            else:
                logger.error("지원하지 않는 데이터 형식입니다")
                return False
            
            # 가격 정규화 적용
            self.mattresses = self._normalize_mattress_prices(mattress_list)
            
            logger.info(f"매트리스 데이터 로드 완료: {len(self.mattresses)}개")
            return True
            
        except Exception as e:
            logger.error(f"매트리스 데이터 로드 실패: {e}")
            return False

    def get_mattresses(self) -> List[Dict]:
        """
        로드된 매트리스 데이터 반환
        
        Returns:
            List[Dict]: 매트리스 데이터 리스트
        """
        return self.mattresses

    def get_mattress_by_id(self, mattress_id: str) -> Optional[Dict]:
        """
        ID로 특정 매트리스 조회
        
        Args:
            mattress_id: 매트리스 ID
            
        Returns:
            Optional[Dict]: 매트리스 정보, 없으면 None
        """
        # 정규화된 ID로 검색
        for mattress in self.mattresses:
            # 생성된 ID 패턴으로 검색
            generated_id = self._generate_unique_id(mattress, set())
            if generated_id == mattress_id:
                return mattress
        
        return None

    def preprocess_for_rag(self) -> List[Dict]:
        """
        RAG 시스템용 데이터 전처리 (ChromaDB 호환성 강화)
        """
        if not self.mattresses:
            logger.warning("매트리스 데이터가 없습니다")
            return []
        
        rag_data = []
        existing_ids = set()
        
        for i, mattress in enumerate(self.mattresses):
            try:
                # 고유 ID 생성
                mattress_id = self._generate_unique_id(mattress, existing_ids)
                existing_ids.add(mattress_id)
                
                # 검색용 텍스트 생성 (가격은 만원 단위 사용)
                search_text_parts = []
                
                # 기본 정보
                search_text_parts.append(f"매트리스 이름: {mattress.get('name', '')}")
                search_text_parts.append(f"브랜드: {mattress.get('brand', '')}")
                search_text_parts.append(f"타입: {mattress.get('type', '')}")
                search_text_parts.append(f"가격: {mattress.get('price', 0)}만원")  # 만원 단위
                
                # 특징
                features = mattress.get('features', [])
                if isinstance(features, list) and features:
                    search_text_parts.append(f"특징: {', '.join(str(f) for f in features)}")
                
                # 추천 사용자
                target_users = mattress.get('target_users', [])
                if isinstance(target_users, list) and target_users:
                    search_text_parts.append(f"추천 대상: {', '.join(str(t) for t in target_users)}")
                
                # 설명
                description = mattress.get('description', '')
                if description:
                    search_text_parts.append(f"설명: {description}")
                
                search_text = ' '.join(search_text_parts)
                
                # 메타데이터 준비 (ChromaDB 호환 타입으로)
                features_text = ', '.join(str(f) for f in features) if features else ''
                target_users_text = ', '.join(str(t) for t in target_users) if target_users else ''
                
                raw_metadata = {
                    'name': str(mattress.get('name', '')),
                    'brand': str(mattress.get('brand', '')),
                    'type': str(mattress.get('type', '')),
                    'price': float(mattress.get('price', 0)),  # 만원 단위
                    'price_won': int(mattress.get('price_won', 0)),  # 원 단위
                    'features_text': features_text,
                    'target_users_text': target_users_text,
                    'features_count': len(features) if features else 0,
                    'target_users_count': len(target_users) if target_users else 0
                }
                
                # 메타데이터 검증
                metadata = self._validate_metadata(raw_metadata)
                
                rag_data.append({
                    'id': mattress_id,
                    'search_text': search_text,
                    'metadata': metadata
                })
                
            except Exception as e:
                logger.error(f"매트리스 {i} 전처리 실패: {e}")
                # 기본 데이터로라도 추가
                fallback_id = f"mattress_fallback_{i}"
                rag_data.append({
                    'id': fallback_id,
                    'search_text': f"매트리스 {i}",
                    'metadata': {
                        'name': f'매트리스 {i}',
                        'brand': 'Unknown',
                        'type': 'Unknown',
                        'price': 0.0,
                        'price_won': 0,
                        'features_text': '',
                        'target_users_text': '',
                        'features_count': 0,
                        'target_users_count': 0
                    }
                })
        
        logger.info(f"RAG용 데이터 전처리 완료: {len(rag_data)}개 (ChromaDB 호환성 강화)")
        
        # 중복 ID 최종 확인
        ids = [item['id'] for item in rag_data]
        unique_ids = set(ids)
        if len(ids) != len(unique_ids):
            logger.warning(f"중복 ID 발견: {len(ids) - len(unique_ids)}개")
            
            # 중복 제거
            seen_ids = set()
            deduplicated_data = []
            for item in rag_data:
                if item['id'] not in seen_ids:
                    seen_ids.add(item['id'])
                    deduplicated_data.append(item)
            
            logger.info(f"중복 제거 후: {len(deduplicated_data)}개")
            return deduplicated_data
        
        return rag_data

    def get_statistics(self) -> Dict:
        """
        데이터 통계 정보 반환
        
        Returns:
            Dict: 통계 정보
        """
        if not self.mattresses:
            return {"error": "데이터가 로드되지 않았습니다"}
        
        try:
            # 가격 통계 (만원 단위)
            prices = [m.get('price', 0) for m in self.mattresses if isinstance(m.get('price'), (int, float))]
            
            # 브랜드 통계
            brands = [m.get('brand', '') for m in self.mattresses]
            brand_counts = {}
            for brand in brands:
                if brand:
                    brand_counts[brand] = brand_counts.get(brand, 0) + 1
            
            # 타입 통계
            types = [m.get('type', '') for m in self.mattresses]
            type_counts = {}
            for mattress_type in types:
                if mattress_type:
                    type_counts[mattress_type] = type_counts.get(mattress_type, 0) + 1
            
            return {
                'total_mattresses': len(self.mattresses),
                'price_stats': {
                    'min': min(prices) if prices else 0,
                    'max': max(prices) if prices else 0,
                    'avg': sum(prices) / len(prices) if prices else 0,
                    'unit': '만원'
                },
                'brand_distribution': brand_counts,
                'type_distribution': type_counts,
                'valid_prices': len(prices)
            }
        except Exception as e:
            logger.error(f"통계 생성 실패: {e}")
            return {"error": f"통계 생성 실패: {str(e)}"}


# 테스트 실행
if __name__ == "__main__":
    print("📊 매트리스 데이터 로더 테스트 (ChromaDB 호환성 강화)")
    print("=" * 60)
    
    try:
        # 데이터 로더 초기화
        loader = MattressDataLoader()
        
        # 데이터 로드
        if loader.load_mattress_data():
            # 통계 정보 출력
            stats = loader.get_statistics()
            print(f"\n✅ 데이터 로드 성공!")
            print(f"   총 매트리스: {stats['total_mattresses']}개")
            
            if 'price_stats' in stats:
                price_stats = stats['price_stats']
                print(f"   가격 범위: {price_stats['min']:.1f} ~ {price_stats['max']:.1f}만원")
                print(f"   평균 가격: {price_stats['avg']:.1f}만원")
                print(f"   유효 가격: {stats['valid_prices']}개")
            
            # 브랜드 분포
            if 'brand_distribution' in stats:
                print(f"\n📊 브랜드 분포 (상위 5개):")
                brand_items = list(stats['brand_distribution'].items())[:5]
                for brand, count in brand_items:
                    print(f"   {brand}: {count}개")
            
            # RAG 전처리 테스트
            print(f"\n🔍 RAG 전처리 테스트:")
            rag_data = loader.preprocess_for_rag()
            
            if rag_data:
                print(f"   처리된 문서: {len(rag_data)}개")
                
                # ID 중복 검사
                ids = [item['id'] for item in rag_data]
                unique_ids = set(ids)
                print(f"   고유 ID: {len(unique_ids)}개")
                
                if len(ids) == len(unique_ids):
                    print(f"   ✅ ID 중복 없음")
                else:
                    print(f"   ⚠️ ID 중복 있음: {len(ids) - len(unique_ids)}개")
                
                # 샘플 데이터 출력
                sample_rag = rag_data[0]
                print(f"\n   샘플 데이터:")
                print(f"   ID: {sample_rag['id']}")
                print(f"   검색 텍스트 길이: {len(sample_rag['search_text'])}자")
                print(f"   메타데이터 키: {list(sample_rag['metadata'].keys())}")
                
                # 메타데이터 타입 검증
                metadata = sample_rag['metadata']
                print(f"\n   메타데이터 타입 검증:")
                for key, value in metadata.items():
                    print(f"   {key}: {type(value).__name__} = {value}")
            
            print(f"\n✅ ChromaDB 호환성 테스트 완료!")
            
        else:
            print("❌ 데이터 로드 실패")
            print("\n💡 해결 방법:")
            print("1. data/mattress_data.json 파일이 있는지 확인")
            print("2. 없다면 먼저 generate_data.py를 실행하세요:")
            print("   python src/generate_data.py")
        
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 문제 해결:")
        print(f"1. 현재 디렉토리 확인: {Path.cwd()}")
        print(f"2. 데이터 파일 경로 확인")
        print(f"3. JSON 파일 형식 검증")