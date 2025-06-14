import json
import random
import os
from typing import List, Dict

class MattressDataAugmentor:
    def __init__(self):
        # 브랜드 데이터
        self.brands = [
            "IKEA", "오늘의집", "한샘", "쿠팡", "템퍼", "슬리페이스", "코오롱", "에이스", 
            "퍼플", "캐스퍼", "씰리", "사타", "던롭", "심몬스", "킹코일", "라텍스존", 
            "수면공감", "데코뷰", "홈앤하우스", "리바트", "현대리바트", "일룸", "퍼시스",
            "넘버베드", "코코매트", "네이처라텍스", "바디가드", "슬리프넘버", "튜프트앤니들",
            "헬릭스", "레이시", "브룩사이드", "지누스", "모디웨이", "루시드", "베스트프라이스"
        ]
        
        # 매트리스 타입
        self.types = ["스프링", "메모리폼", "라텍스", "하이브리드", "젤메모리폼", "폴리우레탄폼"]
        
        # 경도
        self.firmness_levels = ["소프트", "소프트-미디움", "미디움", "미디움-하드", "하드"]
        
        # 사이즈
        self.sizes_options = [
            ["싱글"],
            ["싱글", "더블"],
            ["싱글", "더블", "퀸"],
            ["싱글", "더블", "퀸", "킹"],
            ["더블", "퀸"],
            ["더블", "퀸", "킹"],
            ["퀸", "킹"]
        ]
        
        # 두께 (cm)
        self.thickness_range = range(15, 31)
        
        # 소재
        self.materials = [
            "독립 포켓스프링", "메모리폼", "천연라텍스", "폴리우레탄폼", "고밀도폼",
            "젤메모리폼", "본넬스프링", "코코넛 코이어", "쿨링젤", "고밀도 메모리폼",
            "서포트 폼", "트랜지션 폼", "오픈셀 폼", "유기농 커버", "하이퍼일라스틱 젤"
        ]
        
        # 특징
        self.features = [
            "통기성 우수", "체압분산", "독립스프링 1000개", "항균처리", "온도조절",
            "움직임 차단", "에지서포트", "허리 지지력", "쿨링젤", "저소음 스프링",
            "진동흡수", "소음 차단", "고밀도 폼", "라텍스층", "무소음", "온도감응",
            "탁월한 탄성", "항진드기", "존별 지지력", "3D 메쉬", "SmartClimate 커버",
            "젤 그리드 기술", "4층 구조", "100일 체험", "온라인 직판", "NASA 기술"
        ]
        
        # 추천 대상
        self.recommended_for = [
            "평균 체중", "모든 수면자세", "커플", "측면수면", "허리통증", "관절염",
            "알레르기 체질", "천연 소재 선호", "예산 제한", "학생", "성장기", "기숙사",
            "혁신 기술 선호", "통기성 중요", "더위 타는 분", "온라인 구매 선호",
            "밸런스 중요", "체험 중요시", "프리미엄 추구", "만성 통증", "최고 품질 원하는 분",
            "장시간 숙면", "열대야에 강함", "허리통증 완화"
        ]
        
        # 비추천 대상
        self.not_recommended_for = [
            "매우 가벼운 체중", "매우 무거운 체중", "라텍스 알레르기", "더위를 많이 타는 분",
            "딱딱한 매트리스 선호", "부드러운 매트리스 선호", "소음에 민감한 분",
            "고급 기능 원하는 분", "예산 제한", "전통적 느낌 선호", "적응 시간 싫어하는 분",
            "매장 체험 필수", "극단적 단단함/부드러움 선호", "추위 많이 타는 분",
            "복부 수면자", "과민성 피부", "예민한 체형"
        ]
        
        # 장점
        self.pros = [
            "긴 보증기간", "우수한 통기성", "적당한 가격", "우수한 체압분산", "움직임 전달 없음",
            "허리 지지력", "천연 소재", "우수한 내구성", "항균 효과", "환경 친화적",
            "저렴한 가격", "단단한 지지력", "빠른 배송", "최고급 메모리폼", "탁월한 체압분산",
            "브랜드 신뢰도", "두 소재의 장점 결합", "존별 차별화", "우수한 온도조절",
            "메모리폼의 단점 보완", "매우 저렴", "성장기에 적합한 단단함", "가벼움",
            "혁신적 기술", "최고의 통기성", "독특한 느낌", "밸런스 좋은 느낌",
            "온라인 편의성", "긴 체험기간", "진동 억제", "지지력 우수", "엣지서포트 강화"
        ]
        
        # 단점
        self.cons = [
            "초기 냄새", "가장자리 지지력 부족", "초기 적응기간", "열 보유", "가격대",
            "높은 가격", "무게가 무거움", "초기 라텍스 냄새", "파트너 움직임 전달",
            "스프링 소음", "내구성 한계", "매우 높은 가격", "무거움", "적응 기간 필요",
            "복잡한 구조", "기본적인 기능", "편의 기능 없음", "호불호 갈림",
            "평범함", "높은 기대치", "배송비", "라텍스 냄새", "가격 다소 높음",
            "배송 지연", "겨울철 차가움", "무게 무거움"
        ]
        
        # 사용자 리뷰 템플릿
        self.review_templates = [
            "전반적으로 만족도가 높으며 가성비가 우수하다는 평가",
            "허리 통증이 개선되었다는 후기가 많으며, 수면의 질이 향상됨",
            "천연 소재에 대한 만족도가 높고, 오래 사용해도 변형이 적음",
            "가격 대비 괜찮다는 평가이지만, 장기 사용 시 한계 있음",
            "가격은 비싸지만 수면의 질이 확실히 다르다는 평가",
            "스프링의 탄성과 메모리폼의 편안함을 동시에 느낄 수 있어 만족",
            "여름에도 시원하게 잘 수 있어서 만족도가 높음",
            "학생용으로는 가격 대비 괜찮다는 평가",
            "독특한 느낌이지만 통기성과 지지력이 뛰어나다는 평가",
            "무난하게 좋다는 평가가 많으며, 대부분의 사람에게 적합",
            "장시간 사용 후에도 변형이 거의 없어 오랫동안 사용할 수 있을 것 같습니다",
            "오래 사용해도 꺼짐이 거의 없어 처음 구입했을 때의 느낌이 유지됩니다",
            "수면 중에 자세를 자주 바꾸는데도 체압 분산이 잘 되어 몸이 쉽게 피로하지 않습니다",
            "처음에는 조금 단단하게 느껴졌지만 사용할수록 몸에 맞게 적응되어 매우 편안합니다",
            "소프트하면서도 허리를 잘 받쳐줘 장시간 누워 있어도 불편함이 없습니다",
            "소음이 거의 없어 파트너의 뒤척임에도 깨지 않고 푹 잘 수 있습니다",
            "가격 대비 품질이 정말 만족스럽고 지인에게도 추천하고 싶은 매트리스입니다"
        ]

    def generate_price(self, brand: str, mattress_type: str, firmness: str) -> int:
        """브랜드와 타입에 따른 현실적인 가격 생성"""
        base_prices = {
            "템퍼": (800000, 1500000),
            "퍼플": (700000, 1200000),
            "캐스퍼": (400000, 800000),
            "심몬스": (600000, 1000000),
            "씰리": (500000, 900000),
            "사타": (400000, 700000),
            "한샘": (300000, 600000),
            "슬리페이스": (350000, 650000),
            "IKEA": (150000, 300000),
            "오늘의집": (200000, 500000),
            "쿠팡": (100000, 250000),
            "에이스": (300000, 4000000)
        }
        
        # 브랜드별 기본 가격 범위
        if brand in base_prices:
            min_price, max_price = base_prices[brand]
        else:
            min_price, max_price = (200000, 600000)  # 기본값
        
        # 타입별 가격 조정
        type_multipliers = {
            "하이브리드": 1.3,
            "젤메모리폼": 1.2,
            "메모리폼": 1.1,
            "라텍스": 1.15,
            "스프링": 0.8,
            "폴리우레탄폼": 0.7
        }
        
        multiplier = type_multipliers.get(mattress_type, 1.0)
        min_price = int(min_price * multiplier)
        max_price = int(max_price * multiplier)
        
        # 최소 가격 보장 (10만원 이상)
        min_price = max(min_price, 100000)
        max_price = max(max_price, min_price + 50000)
        
        # 천원 단위로 반올림
        price = random.randint(min_price, max_price)
        price = round(price, -3)
        
        # 최종 검증 - 가격이 0이 되지 않도록
        return max(price, 100000)

    def generate_warranty(self, price: int) -> str:
        """가격대에 따른 보증기간 생성"""
        if price < 200000:
            return random.choice(["2년", "3년", "5년"])
        elif price < 500000:
            return random.choice(["5년", "8년", "10년"])
        elif price < 1000000:
            return random.choice(["10년", "12년", "15년"])
        else:
            return random.choice(["15년", "20년", "25년"])

    def generate_mattress_name(self, brand: str, mattress_type: str) -> str:
        """브랜드와 타입에 따른 매트리스 이름 생성"""
        
        # 브랜드별 네이밍 패턴
        name_patterns = {
            "IKEA": ["HÄFSLO", "MALFORS", "MORGEDAL", "MYRBACKA", "HIDRASUND"],
            "템퍼": ["오리지널", "클라우드", "프로", "엘리트", "프리미엄"],
            "에이스": ["HYBRID TECH", "ROYAL ACE", "ACE TIME", "ACE WHITE", "THE PRIME", "PLATINUM"],
            "캐스퍼": ["오리지널", "웨이브", "노바", "엘리먼트", "에센셜"],
            "퍼플": ["그리드", "하이브리드", "프리미어", "리스토어", "플러스"]
        }
        
        # 일반적인 형용사와 명사
        adjectives = ["프리미엄", "컴포트", "디럭스", "엘리트", "클래식", "모던", "스마트", "네이처", 
                     "퍼펙트", "슈프림", "마스터", "플래티넘", "골드", "실버", "블랙", "화이트",
                     "소프트", "하드", "밸런스", "케어", "헬스", "드림", "클라우드", "테크"]
        
        nouns = ["매트리스", "베드", "슬립", "컴포트", "라인", "시리즈", "에디션", "컬렉션",
                "프로", "플러스", "맥스", "미니", "라이트", "헤비"]
        
        if brand in name_patterns:
            base_name = random.choice(name_patterns[brand])
        else:
            base_name = random.choice(adjectives)
        
        # 추가 수식어나 명사 붙이기 (50% 확률)
        if random.random() < 0.5:
            if random.random() < 0.5:
                return f"{brand} {base_name} {random.choice(adjectives)}"
            else:
                return f"{brand} {base_name} {random.choice(nouns)}"
        else:
            return f"{brand} {base_name} 매트리스"

    def generate_description(self, name: str, features: List[str], mattress_type: str) -> str:
        """특징에 맞는 설명 생성"""
        descriptions = [
            f"{', '.join(features[:2])}으로 구성된 매트리스. 우수한 통기성과 적당한 지지력을 제공하며, 장기간 사용 가능.",
            f"{mattress_type} 기술을 바탕으로 한 프리미엄 매트리스. 완벽한 체압분산과 온도감응 기능으로 최상의 수면 경험을 제공.",
            f"혁신적인 {mattress_type} 구조로 몸의 곡선에 완벽하게 맞춰주며, 체압을 고르게 분산시켜 편안한 수면을 제공.",
            f"다층 구조의 지지 시스템이 신체 각 부위의 하중을 분산시키고, 오랜 시간 사용해도 꺼짐 현상이 거의 없습니다.",
            f"친환경 인증 소재를 사용하여 피부에 자극이 적고, 오랜 사용에도 형태가 변형되지 않는 우수한 복원력을 자랑합니다.",
            f"허리 지지력을 강화한 설계로 척추의 자연스러운 곡선을 유지하며, 체압 분산 기능으로 어깨와 엉덩이에 가해지는 부담을 줄여줍니다.",
            f"진동 차단과 소음 감소 기능이 적용되어 파트너의 뒤척임에도 방해받지 않는 깊은 수면이 가능합니다.",
            f"에지 서포트가 강화되어 침대 가장자리에서도 균형 잡힌 지지력을 제공하고, 넓은 수면 공간을 활용할 수 있도록 설계되었습니다."
        ]
        return random.choice(descriptions)

    def generate_mattress(self, index: int) -> Dict:
        """단일 매트리스 데이터 생성"""
        brand = random.choice(self.brands)
        mattress_type = random.choice(self.types)
        firmness = random.choice(self.firmness_levels)
        thickness = f"{random.choice(self.thickness_range)}cm"
        sizes = random.choice(self.sizes_options)
        
        # 가격 생성 (최소값 보장)
        price = self.generate_price(brand, mattress_type, firmness)
        
        # 보증기간 생성
        warranty = self.generate_warranty(price)
        
        # 이름 생성
        name = self.generate_mattress_name(brand, mattress_type)
        
        # 특징 선택 (3-6개)
        features = random.sample(self.features, random.randint(3, 6))
        
        # 소재 선택 (2-4개)
        materials = random.sample(self.materials, random.randint(2, 4))
        
        # 추천/비추천 대상 선택
        recommended = random.sample(self.recommended_for, random.randint(1, 3))
        not_recommended = random.sample(self.not_recommended_for, random.randint(1, 3))
        
        # 장단점 선택
        pros = random.sample(self.pros, random.randint(2, 4))
        cons = random.sample(self.cons, random.randint(2, 3))
        
        # 사용자 리뷰 선택
        user_review = random.choice(self.review_templates)
        
        # 설명 생성
        description = self.generate_description(name, features, mattress_type)
        
        # 데이터 검증
        mattress_data = {
            "id": f"mattress_{index:05d}",
            "name": name,
            "brand": brand,
            "type": mattress_type,
            "price": price,
            "size": sizes,
            "firmness": firmness,
            "thickness": thickness,
            "features": features,
            "recommended_for": recommended,
            "not_recommended_for": not_recommended,
            "warranty": warranty,
            "materials": materials,
            "pros": pros,
            "cons": cons,
            "user_reviews": user_review,
            "description": description
        }
        
        # 최종 검증
        if mattress_data["price"] <= 0:
            mattress_data["price"] = 150000  # 기본값 설정
            
        return mattress_data

    def generate_dataset(self, num_mattresses: int = 10000) -> Dict:
        """전체 데이터셋 생성"""
        mattresses = []
        
        print(f"매트리스 데이터 {num_mattresses}건 생성 중...")
        
        for i in range(num_mattresses):
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}건 완료...")
            
            mattress = self.generate_mattress(i + 1)
            mattresses.append(mattress)
        
        # 대폭 확장된 구매 가이드
        buying_guide = {
            "by_sleep_position": {
                "side": {
                    "recommended_types": ["메모리폼", "라텍스", "소프트 하이브리드", "젤메모리폼"],
                    "firmness": "소프트-미디움",
                    "thickness": "23-28cm",
                    "key_features": ["체압분산", "어깨/엉덩이 압력완화", "척추정렬"],
                    "avoid": ["너무 단단한 매트리스", "얇은 매트리스(20cm 이하)"],
                    "reason": "어깨와 엉덩이의 압력을 분산시키고 척추를 정렬하기 위해",
                    "additional_tips": [
                        "베개 높이도 중요 - 목과 척추가 일직선이 되도록",
                        "무릎 사이에 쿠션을 끼우면 허리 부담 감소",
                        "임산부는 더욱 부드러운 매트리스 권장"
                    ]
                },
                "back": {
                    "recommended_types": ["하이브리드", "라텍스", "미디움 메모리폼", "독립스프링"],
                    "firmness": "미디움-펌",
                    "thickness": "20-25cm",
                    "key_features": ["척추지지", "적절한 탄성", "존별지지"],
                    "avoid": ["너무 부드러운 매트리스", "푹 꺼지는 매트리스"],
                    "reason": "척추의 자연스러운 S커브를 유지하기 위해",
                    "additional_tips": [
                        "허리 아래 작은 쿠션으로 지지력 보강 가능",
                        "무릎 아래 베개로 허리 곡선 유지",
                        "아침에 허리 통증이 있다면 더 단단한 매트리스 고려"
                    ]
                },
                "stomach": {
                    "recommended_types": ["스프링", "펌 하이브리드", "라텍스", "고밀도 폼"],
                    "firmness": "펌-하드",
                    "thickness": "18-23cm",
                    "key_features": ["강한 지지력", "허리 처짐 방지", "통기성"],
                    "avoid": ["메모리폼", "너무 부드러운 매트리스", "두꺼운 매트리스"],
                    "reason": "허리가 과도하게 굽어지는 것을 방지하기 위해",
                    "additional_tips": [
                        "베개는 낮거나 아예 사용하지 않는 것이 좋음",
                        "배 아래 얇은 베개로 허리 지지",
                        "목 부담을 줄이기 위해 다른 수면자세 연습 권장"
                    ]
                },
                "combination": {
                    "recommended_types": ["하이브리드", "미디움 라텍스", "존별 지지 매트리스"],
                    "firmness": "미디움",
                    "thickness": "22-26cm",
                    "key_features": ["적응성", "균형잡힌 지지", "동작 흡수"],
                    "avoid": ["극단적으로 단단하거나 부드러운 매트리스"],
                    "reason": "다양한 자세에 유연하게 적응할 수 있도록",
                    "additional_tips": [
                        "각 자세별 최적화보다는 균형을 중시",
                        "파트너와 선호 자세가 다르다면 이 타입 권장",
                        "적응 기간을 충분히 가져볼 것"
                    ]
                }
            },
            "by_body_type": {
                "light": {
                    "weight_range": "50kg 이하",
                    "recommended_firmness": "소프트-미디움",
                    "recommended_types": ["메모리폼", "라텍스", "젤메모리폼"],
                    "thickness": "20-25cm",
                    "key_concerns": ["충분한 압력완화", "몸 윤곽 추종", "온도조절"],
                    "avoid": ["너무 단단한 매트리스", "스프링이 느껴지는 매트리스"],
                    "reason": "충분한 컨투어링과 압력 완화 필요",
                    "budget_recommendations": {
                        "budget": "오늘의집 메모리폼, 코오롱 쿨젤",
                        "mid_range": "한샘 라텍스, IKEA HÄFSLO",
                        "premium": "템퍼 오리지널, 캐스퍼 웨이브"
                    }
                },
                "average": {
                    "weight_range": "50-80kg",
                    "recommended_firmness": "미디움",
                    "recommended_types": ["하이브리드", "라텍스", "메모리폼", "독립스프링"],
                    "thickness": "22-26cm",
                    "key_concerns": ["지지력과 편안함 균형", "내구성", "온도조절"],
                    "avoid": ["극단적 경도의 매트리스"],
                    "reason": "지지력과 편안함의 균형 필요",
                    "budget_recommendations": {
                        "budget": "쿠팡 독립스프링, 에이스 기본형",
                        "mid_range": "슬리페이스 하이브리드, 한샘 라텍스",
                        "premium": "퍼플 그리드, 템퍼 클라우드"
                    }
                },
                "heavy": {
                    "weight_range": "80kg 이상",
                    "recommended_firmness": "미디움펌-펌",
                    "recommended_types": ["하이브리드", "스프링", "고밀도 폼", "라텍스"],
                    "thickness": "25-30cm",
                    "key_concerns": ["강한 지지력", "내구성", "엣지서포트", "깊은 침몰 방지"],
                    "avoid": ["얇은 매트리스", "저밀도 폼", "너무 부드러운 메모리폼"],
                    "reason": "충분한 지지력과 내구성 필요",
                    "budget_recommendations": {
                        "budget": "에이스 하드타입, 쿠팡 본넬스프링",
                        "mid_range": "슬리페이스 펌 하이브리드, 씰리 독립스프링",
                        "premium": "심몬스 뷰티레스트, 템퍼 프로"
                    }
                },
                "couple_different_weights": {
                    "weight_range": "파트너 간 체중차 20kg 이상",
                    "recommended_firmness": "미디움 또는 존별 지지",
                    "recommended_types": ["존별 지지 하이브리드", "분리형 매트리스", "고급 메모리폼"],
                    "thickness": "25-28cm",
                    "key_concerns": ["개별 지지력", "동작 격리", "엣지서포트"],
                    "avoid": ["본넬스프링", "균일한 경도의 매트리스"],
                    "reason": "각자에게 맞는 지지력을 제공하면서 동작 전달 최소화",
                    "special_options": ["분리형 매트리스", "커스텀 존별 하이브리드", "조절 가능한 매트리스"]
                }
            },
            "by_budget": {
                "ultra_budget": {
                    "range": "5-15만원",
                    "options": ["기본 본넬스프링", "얇은 폴리우레탄폼", "학생용 매트리스"],
                    "brands": ["쿠팡 기본형", "에이스 학생용", "일부 온라인 브랜드"],
                    "pros": ["저렴한 가격", "임시 사용 적합"],
                    "cons": ["내구성 한계", "편안함 부족", "짧은 보증기간"],
                    "recommended_for": ["학생", "원룸", "임시 거주", "게스트룸"],
                    "buying_tips": ["보증기간 확인", "배송비 포함 가격 비교", "후기 꼼꼼히 확인"]
                },
                "budget": {
                    "range": "15-30만원",
                    "options": ["독립스프링", "기본 메모리폼", "얇은 라텍스"],
                    "brands": ["쿠팡", "에이스", "IKEA", "오늘의집 기본형"],
                    "pros": ["합리적 가격", "기본 기능 충족", "다양한 선택지"],
                    "cons": ["고급 기능 부족", "내구성 아쉬움"],
                    "recommended_for": ["신혼부부", "첫 독립", "예산 제한"],
                    "buying_tips": ["체험 기간 활용", "A/S 정책 확인", "리뷰 신뢰도 검증"]
                },
                "mid_range": {
                    "range": "30-60만원",
                    "options": ["하이브리드", "고품질 메모리폼", "천연라텍스", "젤메모리폼"],
                    "brands": ["한샘", "슬리페이스", "오늘의집", "코오롱", "IKEA 프리미엄"],
                    "pros": ["균형잡힌 성능", "적절한 내구성", "다양한 기능"],
                    "cons": ["최고급 기능은 아님"],
                    "recommended_for": ["대부분의 성인", "장기 사용 계획", "품질과 가격 균형 추구"],
                    "buying_tips": ["여러 브랜드 비교", "매장 체험 필수", "할인 시기 노려보기"]
                },
                "premium": {
                    "range": "60-120만원",
                    "options": ["고급 하이브리드", "프리미엄 메모리폼", "100% 천연라텍스"],
                    "brands": ["템퍼", "캐스퍼", "퍼플", "씰리", "사타"],
                    "pros": ["뛰어난 품질", "긴 보증기간", "우수한 기능"],
                    "cons": ["높은 가격", "오버스펙 가능성"],
                    "recommended_for": ["수면 품질 중시", "건강 문제", "장기 투자"],
                    "buying_tips": ["체험 기간 최대 활용", "할부 혜택 확인", "추가 서비스 포함 여부"]
                },
                "luxury": {
                    "range": "120만원 이상",
                    "options": ["최고급 하이브리드", "커스텀 매트리스", "스마트 매트리스"],
                    "brands": ["심몬스", "템퍼 최고급형", "커스텀 브랜드"],
                    "pros": ["최고 품질", "개인 맞춤", "평생 보증 가능"],
                    "cons": ["매우 높은 가격", "필요성 검토 필요"],
                    "recommended_for": ["최고급 추구", "특별한 요구사항", "건강상 필수"],
                    "buying_tips": ["전문가 상담", "장기 A/S 계획", "투자 가치 신중 검토"]
                }
            },
            "by_age_group": {
                "children": {
                    "age_range": "3-12세",
                    "recommended_firmness": "미디움-펌",
                    "recommended_types": ["독립스프링", "라텍스", "고밀도 폼"],
                    "key_features": ["성장 지원", "항균", "안전소재", "적절한 지지력"],
                    "avoid": ["너무 부드러운 매트리스", "화학 냄새 나는 제품"],
                    "special_considerations": ["성장기 척추 건강", "알레르기 예방", "안전성"],
                    "size_recommendations": ["싱글", "작은 더블"],
                    "replacement_cycle": "5-7년 (성장에 따라 조정)"
                },
                "teenagers": {
                    "age_range": "13-19세",
                    "recommended_firmness": "미디움",
                    "recommended_types": ["하이브리드", "메모리폼", "라텍스"],
                    "key_features": ["성장 지원", "체압분산", "온도조절", "내구성"],
                    "avoid": ["너무 저렴한 제품", "성인용 과도한 기능"],
                    "special_considerations": ["급속 성장기", "수면 패턴 변화", "활동량 많음"],
                    "size_recommendations": ["싱글", "더블"],
                    "replacement_cycle": "7-10년"
                },
                "young_adults": {
                    "age_range": "20-35세",
                    "recommended_firmness": "개인 선호에 따라",
                    "recommended_types": ["모든 타입", "온라인 브랜드 고려"],
                    "key_features": ["개성 반영", "가성비", "온라인 구매 편의"],
                    "avoid": ["과도한 마케팅", "검증되지 않은 신제품"],
                    "special_considerations": ["라이프스타일 변화", "파트너 고려", "이사 빈도"],
                    "size_recommendations": ["더블", "퀸"],
                    "replacement_cycle": "8-12년"
                },
                "middle_aged": {
                    "age_range": "36-55세",
                    "recommended_firmness": "미디움-펌",
                    "recommended_types": ["하이브리드", "메모리폼", "라텍스"],
                    "key_features": ["건강 지원", "체압분산", "파트너 배려", "내구성"],
                    "avoid": ["유행만 좇는 제품", "과도한 실험"],
                    "special_considerations": ["직업병 예방", "수면의 질", "스트레스 해소"],
                    "size_recommendations": ["퀸", "킹"],
                    "replacement_cycle": "10-15년"
                },
                "seniors": {
                    "age_range": "55세 이상",
                    "recommended_firmness": "미디움 (관절 상태에 따라)",
                    "recommended_types": ["메모리폼", "라텍스", "하이브리드"],
                    "key_features": ["관절 보호", "체압분산", "진입 용이성", "안전성"],
                    "avoid": ["너무 낮거나 높은 매트리스", "복잡한 기능"],
                    "special_considerations": ["관절염", "혈액순환", "기상 편의성"],
                    "size_recommendations": ["퀸", "킹", "조절 가능한 베드"],
                    "replacement_cycle": "15-20년"
                }
            },
            "special_needs": {
                "back_pain": {
                    "recommended": ["메모리폼", "라텍스", "하이브리드", "조절형 매트리스"],
                    "firmness": "미디움 (개인차 있음)",
                    "features": ["체압분산", "척추정렬", "존별지지", "적응성"],
                    "medical_advice": "심한 경우 의사 상담 후 선택",
                    "trial_period": "최소 3개월 체험 권장",
                    "complementary": ["적절한 베개", "스트레칭", "수면자세 교정"]
                },
                "neck_pain": {
                    "recommended": ["메모리폼", "라텍스", "적응형 하이브리드"],
                    "firmness": "미디움-소프트",
                    "features": ["목과 어깨 지지", "압력 완화", "온도 중립"],
                    "pillow_matching": "매트리스와 베개의 조화 중요",
                    "sleep_position": "측면 수면 권장",
                    "avoid": ["너무 단단한 매트리스", "목이 꺾이는 높이"]
                },
                "arthritis": {
                    "recommended": ["메모리폼", "젤메모리폼", "라텍스"],
                    "firmness": "소프트-미디움",
                    "features": ["관절 압력 완화", "온도 조절", "항염 소재"],
                    "special_care": ["아침 경직 완화", "혈액순환 도움"],
                    "avoid": ["스프링이 느껴지는 매트리스", "너무 단단한 매트리스"],
                    "additional": ["온열 매트리스 토퍼 고려", "정기적인 운동 병행"]
                },
                "fibromyalgia": {
                    "recommended": ["메모리폼", "라텍스", "압력완화 특화 매트리스"],
                    "firmness": "개인차 매우 큼 (체험 필수)",
                    "features": ["전신 압력 분산", "온도 안정성", "동작 격리"],
                    "sensitivity": "화학 냄새에 민감할 수 있음",
                    "trial_importance": "충분한 체험 기간 절대 필요",
                    "complementary": ["스트레스 관리", "규칙적 수면패턴"]
                },
                "hot_sleeper": {
                    "recommended": ["하이브리드", "라텍스", "젤메모리폼", "스프링"],
                    "features": ["통기성", "쿨링젤", "온도조절", "수분 증발"],
                    "avoid": ["기본 메모리폼", "밀도 높은 폼", "합성 커버"],
                    "room_environment": ["적절한 실내온도", "통풍", "습도 조절"],
                    "bedding": ["통기성 좋은 침구", "천연소재 시트"],
                    "seasonal": ["여름철 추가 쿨링 제품 고려"]
                },
                "cold_sleeper": {
                    "recommended": ["메모리폼", "라텍스", "밀도 높은 폼"],
                    "features": ["체온 보존", "열 저장", "보온성"],
                    "avoid": ["젤메모리폼", "과도한 통기성", "스프링 단독"],
                    "bedding": ["보온성 좋은 침구", "울소재 고려"],
                    "mattress_topper": ["보온 기능 토퍼 추가 고려"],
                    "health_check": ["갑상선 기능 등 건강 검진 고려"]
                },
                "couple": {
                    "recommended": ["메모리폼", "하이브리드", "포켓스프링"],
                    "features": ["motion isolation", "엣지서포트", "다양한 사이즈", "정온성"],
                    "size": "퀸 이상 권장 (킹 최적)",
                    "firmness": "두 사람의 선호도 타협점 찾기",
                    "special_solutions": ["분리형 매트리스", "조절형 베드", "존별 지지"],
                    "trial_together": "두 사람이 함께 체험해볼 것"
                },
                "pregnancy": {
                    "recommended": ["메모리폼", "라텍스", "조절형 매트리스"],
                    "firmness": "미디움-소프트 (시기별 조정)",
                    "features": ["체압분산", "측면 지지", "안전 소재", "적응성"],
                    "special_support": ["복부 지지", "허리 보호", "혈액순환"],
                    "avoid": ["화학물질", "너무 단단한 매트리스", "높은 온도"],
                    "accessories": ["임산부 베개", "측면 지지 쿠션"]
                },
                "allergies": {
                    "recommended": ["라텍스", "하이브리드", "항균 처리 매트리스"],
                    "features": ["항균", "항진드기", "천연소재", "통기성"],
                    "avoid": ["화학 처리된 폼", "밀도 높은 소재", "습기 보존 소재"],
                    "certifications": ["친환경 인증", "알레르기 테스트 완료 제품"],
                    "maintenance": ["정기적 청소", "방수 커버 사용", "환기"],
                    "room_care": ["실내 습도 조절", "공기청정기", "정기 청소"]
                },
                "insomnia": {
                    "recommended": ["메모리폼", "라텍스", "온도조절 매트리스"],
                    "features": ["편안함", "온도 안정성", "동작 격리", "압력 완화"],
                    "avoid": ["너무 자극적인 소재", "온도 변화 큰 매트리스"],
                    "sleep_hygiene": ["일정한 수면시간", "수면 환경 조성"],
                    "relaxation": ["스트레스 관리", "이완 기법"],
                    "medical": "심한 경우 수면 전문의 상담"
                }
            },
            "mattress_care": {
                "daily_care": [
                    "매일 환기시키기 (침구 걷어두기)",
                    "습도 조절 (50-60% 유지)",
                    "직사광선 피하기",
                    "무거운 물건 올려두지 않기"
                ],
                "weekly_care": [
                    "시트와 커버 세탁",
                    "매트리스 표면 청소기로 청소",
                    "뒤집기 또는 회전 (가능한 경우)",
                    "방수 커버 점검"
                ],
                "monthly_care": [
                    "매트리스 전체 점검",
                    "얼룩이나 손상 확인",
                    "받침대나 프레임 점검",
                    "방습제 교체"
                ],
                "seasonal_care": [
                    "계절별 침구 교체",
                    "매트리스 깊은 청소",
                    "보관용품 점검",
                    "프레임 윤활 및 점검"
                ],
                "stain_removal": {
                    "blood": "찬물과 과산화수소 사용",
                    "urine": "식초와 베이킹소다 활용",
                    "sweat": "효소 세제로 처리",
                    "food": "즉시 제거 후 중성세제"
                },
                "odor_removal": [
                    "베이킹소다 뿌리고 몇 시간 후 청소기로 제거",
                    "활성탄 주머니 사용",
                    "오존 처리 (전문업체)",
                    "충분한 환기"
                ]
            },
            "replacement_signs": [
                "7-10년 이상 사용 (일반적 교체 주기)",
                "아침에 일어날 때 몸이 아픔",
                "매트리스에 꺼진 부분이나 덩어리",
                "스프링이 느껴지거나 소음 발생",
                "알레르기 증상 악화",
                "파트너의 움직임이 과도하게 전달됨",
                "수면의 질 현저히 저하",
                "호텔에서 더 잘 잤다는 느낌",
                "매트리스 냄새나 알레르기 반응"
            ],
            "buying_process": {
                "research_phase": {
                    "duration": "2-4주 권장",
                    "steps": [
                        "개인 수면 패턴 및 선호도 파악",
                        "예산 설정 및 우선순위 정하기",
                        "온라인 리뷰 및 전문가 의견 수집",
                        "후보 매트리스 3-5개 선정",
                        "매장 방문 계획 수립"
                    ],
                    "key_questions": [
                        "주로 어떤 자세로 주무시나요?",
                        "현재 매트리스의 불만점은?",
                        "파트너와 함께 사용하시나요?",
                        "알레르기나 특별한 건강 상태가 있나요?",
                        "예산 범위는 어느 정도인가요?"
                    ]
                },
                "testing_phase": {
                    "duration": "1-2주",
                    "offline_testing": [
                        "매장에서 최소 15분 이상 누워보기",
                        "평소 수면 자세로 테스트",
                        "파트너와 함께 테스트 (해당시)",
                        "다양한 브랜드 비교 체험",
                        "판매직원 상담 받기"
                    ],
                    "online_consideration": [
                        "체험 기간 정책 확인",
                        "반품 조건 및 비용 파악",
                        "배송 및 설치 서비스 확인",
                        "고객 서비스 품질 평가"
                    ]
                },
                "decision_phase": {
                    "final_checklist": [
                        "예산 내 최적 선택인가?",
                        "체험 결과 만족스러운가?",
                        "보증 및 A/S 조건 적절한가?",
                        "배송 및 설치 일정 확인",
                        "기존 매트리스 처리 계획"
                    ],
                    "red_flags": [
                        "과도한 할인 압박",
                        "체험 거부 또는 제한",
                        "불분명한 보증 조건",
                        "과장된 건강 효과 주장",
                        "검증되지 않은 신기술"
                    ]
                }
            },
            "seasonal_buying_guide": {
                "spring": {
                    "best_for": ["알레르기 관리", "새학기 준비", "이사철"],
                    "promotions": ["새학기 할인", "봄맞이 프로모션"],
                    "considerations": ["꽃가루 알레르기", "습도 변화", "환기 중요성"],
                    "recommended_features": ["항균", "통기성", "항알레르기"]
                },
                "summer": {
                    "best_for": ["더위 대책", "휴가철 준비"],
                    "promotions": ["여름 세일", "쿨링 매트리스 특가"],
                    "considerations": ["고온다습", "에어컨 사용", "통풍 중요"],
                    "recommended_features": ["쿨링젤", "통기성", "온도조절", "항균"]
                },
                "autumn": {
                    "best_for": ["건조함 관리", "겨울 준비"],
                    "promotions": ["추석 할인", "환절기 프로모션"],
                    "considerations": ["건조함", "온도차", "정전기"],
                    "recommended_features": ["보습", "온도안정성", "정전기방지"]
                },
                "winter": {
                    "best_for": ["보온", "연말 할인"],
                    "promotions": ["연말 대세일", "신년 할인"],
                    "considerations": ["난방", "건조함", "정전기"],
                    "recommended_features": ["보온성", "습도조절", "따뜻한 소재"]
                }
            },
            "online_vs_offline": {
                "online_advantages": [
                    "더 많은 선택지와 정보",
                    "고객 리뷰 및 평점 확인 가능",
                    "가격 비교 용이",
                    "체험 기간 제공 (대부분)",
                    "집에서 편리한 구매",
                    "할인 혜택 많음"
                ],
                "online_disadvantages": [
                    "직접 체험 불가 (초기)",
                    "배송 기간 필요",
                    "설치/반품 번거로움",
                    "상담의 한계",
                    "충동 구매 위험"
                ],
                "offline_advantages": [
                    "직접 체험 가능",
                    "전문가 상담",
                    "즉시 가져갈 수 있음",
                    "실제 품질 확인",
                    "협상 가능성"
                ],
                "offline_disadvantages": [
                    "제한된 매장 시간",
                    "압박감 있는 판매",
                    "선택지 제한",
                    "가격 비교 어려움",
                    "재고 한계"
                ]
            },
            "common_mistakes": [
                "너무 짧은 체험 시간",
                "가격만 고려한 선택",
                "파트너 의견 무시",
                "현재 매트리스와 비교 없이 구매",
                "보증 조건 미확인",
                "배송/설치 비용 미고려",
                "충동적인 구매 결정",
                "트렌드만 좇는 선택",
                "과도한 기능 추구",
                "사후 관리 계획 없음"
            ],
            "expert_recommendations": {
                "sleep_specialists": [
                    "개인의 수면 패턴을 먼저 파악하라",
                    "건강 상태를 고려한 선택이 중요",
                    "체험 기간을 충분히 활용하라",
                    "파트너와의 호환성 고려 필수"
                ],
                "chiropractors": [
                    "척추 정렬이 가장 중요",
                    "개인의 체형에 맞는 지지력 선택",
                    "너무 부드럽거나 딱딱한 것 피하기",
                    "베개와의 조화도 중요"
                ],
                "physical_therapists": [
                    "기존 통증 부위 고려한 선택",
                    "압력 분산 기능 중요",
                    "회복을 돕는 소재 선택",
                    "장기적 건강 효과 고려"
                ]
            },
            "technology_trends": {
                "smart_mattresses": {
                    "features": ["수면 추적", "온도 조절", "스마트폰 연동", "자동 조절"],
                    "pros": ["데이터 기반 수면 개선", "개인 맞춤", "편의성"],
                    "cons": ["높은 가격", "기술 의존", "프라이버시 우려"],
                    "recommended_for": ["기술 애호가", "데이터 중시", "건강 관리"]
                },
                "cooling_technology": {
                    "types": ["젤 메모리폼", "상변화물질", "통풍 시스템", "쿨링 파이버"],
                    "effectiveness": "2-5도 온도 저하 효과",
                    "best_for": ["더위 많이 타는 분", "갱년기", "운동선수"],
                    "considerations": ["초기 투자 비용", "유지보수", "개인차"]
                },
                "eco_friendly": {
                    "materials": ["천연 라텍스", "유기농 면", "대나무 섬유", "재활용 소재"],
                    "certifications": ["CertiPUR-US", "GREENGUARD", "OEKO-TEX"],
                    "benefits": ["환경 보호", "건강 안전", "지속 가능성"],
                    "trend": "젊은 층을 중심으로 급성장"
                }
            },
            "regional_preferences": {
                "korea": {
                    "popular_types": ["메모리폼", "하이브리드", "라텍스"],
                    "preferred_firmness": "미디움-펌",
                    "key_features": ["통기성", "항균", "체압분산"],
                    "cultural_factors": ["온돌 문화", "바닥 생활", "작은 공간"],
                    "seasonal_needs": ["여름 쿨링", "겨울 보온", "습도 조절"]
                },
                "climate_considerations": {
                    "humid_regions": ["통기성 우선", "항균 필수", "빠른 건조"],
                    "dry_regions": ["보습 기능", "정전기 방지", "온도 안정성"],
                    "cold_regions": ["보온성", "습도 조절", "두꺼운 매트리스"],
                    "hot_regions": ["쿨링 기능", "통풍", "온도 조절"]
                }
            }
        }
        
        return {
            "mattresses": mattresses,
            "buying_guide": buying_guide
        }

    def save_to_file(self, data: Dict, filepath: str):
        """데이터를 JSON 파일로 저장"""
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"데이터가 {filepath}에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    augmentor = MattressDataAugmentor()
    
    # 1만건 데이터 생성
    dataset = augmentor.generate_dataset(3000)
    
    # 데이터 검증 (가격 0 체크)
    invalid_count = 0
    for mattress in dataset['mattresses']:
        if mattress['price'] <= 0:
            mattress['price'] = 150000  # 기본값으로 수정
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"⚠️  {invalid_count}개의 잘못된 가격 데이터를 수정했습니다.")
    
    # 파일 저장
    filepath = "data/mattress_data.json"
    augmentor.save_to_file(dataset, filepath)
    
    # 통계 출력
    prices = [m['price'] for m in dataset['mattresses']]
    print(f"\n=== 생성 완료 ===")
    print(f"총 매트리스 수: {len(dataset['mattresses'])}건")
    print(f"브랜드 수: {len(set(m['brand'] for m in dataset['mattresses']))}개")
    print(f"매트리스 타입 수: {len(set(m['type'] for m in dataset['mattresses']))}개")
    print(f"가격 범위: {min(prices):,}원 ~ {max(prices):,}원")
    print(f"평균 가격: {sum(prices) // len(prices):,}원")
    
    # 가격 분포 확인
    price_ranges = {
        "10만원 미만": len([p for p in prices if p < 100000]),
        "10-30만원": len([p for p in prices if 100000 <= p < 300000]),
        "30-60만원": len([p for p in prices if 300000 <= p < 600000]),
        "60-100만원": len([p for p in prices if 600000 <= p < 1000000]),
        "100만원 이상": len([p for p in prices if p >= 1000000])
    }
    
    print("\n=== 가격 분포 ===")
    for range_name, count in price_ranges.items():
        percentage = (count / len(prices)) * 100
        print(f"{range_name}: {count}건 ({percentage:.1f}%)")

if __name__ == "__main__":
    main()