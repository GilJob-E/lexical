# 가중치 설계 및 매핑 문서

## 1. 개요

### 1.1 참고 논문
- **논문**: Naim, I., et al. (2018). "Automated Analysis and Prediction of Job Interview Performance"
- **출처**: IEEE Transactions on Affective Computing
- **데이터셋**: MIT Interview Dataset (138개 면접 영상, 69명 참가자)

### 1.2 설계 원칙
- **원논문 가중치 사용**: 멀티모달 통합을 위해 Table 6의 원본 가중치 적용
- **피처별 정규화**: Min-Max 정규화 (0-1 범위)
- **확장성**: Facial, Prosodic 모듈과 직접 통합 가능한 구조

---

## 2. 원논문 가중치 (Table 6)

> **중요**: Table 6은 각 trait별 **Top 20 피처**만 표시합니다.
> Top 20에 포함되지 않은 피처는 weight=0으로 처리합니다.
> 아래 값들은 PDF page 12에서 직접 추출한 정확한 SVR 모델 가중치입니다.

### 2.1 Lexical 피처 가중치 (Table 6 Top 20 기준)

#### Core Lexical Features (모든 trait의 Top 20에 포함)
| Feature | Overall | Rec.Hiring | Excited | Engagement | Friendly | 설명 |
|---------|---------|------------|---------|------------|----------|------|
| wpsec | **0.11** | **0.139** | **0.123** | **0.135** | **0.089** | 초당 단어 수 |
| upsec | **0.093** | **0.098** | **0.077** | **0.097** | **0.073** | 초당 고유 단어 수 |
| fpsec | **-0.086** | **-0.130** | **-0.069** | **-0.077** | **-0.063** | 초당 필러 수 (Fillers) |
| quantifier_ratio | **0.086** | **0.109** | **0.068** | **0.075** | **0.061** | 수량사 비율 |

#### Partial Lexical Features (일부 trait의 Top 20에 포함)
| Feature | Overall | Rec.Hiring | Excited | Engagement | Friendly | 설명 |
|---------|---------|------------|---------|------------|----------|------|
| we_ratio | **0.067** | **0.067** | 0 | 0 | 0 | 1인칭 복수 비율 |
| work_ratio | 0 | **0.072** | 0 | 0 | 0 | 업무 관련어 비율 |
| adverb_ratio | 0 | **0.066** | 0 | 0 | 0 | 부사 비율 |
| preposition_ratio | 0 | **0.066** | 0 | 0 | 0 | 전치사(조사) 비율 |

#### Non-Top 20 Features (weight=0)
아래 피처들은 모든 trait에서 Top 20에 포함되지 않아 weight=0입니다:
- Speaking rate: wc, uc
- Pronouns: i_ratio, they_ratio
- POS: article_ratio, verb_ratio, conjunction_ratio, number_ratio
- Emotion: pos_emotion_ratio, neg_emotion_ratio, anxiety_ratio, anger_ratio, sadness_ratio
- Cognitive: cognitive_ratio, inhibition_ratio, perceptual_ratio
- Misc: nonfluency_ratio, negation_ratio, relativity_ratio, swear_ratio

### 2.2 Facial 피처 가중치 (Table 6 - 통합 예정)

| Feature | Overall | Rec.Hiring | Excited | Engagement | Friendly | 설명 |
|---------|---------|------------|---------|------------|----------|------|
| smile | **0.074** | **0.093** | **0.122** | **0.082** | **0.238** | 미소 |
| nod | **0.068** | 0 | 0 | 0 | 0 | 고개 끄덕임 |
| LipCDt | 0 | **0.076** | **0.069** | **0.069** | **0.095** | 입꼬리 거리 |
| LeftEye_h | 0 | **0.069** | 0 | 0 | 0 | 왼쪽 눈 높이 |

### 2.3 Prosodic 피처 가중치 (Table 6 - 통합 예정)

| Feature | Overall | Rec.Hiring | Excited | Engagement | Friendly | 설명 |
|---------|---------|------------|---------|------------|----------|------|
| avgBand1 | **-0.12** | **-0.132** | **-0.159** | **-0.171** | 0 | 저주파 대역 평균 |
| avgDurPause | **-0.09** | **-0.094** | 0 | **-0.078** | 0 | 평균 휴지 길이 |
| percentUnvoiced | **-0.076** | **-0.111** | 0 | **-0.083** | 0 | 무성음 비율 |
| maxDurPause | **-0.076** | **-0.083** | 0 | 0 | 0 | 최대 휴지 길이 |
| intensityMax | 0 | 0 | **0.124** | **0.174** | **0.094** | 최대 강도 |
| intensityMean | 0 | 0 | **0.120** | **0.146** | **0.090** | 평균 강도 |
| diffIntMaxMin | 0 | 0 | **0.132** | **0.151** | **0.089** | 강도 차이 |
| mean pitch | 0 | 0 | 0 | 0 | **0.136** | 평균 피치 |

---

## 3. 영어→한국어 피처 매핑

### 3.1 사전 기반 피처 (Dictionary-based)

#### Pronouns (대명사)
| LIWC (영어) | 한국어 매핑 | 예시 |
|-------------|-------------|------|
| I (1인칭 단수) | 나, 저, 내, 제 | "나는", "저는", "내가", "제가" |
| We (1인칭 복수) | 우리, 저희 | "우리는", "저희가" |
| They (3인칭 복수) | 그들, 그녀들 | "그들은", "그녀들이" |

```python
# pronouns.py
PRONOUNS_I = {"나", "저", "내", "제", "나의", "저의", ...}
PRONOUNS_WE = {"우리", "저희", "우리의", "저희의", ...}
PRONOUNS_THEY = {"그들", "그녀들", "저들", ...}
```

#### Emotions (감정어)
| LIWC (영어) | 한국어 매핑 | 예시 |
|-------------|-------------|------|
| Positive Emotion | 긍정 감정 사전 | 좋다, 행복하다, 감사하다, 기쁘다 |
| Negative Emotion | 부정 감정 사전 | 싫다, 나쁘다, 슬프다, 힘들다 |
| Anxiety | 불안 사전 | 걱정, 불안, 긴장, 두렵다 |
| Anger | 분노 사전 | 화나다, 짜증, 분노, 열받다 |
| Sadness | 슬픔 사전 | 슬프다, 우울, 쓸쓸하다 |

#### Work (업무 관련어)
| LIWC (영어) | 한국어 매핑 | 예시 |
|-------------|-------------|------|
| Work | 업무/직장 사전 | 회사, 업무, 프로젝트, 팀, 협업 |

#### Cognitive (인지 관련어)
| LIWC (영어) | 한국어 매핑 | 예시 |
|-------------|-------------|------|
| Cognitive | 인지 과정 사전 | 생각하다, 알다, 이해하다, 결정하다 |
| Inhibition | 억제 사전 | 막다, 제한, 금지, 참다 |
| Perceptual | 지각 사전 | 보다, 듣다, 느끼다, 만지다 |

#### Non-fluencies (비유창성)
| LIWC (영어) | 한국어 매핑 | 예시 |
|-------------|-------------|------|
| Fillers | 필러 사전 | 음, 어, 그, 저, 뭐, 그러니까 |

### 3.2 POS 기반 피처 (Part-of-Speech)

Kiwi 형태소 분석기의 Sejong 태그셋을 LIWC 카테고리로 매핑합니다.

| LIWC 카테고리 | Kiwi POS 태그 | 설명 |
|---------------|---------------|------|
| Article | MM | 관형사 (이, 그, 저, 새, 헌) |
| Verb | VV, VA, VX, VCP, VCN | 동사, 형용사, 보조동사 |
| Adverb | MAG, MAJ | 일반 부사, 접속 부사 |
| Preposition | JKB | 부사격 조사 (에, 에서, 로, 으로) |
| Conjunction | JC | 접속 조사 (와, 과, 하고) |
| Number | SN, NR | 수사 (하나, 둘, 1, 2) |

```python
# pos_mapping.py
POS_MAPPING = {
    "MM": "article",      # 관형사 → Article
    "JKB": "preposition", # 부사격 조사 → Preposition
    "MAG": "adverb",      # 일반 부사 → Adverb
    "MAJ": "adverb",      # 접속 부사 → Adverb
    "JC": "conjunction",  # 접속 조사 → Conjunction
    "VV": "verb",         # 동사 → Verb
    "VA": "verb",         # 형용사 → Verb
    "VX": "verb",         # 보조 동사 → Verb
    "SN": "number",       # 수사 → Number
}
```

### 3.3 매핑 시 고려사항

#### 영어 vs 한국어 언어적 차이

| 차이점 | 영어 | 한국어 | 대응 방법 |
|--------|------|--------|-----------|
| 관사 | a, the | 없음 | 관형사(MM)로 대체 |
| 전치사 | in, on, at | 없음 | 부사격 조사(JKB)로 대체 |
| 형용사 | Adjective (별도) | 동사와 활용 동일 | VA 태그로 동사에 포함 |
| 어순 | SVO | SOV | 영향 없음 (비율 계산) |
| 교착어 | 분리된 단어 | 조사 결합 | 형태소 분석으로 분리 |

---

## 4. 정규화 (Normalization)

> **참고**: 자세한 통계 분석 및 시각화는 [FEATURE_STATISTICS.md](FEATURE_STATISTICS.md) 참조

### 4.1 Z-Score 정규화 (권장)

**76,100개 한국어 면접 데이터 기반 통계** (2025-12-02 계산)

논문(Naim et al. 2018)에서 SVR 모델에 Z-Score 정규화를 사용했습니다.

#### Z-Score 공식
```python
z = (x - μ) / σ
```

#### Target Features 통계 (8개)

| Feature | Mean (μ) | Std (σ) | z=-2 | z=-1 | z=+1 | z=+2 |
|---------|----------|---------|------|------|------|------|
| wpsec | 2.859 | 0.578 | 1.70 | 2.28 | 3.44 | 4.01 |
| upsec | 1.281 | 0.258 | 0.77 | 1.02 | 1.54 | 1.80 |
| fpsec | 0.289 | 0.104 | 0.08 | 0.18 | 0.39 | 0.50 |
| quantifier_ratio | 0.221 | 0.039 | 0.14 | 0.18 | 0.26 | 0.30 |
| we_ratio | 0.001 | 0.004 | 0.00* | 0.00 | 0.01 | 0.01 |
| work_ratio | 0.116 | 0.031 | 0.05 | 0.09 | 0.15 | 0.18 |
| adverb_ratio | 0.055 | 0.022 | 0.01 | 0.03 | 0.08 | 0.10 |
| preposition_ratio | 0.042 | 0.015 | 0.01 | 0.03 | 0.06 | 0.07 |

*음수값은 ratio feature의 경우 0으로 클리핑

#### Z-Score 해석 가이드

![Z-Score Interpretation](images/zscore_interpretation.png)

| Z-Score | 해석 | 백분위 |
|---------|------|--------|
| z > +2 | 매우 높음 | P97.7+ |
| z > +1 | 높음 | P84+ |
| z ≈ 0 | 평균 | P50 |
| z < -1 | 낮음 | P16- |
| z < -2 | 매우 낮음 | P2.3- |

#### Z-Score 사용 예시

```python
from ko_liwc.scoring import ZScoreNormalizer
from ko_liwc.scoring.normalizer import DEFAULT_FEATURE_STATS

# 통계 기반 정규화
normalizer = ZScoreNormalizer(preset_stats=DEFAULT_FEATURE_STATS)
z_scores = normalizer.transform(features)

# 예시: wpsec=3.0인 경우
# z = (3.0 - 2.859) / 0.578 = 0.244 (평균보다 약간 높음)
```

### 4.2 Min-Max 정규화 (하위 호환)

기존 Min-Max 범위 (0-1 스케일링)

#### Speaking Rate Features
| Feature | Min | Max | 단위 |
|---------|-----|-----|------|
| wpsec | 0.0 | 5.0 | 형태소/초 |
| upsec | 0.0 | 3.0 | 고유 형태소/초 |
| fpsec | 0.0 | 0.5 | 필러/초 |
| wc | 0 | 2000 | 형태소 |
| uc | 0 | 500 | 고유 형태소 |

#### Ratio Features
| Feature | Min | Max | 비고 |
|---------|-----|-----|------|
| i_ratio | 0.0 | 0.15 | - |
| we_ratio | 0.0 | 0.10 | - |
| they_ratio | 0.0 | 0.05 | - |
| article_ratio | 0.0 | 0.10 | - |
| verb_ratio | 0.0 | 0.40 | - |
| adverb_ratio | 0.0 | 0.15 | - |
| preposition_ratio | 0.0 | 0.15 | - |
| conjunction_ratio | 0.0 | 0.08 | - |
| number_ratio | 0.0 | 0.05 | - |
| pos_emotion_ratio | 0.0 | 0.10 | - |
| neg_emotion_ratio | 0.0 | 0.05 | - |
| anxiety_ratio | 0.0 | 0.03 | - |
| anger_ratio | 0.0 | 0.02 | - |
| sadness_ratio | 0.0 | 0.02 | - |
| cognitive_ratio | 0.0 | 0.15 | - |
| inhibition_ratio | 0.0 | 0.05 | - |
| perceptual_ratio | 0.0 | 0.10 | - |
| nonfluency_ratio | 0.0 | 0.10 | - |
| negation_ratio | 0.0 | 0.08 | - |
| quantifier_ratio | 0.0 | 0.10 | - |
| work_ratio | 0.0 | 0.15 | - |
| relativity_ratio | 0.0 | 0.12 | - |
| swear_ratio | 0.0 | 0.01 | - |

#### Min-Max 공식

```python
normalized = (value - min) / (max - min)
normalized = max(0.0, min(1.0, normalized))  # Clip to [0, 1]
```

---

## 5. 멀티모달 통합 가이드

### 5.1 통합 아키텍처

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Lexical   │    │   Facial    │    │  Prosodic   │
│   Module    │    │   Module    │    │   Module    │
│  (ko-liwc)  │    │  (외부 팀)  │    │  (외부 팀)  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────────────────────────────────────────┐
│              Feature Normalizer                   │
│         (각 피처별 Min-Max 정규화)                │
└──────────────────────────────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────────────────────────────────────────┐
│              Multimodal Scorer                    │
│    score = Σ(weight_i × normalized_feature_i)    │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│              Final Score (0-100)                  │
└──────────────────────────────────────────────────┘
```

### 5.2 통합 점수 계산

```python
class MultimodalScorer:
    def score(self, lexical: Dict, facial: Dict, prosodic: Dict) -> float:
        """멀티모달 점수 계산.

        Args:
            lexical: 정규화된 Lexical 피처 딕셔너리
            facial: 정규화된 Facial 피처 딕셔너리
            prosodic: 정규화된 Prosodic 피처 딕셔너리

        Returns:
            0-100 범위의 최종 점수
        """
        # 각 모달리티별 가중합
        lex_score = sum(
            LEXICAL_WEIGHTS[trait][f] * lexical[f]
            for f in lexical
        )
        fac_score = sum(
            FACIAL_WEIGHTS[trait][f] * facial[f]
            for f in facial
        )
        pro_score = sum(
            PROSODIC_WEIGHTS[trait][f] * prosodic[f]
            for f in prosodic
        )

        # 합산 후 스케일링
        raw_score = lex_score + fac_score + pro_score
        return max(0, min(100, raw_score * 100))
```

### 5.3 Facial 모듈 통합 방법

```python
# Facial 팀에서 제공해야 할 인터페이스
class FacialFeatureExtractor:
    def extract(self, video_path: str) -> Dict[str, float]:
        """비디오에서 Facial 피처 추출.

        Returns:
            {
                "smile": 0.0~1.0,      # 미소 확률
                "nod": 0.0~1.0,        # 고개 끄덕임 비율
                "lipCDt": 0.0~50.0,    # 입꼬리 거리 (픽셀)
                "eye_contact": 0.0~1.0  # 시선 접촉 비율
            }
        """
        pass

# Facial 피처 정규화 범위
FACIAL_FEATURE_RANGES = {
    "smile": (0.0, 1.0),
    "nod": (0.0, 1.0),
    "lipCDt": (0.0, 50.0),
    "eye_contact": (0.0, 1.0),
}
```

### 5.4 Prosodic 모듈 통합 방법

```python
# Prosodic 팀에서 제공해야 할 인터페이스
class ProsodicFeatureExtractor:
    def extract(self, audio_path: str) -> Dict[str, float]:
        """오디오에서 Prosodic 피처 추출.

        Returns:
            {
                "pitch_mean": float,      # 평균 피치 (Hz)
                "pitch_std": float,       # 피치 표준편차
                "intensity_mean": float,  # 평균 강도 (dB)
                "pause_ratio": 0.0~1.0    # 휴지 비율
            }
        """
        pass

# Prosodic 피처 정규화 범위 (예시)
PROSODIC_FEATURE_RANGES = {
    "pitch_mean": (50.0, 400.0),     # Hz
    "pitch_std": (0.0, 100.0),       # Hz
    "intensity_mean": (30.0, 80.0),  # dB
    "pause_ratio": (0.0, 0.5),       # ratio
}
```

### 5.5 통합 예제 코드

```python
from ko_liwc import InterviewAnalyzer
from facial_module import FacialAnalyzer  # 외부 모듈
from prosodic_module import ProsodicAnalyzer  # 외부 모듈

# 각 모달리티 분석기
lexical_analyzer = InterviewAnalyzer()
facial_analyzer = FacialAnalyzer()
prosodic_analyzer = ProsodicAnalyzer()

# 피처 추출
text = "안녕하세요. 저는 이 직무에 매우 관심이 있습니다..."
video_path = "interview.mp4"
audio_path = "interview.wav"

lexical_features = lexical_analyzer.extract_features(text, duration=60.0)
facial_features = facial_analyzer.extract(video_path)
prosodic_features = prosodic_analyzer.extract(audio_path)

# 통합 점수 계산
multimodal_scorer = MultimodalScorer()
final_score = multimodal_scorer.score(
    lexical=lexical_features,
    facial=facial_features,
    prosodic=prosodic_features,
    trait="overall"
)

print(f"Overall Score: {final_score:.1f}")
```

---

## 6. 참고 문헌

1. Naim, I., Tanveer, M. I., Gildea, D., & Hoque, M. E. (2018). "Automated Analysis and Prediction of Job Interview Performance." IEEE Transactions on Affective Computing.

2. Pennebaker, J. W., Boyd, R. L., Jordan, K., & Blackburn, K. (2015). "The Development and Psychometric Properties of LIWC2015." Austin, TX: University of Texas at Austin.

3. Kiwi 한국어 형태소 분석기: https://github.com/bab2min/Kiwi
