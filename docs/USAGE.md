# ko-liwc 사용 가이드

한국어 면접 텍스트 분석을 위한 LIWC 기반 NLP 패키지 사용법입니다.

## 목차

1. [설치](#1-설치)
2. [빠른 시작](#2-빠른-시작)
3. [기본 사용법](#3-기본-사용법)
4. [고급 사용법](#4-고급-사용법)
5. [출력 결과 설명](#5-출력-결과-설명)
6. [Z-Score 해석 가이드](#6-z-score-해석-가이드)
7. [전체 코드 예시](#7-전체-코드-예시)

---

## 1. 설치

```bash
pip install -e .
```

### 의존성
- Python >= 3.9
- kiwipiepy >= 0.18.0

---

## 2. 빠른 시작

```python
from ko_liwc import InterviewAnalyzer

# 분석기 생성
analyzer = InterviewAnalyzer()

# 면접 텍스트 분석 (텍스트, 발화 시간(초))
result = analyzer.analyze("저는 열심히 일하겠습니다.", duration=5.0)

# 점수 확인
print(f"Overall: {result.scores.overall:.1f}")
print(f"Recommend Hiring: {result.scores.recommend_hiring:.1f}")
```

---

## 3. 기본 사용법

### 3.1 InterviewAnalyzer - 메인 분석기

```python
from ko_liwc import InterviewAnalyzer

analyzer = InterviewAnalyzer()
```

#### 주요 메서드

| 메서드 | 반환 타입 | 설명 |
|--------|----------|------|
| `analyze(text, duration)` | `AnalysisResult` | 전체 분석 (특성 추출 + 점수 계산) |
| `extract_features(text, duration)` | `Dict[str, float]` | 특성만 추출 (28개) |
| `analyze_with_breakdown(text, duration)` | `Dict` | 상세 분석 (기여도 포함) |

#### 예시: 기본 분석

```python
text = """
안녕하세요. 저는 이 직무에 매우 흥미가 있습니다.
우리 팀과 협력하여 좋은 성과를 내겠습니다.
"""

result = analyzer.analyze(text, duration=30.0)

# 점수 확인
print(result.scores.overall)        # 전반적 점수
print(result.scores.recommend_hiring)  # 채용 추천도
```

#### 예시: 특성만 추출

```python
features = analyzer.extract_features(text, duration=30.0)

print(f"Words per second: {features['wpsec']:.3f}")
print(f"Fillers per second: {features['fpsec']:.3f}")
print(f"Quantifier ratio: {features['quantifier_ratio']:.3f}")
```

#### 예시: 상세 분석 (기여도 포함)

```python
breakdown = analyzer.analyze_with_breakdown(text, duration=30.0)

print("긍정적 영향 특성:")
for item in breakdown['top_positive']:
    print(f"  {item['feature']}: {item['contribution']:.3f}")

print("부정적 영향 특성:")
for item in breakdown['top_negative']:
    print(f"  {item['feature']}: {item['contribution']:.3f}")
```

### 3.2 AnalysisResult - 결과 객체

`analyze()` 메서드의 반환 객체입니다.

```python
result = analyzer.analyze(text, duration=30.0)
```

#### 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `result.scores` | `InterviewScore` | 5개 trait 점수 (0-100) |
| `result.features` | `Dict[str, float]` | 28개 추출 특성 |
| `result.text` | `str` | 입력 텍스트 |
| `result.duration` | `float` | 발화 시간 (초) |

#### 메서드

| 메서드 | 반환 타입 | 설명 |
|--------|----------|------|
| `summary()` | `str` | 사람이 읽기 쉬운 요약 |
| `to_dict()` | `Dict` | JSON 변환용 딕셔너리 |

```python
# 요약 출력
print(result.summary())

# JSON 변환
import json
print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
```

### 3.3 편의 함수 - analyze()

한 줄로 빠르게 분석할 수 있는 편의 함수입니다.

```python
from ko_liwc import analyze

# 기본 분석
result = analyze("면접 텍스트", duration=30.0)
print(result.scores.overall)

# 상세 분석
detailed = analyze("면접 텍스트", duration=30.0, detailed=True)
print(detailed['top_positive'])
```

---

## 4. 고급 사용법

### 4.1 Z-Score 정규화

논문(Naim et al. 2018)에서 SVR 모델에 권장하는 정규화 방식입니다.

```python
from ko_liwc import InterviewAnalyzer, ZScoreNormalizer
from ko_liwc.scoring.normalizer import DEFAULT_FEATURE_STATS

# 분석기로 특성 추출
analyzer = InterviewAnalyzer()
features = analyzer.extract_features(text, duration=30.0)

# Z-Score 정규화 (76,100개 샘플 기반 통계 사용)
normalizer = ZScoreNormalizer(preset_stats=DEFAULT_FEATURE_STATS)
z_scores = normalizer.transform(features)

# 결과 해석
for name, z in z_scores.items():
    if z > 1:
        status = "높음"
    elif z < -1:
        status = "낮음"
    else:
        status = "평균"
    print(f"{name}: z={z:.2f} ({status})")
```

#### DEFAULT_FEATURE_STATS (4개 Tier 1 Feature)

| Feature | Mean (μ) | Std (σ) |
|---------|----------|---------|
| wpsec | 2.859 | 0.578 |
| upsec | 1.281 | 0.258 |
| fpsec | 0.289 | 0.104 |
| quantifier_ratio | 0.221 | 0.039 |

### 4.2 Transcript 객체 사용

여러 발화 세그먼트를 시간 정보와 함께 분석할 때 사용합니다.

```python
from ko_liwc import Transcript, InterviewAnalyzer

# Transcript 생성
transcript = Transcript()

# 세그먼트 추가 (텍스트, 시작시간, 종료시간)
transcript.add_segment("안녕하세요. 저는 김철수입니다.", 0.0, 3.0)
transcript.add_segment("이 직무에 지원하게 된 이유가 있습니다.", 3.0, 6.0)
transcript.add_segment("열심히 일하겠습니다.", 6.0, 8.0)

# 분석
analyzer = InterviewAnalyzer()
result = analyzer.analyze_transcript(transcript)

print(f"Total duration: {transcript.total_duration}s")
print(f"Overall score: {result.scores.overall:.1f}")
```

### 4.3 Segment 리스트 사용

딕셔너리 리스트로 세그먼트를 전달할 수도 있습니다.

```python
segments = [
    {"text": "안녕하세요.", "start": 0.0, "end": 2.0},
    {"text": "저는 이 회사에 관심이 많습니다.", "start": 2.0, "end": 5.0},
    {"text": "열심히 하겠습니다.", "start": 5.0, "end": 7.0},
]

result = analyzer.analyze_segments(segments)
print(result.scores.overall)
```

### 4.4 커스텀 Normalizer/Scorer 사용

```python
from ko_liwc import InterviewAnalyzer
from ko_liwc.scoring import MinMaxNormalizer, InterviewScorer

# 커스텀 정규화 범위 설정
custom_ranges = {
    "wpsec": {"min": 1.0, "max": 5.0},
    "fpsec": {"min": 0.0, "max": 1.0},
}
normalizer = MinMaxNormalizer(preset_ranges=custom_ranges)

# 커스텀 스코어러로 분석기 생성
analyzer = InterviewAnalyzer(custom_normalizer=normalizer)
result = analyzer.analyze(text, duration=30.0)
```

---

## 5. 출력 결과 설명

### 5.1 점수 (Scores) - 0~100 범위

`result.scores`로 접근합니다.

| 속성 | 설명 | 텍스트 기반 상관계수 |
|------|------|---------------------|
| `overall` | 전반적 면접 수행 | r ≈ 0.55 |
| `recommend_hiring` | 채용 추천도 | r ≈ 0.55 |
| `excited` | 열정/흥미 | r ≈ 0.35 |
| `engagement` | 참여도 | r ≈ 0.35 |
| `friendliness` | 친근함 | r ≈ 0.35 |

```python
scores = result.scores

print(f"Overall: {scores.overall:.1f}")
print(f"Recommend Hiring: {scores.recommend_hiring:.1f}")
print(f"Excited: {scores.excited:.1f}")
print(f"Engagement: {scores.engagement:.1f}")
print(f"Friendliness: {scores.friendliness:.1f}")
print(f"Average: {scores.average:.1f}")
```

> **참고**: Overall/Hiring은 텍스트만으로 합리적인 예측이 가능하나, Excited/Engagement/Friendliness는 음성적 특성(prosodic features)이 필수적입니다.

### 5.2 추출 특성 (Features)

`result.features`로 접근합니다. 총 28개 특성이 추출됩니다.

#### Target Features (4개 Tier 1) - 점수에 사용

| Feature | 설명 | 방향 |
|---------|------|------|
| `wpsec` | 초당 형태소 수 | 높을수록 좋음 |
| `upsec` | 초당 고유 형태소 수 | 높을수록 좋음 |
| `fpsec` | 초당 필러 수 | **낮을수록 좋음** |
| `quantifier_ratio` | 수량사 비율 | 높을수록 좋음 |

```python
features = result.features

# Target Features (점수에 영향)
print(f"wpsec: {features['wpsec']:.3f}")
print(f"upsec: {features['upsec']:.3f}")
print(f"fpsec: {features['fpsec']:.3f}")
print(f"quantifier_ratio: {features['quantifier_ratio']:.3f}")

# 기타 특성
print(f"Word count: {features['wc']:.0f}")
print(f"Positive emotion: {features['pos_emotion_ratio']:.3f}")
print(f"Cognitive ratio: {features['cognitive_ratio']:.3f}")
```

---

## 6. Z-Score 해석 가이드

### 6.1 Z-Score 범위별 해석

| Z-Score | 해석 | 백분위 |
|---------|------|--------|
| z > +2 | 매우 높음 | 상위 2.3% |
| z > +1 | 높음 | 상위 16% |
| -1 ≤ z ≤ +1 | 평균 | 중간 68% |
| z < -1 | 낮음 | 하위 16% |
| z < -2 | 매우 낮음 | 하위 2.3% |

### 6.2 Feature별 평가 기준

| Feature | 미흡 (z<-1) | 평균 | 우수 (z>+1) | 방향 |
|---------|-------------|------|-------------|------|
| wpsec | < 2.28 | 2.28~3.44 | > 3.44 | 높을수록 좋음 |
| upsec | < 1.02 | 1.02~1.54 | > 1.54 | 높을수록 좋음 |
| fpsec | **> 0.39** | 0.18~0.39 | **< 0.18** | **낮을수록 좋음** |
| quantifier_ratio | < 0.18 | 0.18~0.26 | > 0.26 | 높을수록 좋음 |

### 6.3 평가 코드 예시

```python
from ko_liwc import InterviewAnalyzer, ZScoreNormalizer
from ko_liwc.scoring.normalizer import DEFAULT_FEATURE_STATS

# fpsec만 낮을수록 좋음
NEGATIVE_WEIGHT_FEATURES = {"fpsec"}

def evaluate_feature(name: str, z_score: float) -> str:
    """Z-Score 기반 특성 평가"""
    if name in NEGATIVE_WEIGHT_FEATURES:
        # fpsec: 낮을수록 좋음 (방향 반전)
        if z_score < -1:
            return "우수"
        elif z_score > 1:
            return "미흡"
        else:
            return "평균"
    else:
        # 나머지: 높을수록 좋음
        if z_score > 1:
            return "우수"
        elif z_score < -1:
            return "미흡"
        else:
            return "평균"

# 사용 예시
analyzer = InterviewAnalyzer()
features = analyzer.extract_features(text, duration=30.0)

normalizer = ZScoreNormalizer(preset_stats=DEFAULT_FEATURE_STATS)
z_scores = normalizer.transform(features)

for name, z in z_scores.items():
    evaluation = evaluate_feature(name, z)
    print(f"{name}: z={z:.2f} → {evaluation}")
```

---

## 7. 전체 코드 예시

### 7.1 기본 워크플로우

```python
from ko_liwc import InterviewAnalyzer

# 1. 분석기 생성
analyzer = InterviewAnalyzer()

# 2. 면접 텍스트 준비
text = """
안녕하세요. 저는 김철수입니다.
저는 이 직무에 지원하게 된 이유가 있습니다.
우리 회사의 비전과 제 목표가 일치하기 때문입니다.
열심히 일하고 성과를 내겠습니다.
"""

# 3. 분석 실행 (텍스트, 발화시간 60초)
result = analyzer.analyze(text, duration=60.0)

# 4. 결과 확인
print(result.summary())
print(f"\nOverall Score: {result.scores.overall:.1f}")
print(f"Recommend Hiring: {result.scores.recommend_hiring:.1f}")
```

### 7.2 Z-Score 정규화 워크플로우

```python
from ko_liwc import InterviewAnalyzer, ZScoreNormalizer
from ko_liwc.scoring.normalizer import DEFAULT_FEATURE_STATS

# 1. 분석기로 특성 추출
analyzer = InterviewAnalyzer()
features = analyzer.extract_features(text, duration=60.0)

# 2. Z-Score 정규화
normalizer = ZScoreNormalizer(preset_stats=DEFAULT_FEATURE_STATS)
z_scores = normalizer.transform(features)

# 3. 결과 해석
print("=== Z-Score 분석 결과 ===")
for name, z in z_scores.items():
    if z > 2:
        level = "매우 높음"
    elif z > 1:
        level = "높음"
    elif z < -2:
        level = "매우 낮음"
    elif z < -1:
        level = "낮음"
    else:
        level = "평균"
    print(f"{name}: z={z:+.2f} ({level})")
```

### 7.3 JSON 출력 워크플로우

```python
import json
from ko_liwc import analyze

# 분석 실행
result = analyze(text, duration=60.0)

# JSON으로 변환
output = result.to_dict()

# 파일로 저장
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(json.dumps(output, indent=2, ensure_ascii=False))
```

---

## 참고 자료

- [README.md](../README.md) - 프로젝트 개요
- [FEATURE_STATISTICS.md](FEATURE_STATISTICS.md) - 통계 상세
- [WEIGHTS_MAPPING.md](WEIGHTS_MAPPING.md) - SVR 가중치 상세
- Naim et al. (2018) - "Automated Analysis and Prediction of Job Interview Performance"
