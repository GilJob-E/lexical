"""ko-liwc 데모 스크립트."""

from ko_liwc import InterviewAnalyzer

def main():
    analyzer = InterviewAnalyzer()

    # 테스트할 면접 텍스트
    texts = [
        # 긍정적인 면접
        ("긍정적 면접", """
        안녕하세요. 저는 이 직무에 매우 흥미가 있습니다.
        우리 팀과 협력하여 좋은 성과를 내겠습니다.
        열정적으로 일하고 회사 발전에 기여하겠습니다.
        감사합니다.
        """),

        # 불안한 면접
        ("불안한 면접", """
        음... 저는... 그러니까...
        걱정이 되긴 하는데요...
        잘 모르겠습니다.
        어... 힘들 것 같아요.
        """),

        # 업무 중심 면접
        ("업무 중심 면접", """
        저는 이전 회사에서 프로젝트 매니저로 근무했습니다.
        팀원들과 협업하여 프로젝트를 성공적으로 완료했습니다.
        업무 경험을 바탕으로 성과를 내겠습니다.
        """),
    ]

    for name, text in texts:
        print(f"\n{'='*50}")
        print(f"📋 {name}")
        print('='*50)

        result = analyzer.analyze(text.strip(), duration=30.0)

        print(f"\n📊 점수 (0-100)")
        print(f"  Overall:          {result.scores.overall:.1f}")
        print(f"  Recommend Hiring: {result.scores.recommend_hiring:.1f}")
        print(f"  Excited:          {result.scores.excited:.1f}")
        print(f"  Engagement:       {result.scores.engagement:.1f}")
        print(f"  Friendliness:     {result.scores.friendliness:.1f}")

        print(f"\n📈 주요 특성")
        print(f"  단어 수:          {result.features['wc']:.0f}")
        print(f"  긍정 감정어:      {result.features['pos_emotion_ratio']:.3f}")
        print(f"  부정 감정어:      {result.features['neg_emotion_ratio']:.3f}")
        print(f"  불안 관련어:      {result.features['anxiety_ratio']:.3f}")
        print(f"  업무 관련어:      {result.features['work_ratio']:.3f}")
        print(f"  비유창성:         {result.features['nonfluency_ratio']:.3f}")


if __name__ == "__main__":
    main()
