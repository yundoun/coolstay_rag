# 🍯 꿀스테이 RAG 시스템

LangGraph 기반의 꿀스테이 회사 내부 문서 검색 및 질의응답 시스템입니다.

## 📚 지원 도메인

1. **HR Policy** - 인사정책 및 근무규정
2. **Tech Policy** - 기술정책 및 개발 가이드라인
3. **Architecture** - CMS 아키텍처 및 시스템 설계
4. **Component** - 컴포넌트 가이드라인
5. **Deployment** - 배포 프로세스
6. **Development** - 개발 프로세스
7. **Business Policy** - 비즈니스 정책

## 🚀 시작하기

### 환경 설정
```bash
# 가상환경 활성화
source .venv/bin/activate

# 환경변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정
```

### 실행
```bash
# 노트북에서 개발
jupyter lab

# Streamlit으로 실행 (추후)
streamlit run app.py
```

## 🏗️ 아키텍처

- **Multi-Agent RAG**: 7개 도메인별 전문 에이전트
- **Corrective RAG**: 자가교정 메커니즘으로 답변 품질 향상
- **Human-in-the-Loop**: 인간 검증을 통한 품질 보증
- **Vector Store**: ChromaDB 기반 도메인별 벡터 저장소

## 📁 프로젝트 구조

```
coolstay_rag/
├── .venv/              # 가상환경
├── notebooks/          # Jupyter 노트북
├── src/               # 소스 코드
├── docs/              # 문서
├── tests/             # 테스트
├── chroma_db/         # 벡터 저장소
├── .env              # 환경 변수
└── requirements.txt   # 의존성
```