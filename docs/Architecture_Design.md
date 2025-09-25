# 🍯 꿀스테이 RAG 시스템 아키텍처 설계

## 📋 시스템 개요

### 기존 법률 RAG vs 꿀스테이 RAG 비교
```yaml
법률 RAG (3개 도메인):
  - 개인정보보호법 → HR_Policy (인사정책)
  - 근로기준법 → Tech_Policy (기술정책)
  - 주택임대차보호법 → Architecture (아키텍처)
  + 웹검색

꿀스테이 RAG (7+1개 도메인):
  - HR_Policy (인사정책)
  - Tech_Policy (기술정책)
  - Architecture (아키텍처)
  - Component (컴포넌트)
  - Deployment (배포)
  - Development (개발프로세스)
  - Business_Policy (비즈니스정책)
  + 웹검색
```

## 🏗️ 전체 시스템 아키텍처

### 레이어 구조
```
┌─────────────────────────────────────────────────────────┐
│                 Presentation Layer                      │
│        (Streamlit UI / Jupyter Interface)              │
├─────────────────────────────────────────────────────────┤
│                  Orchestration Layer                   │
│     (LangGraph Master Agent + Question Router)         │
├─────────────────────────────────────────────────────────┤
│                   Agent Layer                          │
│  [HR] [Tech] [Arch] [Comp] [Deploy] [Dev] [Biz] [Web]  │
├─────────────────────────────────────────────────────────┤
│                 Knowledge Layer                        │
│     (7 Domain Vector Stores + Embedding Model)        │
├─────────────────────────────────────────────────────────┤
│                   Data Layer                           │
│              (Markdown Documents)                      │
└─────────────────────────────────────────────────────────┘
```

### 플로우 다이어그램
```
📥 사용자 질문
    ↓
🧭 질문 분석 & 라우팅 (Master Agent)
    ↓
🔀 도메인 선택 (1~7개 에이전트 동적 선택)
    ↓
🔄 각 선택된 에이전트에서 Corrective RAG 실행:
    ├─ 📚 문서 검색 (벡터 DB)
    ├─ 🔍 정보 추출 및 평가
    ├─ ❓ [품질 부족시] 쿼리 재작성 → 재검색
    └─ 💬 도메인별 답변 생성
    ↓
🔗 멀티 에이전트 답변 통합 (Master Agent)
    ↓
🧑‍⚖️ ReAct 에이전트 품질 평가 (6차원, 60점 만점)
    ↓
⏸️ Human Review (HITL) - 인터럽트
    ├─ ✅ 승인 → 완료
    └─ ❌ 거절 → 질문 분석 재시작
    ↓
📱 최종 답변 반환
```

## 🎯 핵심 RAG 기법 적용

### 1. Domain-Specific Multi-Knowledge Base RAG
- **7개 독립 벡터 저장소**: 각 도메인별 ChromaDB
- **메타데이터 강화**: 문서명, 섹션, 제목 정보
- **구조화된 청킹**: 마크다운 구조 기반 분할

### 2. Multi-Agent Corrective RAG
- **8개 전문 에이전트**: 7개 도메인 + 1개 웹검색
- **자가교정 메커니즘**: 품질 평가 후 쿼리 재작성
- **동적 라우팅**: 질문 내용 기반 에이전트 선택

### 3. Human-in-the-Loop (HITL) Quality Assurance
- **AI 자동 평가**: 6차원 품질 평가 시스템
- **인간 최종 검증**: interrupt 기반 승인/거절
- **순환 개선**: 거절 시 재처리 메커니즘

## 📁 데이터 구조 설계

### 도메인별 데이터 매핑
```yaml
hr_policy:
  file: "HR_Policy_Guide.md"
  description: "인사정책, 근무시간, 휴가, 급여, 복리후생"
  agent_name: "hr_policy_agent"
  db_name: "hr_policy_db"

tech_policy:
  file: "Tech_Policy_Guide.md"
  description: "기술정책, 개발환경, 코딩표준, 보안정책"
  agent_name: "tech_policy_agent"
  db_name: "tech_policy_db"

architecture:
  file: "Architecture_Guide.md"
  description: "CMS 아키텍처, 시스템설계, 레이어구조"
  agent_name: "architecture_agent"
  db_name: "architecture_db"

component:
  file: "Component_Guide.md"
  description: "컴포넌트 가이드라인, UI/UX 표준"
  agent_name: "component_agent"
  db_name: "component_db"

deployment:
  file: "Deployment_Guide.md"
  description: "배포프로세스, CI/CD, 환경관리"
  agent_name: "deployment_agent"
  db_name: "deployment_db"

development:
  file: "Development_Process_Guide.md"
  description: "개발프로세스, 워크플로우, 협업규칙"
  agent_name: "development_agent"
  db_name: "development_db"

business_policy:
  file: "Business_Policy_Guide.md"
  description: "비즈니스정책, 운영규칙, 의사결정"
  agent_name: "business_policy_agent"
  db_name: "business_policy_db"

web_search:
  description: "실시간 웹 검색, 최신 정보"
  agent_name: "web_search_agent"
  api: "Tavily Search API"
```

## 🔧 기술 스택

### Core Framework
```yaml
LangChain: 0.3.27        # RAG 파이프라인
LangGraph: 0.6.7         # 워크플로우 오케스트레이션
ChromaDB: 1.1.0          # 벡터 저장소
```

### LLM & Embedding
```yaml
OpenAI: GPT-4o-mini      # 메인 언어모델
Embedding: bge-m3        # 다국어 임베딩 (OllamaEmbeddings)
```

### External APIs
```yaml
Tavily API: 웹 검색
OpenAI API: LLM 서비스
```

## 🎯 성능 최적화 전략

### 1. 검색 품질 향상
- **도메인 분리**: 각 도메인별 독립 벡터 저장소
- **메타데이터 활용**: 문서 구조 정보 포함
- **하이브리드 검색**: 벡터 + 웹 검색 조합

### 2. 응답 속도 최적화
- **병렬 에이전트**: 다중 도메인 동시 검색
- **캐싱**: 벡터 저장소 및 임베딩 캐시
- **청킹 최적화**: 적절한 청크 크기 설정

### 3. 답변 품질 보장
- **Corrective RAG**: 자동 품질 평가 및 재검색
- **HITL 검증**: 인간 최종 검토
- **다차원 평가**: 6가지 기준 체계적 평가

## 📊 평가 지표

### 시스템 성능 지표
```yaml
검색 정확도: 관련 문서 검색 비율
응답 품질: 6차원 평가 점수 (60점 만점)
응답 속도: 평균 응답 시간
사용자 만족도: HITL 승인 비율
```

### 도메인별 커버리지
```yaml
HR Policy: 인사 관련 질문 처리율
Tech Policy: 기술 관련 질문 처리율
Architecture: 아키텍처 관련 질문 처리율
Component: 컴포넌트 관련 질문 처리율
Deployment: 배포 관련 질문 처리율
Development: 개발 관련 질문 처리율
Business: 비즈니스 관련 질문 처리율
```

## 🔄 확장성 고려사항

### 수평적 확장
- **새 도메인 추가**: 새로운 MD 파일 → 새로운 에이전트
- **다국어 지원**: bge-m3 모델의 다국어 임베딩 활용
- **API 통합**: 외부 데이터 소스 추가 가능

### 수직적 확장
- **모델 업그레이드**: GPT-4 → GPT-5 등
- **벡터 DB 확장**: ChromaDB → 분산 벡터 DB
- **부하 분산**: 에이전트별 독립 서버 배치

---

**다음 단계**: 구현 방식 결정 (노트북 vs 파이썬 파일) 및 디렉토리 구조 설계