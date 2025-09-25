# 🍯 CoolStay RAG 시스템 설치 가이드

Phase 2 모듈화 완료 후 전체 시스템을 설치하고 실행하기 위한 단계별 가이드입니다.

## 📋 사전 요구사항

### 시스템 요구사항
- **Python**: 3.9 이상
- **운영체제**: macOS, Linux, Windows
- **메모리**: 8GB 이상 권장
- **디스크 공간**: 2GB 이상

### API 키 요구사항
- **OpenAI API 키**: GPT-4o-mini 모델 사용
- **Tavily API 키** (선택사항): 웹 검색 기능

---

## 🚀 설치 단계

### 1단계: 가상환경 생성 및 활성화

```bash
# 프로젝트 디렉토리로 이동
cd coolstay_rag

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
# macOS/Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate
```

### 2단계: 의존성 패키지 설치

```bash
# 모든 필수 패키지 설치
pip install -r requirements.txt
```

**주요 패키지 목록:**
- `langchain==0.3.27` - LangChain 프레임워크
- `langchain-community==0.3.29` - LangChain 커뮤니티 패키지
- `langchain-chroma==0.2.6` - ChromaDB 연동
- `langchain-openai==0.3.33` - OpenAI 연동
- `langchain-ollama==0.3.8` - Ollama 연동
- `langgraph==0.6.7` - 워크플로우 관리
- `streamlit>=1.28.0` - 웹 애플리케이션
- `pandas>=2.0.0` - 데이터 처리
- `sentence-transformers==5.1.1` - 임베딩 모델

### 3단계: 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env  # 없다면 수동으로 생성

# .env 파일 편집
nano .env  # 또는 원하는 에디터 사용
```

**.env 파일 내용:**
```bash
# OpenAI API 설정 (필수)
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini

# Tavily API 설정 (선택사항 - 웹 검색용)
TAVILY_API_KEY=your-tavily-api-key-here

# Ollama 설정 (로컬 임베딩 모델용)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=bge-m3

# 기타 설정
LOG_LEVEL=INFO
CHROMA_DB_IMPL=duckdb+parquet
```

### 4단계: 시스템 테스트

```bash
# 기본 기능 테스트
python3 test_basic_functionality.py

# 통합 테스트
python3 test_integration.py
```

### 5단계: 웹 애플리케이션 실행

```bash
# 실행 스크립트 사용 (권장)
./run_app.sh

# 또는 직접 실행
streamlit run streamlit_app.py
```

---

## 🔧 설치 문제 해결

### 일반적인 문제

#### 1. `externally-managed-environment` 오류
```bash
# 해결방법 1: 가상환경 사용 (권장)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 해결방법 2: --user 플래그 사용
pip install --user -r requirements.txt
```

#### 2. LangChain 패키지 설치 실패
```bash
# 개별 패키지 설치
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install langchain-ollama
pip install langchain-chroma
pip install langgraph
```

#### 3. Ollama 설치 (로컬 임베딩용)
```bash
# macOS
brew install ollama

# Linux/Windows: https://ollama.ai/download 참조

# bge-m3 모델 다운로드
ollama pull bge-m3
```

#### 4. ChromaDB 의존성 문제
```bash
# 시스템 패키지 설치 (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential

# macOS
xcode-select --install
```

### 성능 최적화

#### 1. 메모리 사용량 최적화
```bash
# 환경 변수 추가
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
```

#### 2. GPU 사용 (선택사항)
```bash
# CUDA 설치 후
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ 설치 검증

### 시스템 상태 확인
1. 웹 브라우저에서 `http://localhost:8501` 접속
2. 사이드바에서 "🔄 상태 확인" 클릭
3. 모든 모듈이 ✅ 표시되는지 확인

### 기능 테스트
1. **채팅 탭**에서 간단한 질문 입력
2. AI 응답 확인
3. 피드백 시스템 테스트

---

## 📱 사용법

### 기본 사용법
1. **질문 입력**: 채팅창에 질문 입력
2. **응답 확인**: AI가 생성한 답변 검토
3. **평가 제공**: 응답 품질에 대한 피드백 제공

### 고급 기능
- **도메인 특화**: 7개 전문 도메인 활용
- **Multi-Agent**: 여러 전문가 의견 종합
- **Corrective RAG**: 자동 품질 개선
- **Human-in-the-Loop**: 인간 피드백 수집

---

## 🐳 Docker 실행 (선택사항)

### Dockerfile 생성
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker 실행
```bash
# 이미지 빌드
docker build -t coolstay-rag .

# 컨테이너 실행
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key coolstay-rag
```

---

## 📞 지원

### 문제 보고
- **GitHub Issues**: 기술적 문제 및 버그 리포트
- **이메일**: 일반 문의

### 개발 참여
1. Fork the repository
2. Create feature branch
3. Submit pull request

### 도움말
- **시스템 상태**: 웹앱의 사이드바에서 확인
- **로그**: 터미널에서 실시간 로그 확인
- **테스트**: `python3 test_integration.py` 실행

---

## 🎉 완료!

설치가 성공적으로 완료되었다면:
- ✅ 모든 의존성 패키지 설치됨
- ✅ API 키 설정 완료
- ✅ 웹 애플리케이션 실행 중
- ✅ 시스템 상태 확인 완료

이제 **CoolStay RAG Assistant**를 사용할 준비가 되었습니다! 🍯