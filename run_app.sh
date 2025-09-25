#!/bin/bash

# CoolStay RAG Streamlit 애플리케이션 실행 스크립트

echo "🍯 CoolStay RAG Assistant 시작 중..."
echo "=================================================================="

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경 존재 여부 확인
if [ -d "coolstay_rag_env" ]; then
    echo "✅ 가상환경 감지됨 (coolstay_rag_env)"
    echo "🔄 가상환경 활성화 중..."
    source coolstay_rag_env/bin/activate
elif [ -d ".venv" ]; then
    echo "✅ 가상환경 감지됨 (.venv)"
    echo "🔄 가상환경 활성화 중..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "✅ 가상환경 감지됨 (venv)"
    echo "🔄 가상환경 활성화 중..."
    source venv/bin/activate
else
    echo "⚠️  가상환경이 없습니다."
    echo "💡 다음 명령어로 가상환경을 생성하세요:"
    echo "   python3 -m venv coolstay_rag_env"
    echo "   source coolstay_rag_env/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "🚀 기본 환경에서 실행합니다..."
fi

# 의존성 확인
echo "🔍 주요 의존성 확인 중..."

if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit이 설치되지 않았습니다."
    echo "💡 설치 명령어: pip install streamlit"
    exit 1
fi

echo "✅ Streamlit 설치 확인됨"

# 환경 변수 확인
if [ ! -f ".env" ]; then
    echo "⚠️  .env 파일이 없습니다."
    echo "💡 .env 파일을 생성하고 다음을 설정하세요:"
    echo "   OPENAI_API_KEY=your_api_key_here"
    echo "   TAVILY_API_KEY=your_tavily_key_here"
    echo ""
fi

# 포트 설정
PORT=${PORT:-8501}

echo ""
echo "🚀 CoolStay RAG Assistant 시작!"
echo "📱 브라우저에서 http://localhost:$PORT 로 접속하세요"
echo "🛑 종료하려면 Ctrl+C를 누르세요"
echo "=================================================================="
echo ""

# Streamlit 애플리케이션 실행
streamlit run streamlit_app.py --server.port=$PORT --server.headless=true --server.fileWatcherType=none