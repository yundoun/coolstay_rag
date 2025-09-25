# 꿀스테이 기술정책 가이드

## 📋 개요
꿀스테이 개발팀의 기술 정책 및 개발 가이드라인을 안내합니다.

## 🏗️ 개발 환경 및 인프라

### 개발 환경 구성
- **로컬 개발**: Node.js 18+, React 18, Material-UI v5
- **패키지 매니저**: npm (yarn 금지)
- **IDE**: VS Code 권장, 공통 Extension Pack 사용
- **브라우저**: Chrome 최신버전 기준 개발

### 환경 분리
- **Local**: 개발자 로컬 환경
- **Development**: 개발팀 통합 테스트 환경
- **Stage**: QA 및 운영팀 검수 환경
- **Production**: 실제 서비스 환경

### 인프라 구성
- **Frontend**: AWS CloudFront + S3
- **Backend**: AWS ECS + Application Load Balancer
- **Database**: AWS RDS (MySQL 8.0)
- **Cache**: AWS ElastiCache (Redis)
- **Monitoring**: AWS CloudWatch + Sentry

## 💻 코딩 표준 및 컨벤션

### JavaScript/React 컨벤션
- **들여쓰기**: 스페이스 2칸
- **따옴표**: 작은따옴표('') 사용
- **세미콜론**: 항상 사용
- **네이밍**: camelCase (컴포넌트는 PascalCase)
- **파일명**: kebab-case (컴포넌트는 PascalCase.jsx)

### 폴더 구조 규칙
```
src/
├── components/          # 공통 컴포넌트
├── sections/           # 페이지별 섹션
│   ├── member/         # 도메인별 그룹화
│   │   ├── owner/      # 기능별 분류
│   │   │   ├── data/   # 폼/테이블 설정
│   │   │   └── popup/  # 팝업 컴포넌트
├── utils/              # 유틸리티 함수
├── constant/           # 상수 정의
└── store/             # Redux store
```

### CSS/Styling 규칙
- **스타일링**: Material-UI styled() 또는 sx prop 사용
- **전역 스타일**: themes/ 폴더에서 관리
- **컴포넌트 스타일**: 인라인 스타일 지양, styled component 사용
- **반응형**: Mobile First 접근법

## 🔧 개발 프로세스

### Git 워크플로우
- **브랜치 전략**: Git Flow 기반
- **메인 브랜치**: `main` (운영), `dev` (개발)
- **피처 브랜치**: `feature/FRONT24-000-description`
- **릴리즈 브랜치**: `release/v2.1.0`

### 커밋 컨벤션
```
[FRONT24-000] 커밋 메시지

- 기능 추가/수정 내용 상세 설명
- 버그 수정 내용
- 기타 변경사항
```

### 코드 리뷰 규칙
- **필수 리뷰**: 모든 PR은 최소 1명 이상 승인 필요
- **셀프 리뷰**: PR 생성 전 본인이 먼저 검토
- **리뷰 포인트**: 기능, 성능, 보안, 코드 품질
- **리뷰 시간**: 업무일 기준 24시간 내 리뷰 완료

## 🚀 빌드 및 배포

### 빌드 스크립트
```json
{
  "start": "NODE_ENV=.env.local react-app-rewired start",
  "build-dev": "env-cmd -f .env.development react-app-rewired build",
  "build-stg": "env-cmd -f .env.stage react-app-rewired build",
  "build-prod": "env-cmd -f .env.production react-app-rewired build"
}
```

### 배포 프로세스
1. **개발 환경**: dev 브랜치 자동 배포
2. **스테이지 환경**: release 브랜치 수동 배포
3. **운영 환경**: main 브랜치 승인 후 배포
4. **롤백**: 문제 발생시 이전 버전으로 즉시 롤백

### 환경별 설정
- **API 엔드포인트**: 환경별 .env 파일로 관리
- **빌드 최적화**: 운영환경만 코드 압축 및 최적화
- **소스맵**: 개발/스테이지만 소스맵 생성

## 🔒 보안 및 품질 관리

### 보안 정책
- **API 키**: 환경변수로만 관리, 코드에 하드코딩 금지
- **인증**: JWT 토큰 기반 인증, localStorage 저장 금지
- **HTTPS**: 모든 환경에서 HTTPS 강제
- **XSS 방지**: 사용자 입력값 sanitization 필수

### 의존성 관리
- **보안 취약점**: 주간 npm audit 실행
- **버전 업데이트**: 분기별 주요 패키지 업데이트 검토
- **라이센스 검토**: 새로운 패키지 도입 시 라이센스 확인
- **Bundle 분석**: 월별 번들 크기 분석 및 최적화

### 성능 관리
- **Core Web Vitals**: LCP < 2.5s, FID < 100ms, CLS < 0.1
- **번들 크기**: 초기 로드 < 500KB
- **이미지 최적화**: WebP 포맷 사용, lazy loading 적용
- **코드 스플리팅**: 라우트별 청크 분할

## 🧪 테스트 및 품질 보증

### 테스트 전략
- **Unit Test**: 유틸리티 함수 및 비즈니스 로직
- **Integration Test**: API 연동 및 컴포넌트 상호작용
- **E2E Test**: 핵심 사용자 플로우
- **테스트 커버리지**: 최소 70% 이상 유지

### 코드 품질 도구
- **ESLint**: 코딩 표준 검사
- **Prettier**: 코드 포맷팅 자동화
- **Husky**: Git hooks로 commit 전 검사
- **SonarQube**: 코드 품질 및 보안 취약점 분석

### 성능 모니터링
- **Error Tracking**: Sentry를 통한 에러 모니터링
- **Performance**: Lighthouse CI 자동화
- **User Analytics**: GA4, Hotjar를 통한 사용자 행동 분석
- **APM**: 실시간 성능 지표 모니터링

## 📚 문서화 및 지식 관리

### 기술 문서화
- **API 문서**: Swagger/OpenAPI 명세서
- **컴포넌트 문서**: Storybook 활용
- **아키텍처 문서**: Markdown 형태로 관리
- **변경 이력**: CHANGELOG.md 파일 관리

### 지식 공유
- **Tech Talk**: 월 1회 기술 세미나
- **Code Review**: 지식 공유 목적의 교육적 리뷰
- **Spike**: 새로운 기술 도입 시 사전 연구
- **Retrospective**: 스프린트 회고를 통한 개선점 도출

## 🔄 CI/CD 파이프라인

### 자동화 프로세스
- **Code Quality**: ESLint, Prettier 검사
- **Security**: npm audit, dependency check
- **Build**: 환경별 빌드 및 테스트
- **Deploy**: AWS CodeDeploy를 통한 자동 배포

### 모니터링 및 알림
- **Build Status**: Slack 알림 연동
- **Error Alert**: 운영 에러 시 즉시 알림
- **Performance Alert**: 성능 지표 임계값 초과시 알림
- **Uptime Monitoring**: 서비스 가용성 24시간 모니터링

---
**최종 업데이트**: 2024년 9월 24일
**문의사항**: dev@coolstay.co.kr