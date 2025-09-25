# 꿀스테이 배포 가이드

## 📋 개요
꿀스테이 CMS 배포 프로세스 및 환경 관리 가이드입니다.

## 🌐 환경 구성

### 환경별 특성
| 환경 | 브랜치 | 도메인 | 용도 |
|------|--------|---------|------|
| **Local** | feature/* | localhost:3000 | 개발자 로컬 개발 |
| **Development** | dev | dev-cms.coolstay.co.kr | 통합 개발 테스트 |
| **Staging** | release/* | stage-cms.coolstay.co.kr | QA 및 운영팀 검수 |
| **Production** | main | cms.coolstay.co.kr | 실제 서비스 |

### 환경 설정 파일
```bash
# 로컬 개발
.env.local

# 개발 환경
.env.development
REACT_APP_API_URL=https://dev-api.coolstay.co.kr
REACT_APP_ENV=development

# 스테이징 환경
.env.stage
REACT_APP_API_URL=https://stage-api.coolstay.co.kr
REACT_APP_ENV=staging

# 운영 환경
.env.production
REACT_APP_API_URL=https://api.coolstay.co.kr
REACT_APP_ENV=production
```

## 🔧 빌드 프로세스

### 빌드 스크립트 실행
```bash
# 개발 환경 빌드
npm run build-dev

# 스테이징 환경 빌드
npm run build-stg

# 운영 환경 빌드
npm run build-prod
```

### 빌드 최적화
- **코드 스플리팅**: 라우트별 청크 분할
- **Tree Shaking**: 사용하지 않는 코드 제거
- **이미지 최적화**: WebP 변환, 압축
- **CSS 최적화**: 중복 스타일 제거, 압축
- **번들 분석**: webpack-bundle-analyzer로 크기 분석

### 빌드 산출물
```
build/
├── static/
│   ├── css/           # 스타일 파일
│   ├── js/            # JavaScript 청크
│   └── media/         # 이미지, 폰트 등
├── index.html         # 메인 HTML
└── asset-manifest.json # 에셋 매니페스트
```

## 🚀 CI/CD 파이프라인

### GitHub Actions 워크플로우

#### Development 자동 배포
```yaml
# .github/workflows/deploy-dev.yml
name: Deploy to Development
on:
  push:
    branches: [dev]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
      - name: Setup Node.js
      - name: Install dependencies
      - name: Run tests
      - name: Build for development
      - name: Deploy to S3
      - name: Invalidate CloudFront
```

#### Staging 수동 배포
```yaml
# .github/workflows/deploy-stage.yml
name: Deploy to Staging
on:
  workflow_dispatch:
    inputs:
      branch:
        description: 'Branch to deploy'
        required: true
        default: 'release/v2.1.0'
```

#### Production 승인 배포
```yaml
# .github/workflows/deploy-prod.yml
name: Deploy to Production
on:
  release:
    types: [published]
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Manual approval required
```

### 배포 단계
1. **코드 품질 검사**: ESLint, Prettier
2. **보안 검사**: npm audit, Snyk
3. **유닛 테스트**: Jest 테스트 실행
4. **빌드**: 환경별 빌드 실행
5. **S3 업로드**: 빌드 산출물 업로드
6. **CloudFront 캐시 무효화**: CDN 캐시 갱신
7. **헬스체크**: 배포 후 상태 확인
8. **Slack 알림**: 배포 완료/실패 알림

## 📦 AWS 인프라 구성

### S3 버킷 구성
```
coolstay-cms-dev/        # 개발 환경
├── static/
├── index.html
└── error.html

coolstay-cms-stage/      # 스테이징 환경
├── static/
├── index.html
└── error.html

coolstay-cms-prod/       # 운영 환경
├── static/
├── index.html
└── error.html
```

### CloudFront 설정
- **캐싱 정책**:
  - HTML: 0초 (즉시 갱신)
  - CSS/JS: 1년 (해시 기반 캐시 무효화)
  - 이미지: 30일
- **압축**: Gzip, Brotli 압축 활성화
- **보안 헤더**: HSTS, CSP, X-Frame-Options
- **Error Pages**: 404 → index.html (SPA 라우팅)

### Route 53 DNS
```
# 개발 환경
dev-cms.coolstay.co.kr → CloudFront Distribution

# 스테이징 환경
stage-cms.coolstay.co.kr → CloudFront Distribution

# 운영 환경
cms.coolstay.co.kr → CloudFront Distribution
```

## 🔐 환경별 보안 설정

### IAM 권한 관리
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::coolstay-cms-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudfront:CreateInvalidation"
      ],
      "Resource": "*"
    }
  ]
}
```

### 환경 변수 보안
- **GitHub Secrets**: API 키, AWS 자격증명
- **Parameter Store**: 환경별 설정값
- **KMS**: 민감 정보 암호화
- **VPC**: 네트워크 격리 (백엔드 API)

## 📊 배포 모니터링

### 배포 메트릭
- **배포 빈도**: 주간 배포 횟수
- **배포 시간**: 평균 배포 소요 시간
- **성공률**: 배포 성공/실패 비율
- **롤백률**: 문제로 인한 롤백 비율

### 알림 설정
```yaml
# Slack 알림
channels:
  - name: "#dev-alert"
    events: [deploy_start, deploy_success, deploy_failure]
  - name: "#operations"
    events: [production_deploy, rollback]

# 이메일 알림
recipients:
  - dev-team@coolstay.co.kr
  - ops-team@coolstay.co.kr
```

### 헬스체크
- **기본 헬스체크**: HTTP 200 응답 확인
- **기능 테스트**: 로그인, 주요 페이지 접근
- **성능 테스트**: Core Web Vitals 측정
- **API 연동**: 백엔드 API 연결 상태

## 🔄 롤백 프로세스

### 자동 롤백 조건
- **4xx/5xx 에러율**: 5% 초과시 자동 롤백
- **응답 시간**: 평균 3초 초과시 자동 롤백
- **헬스체크 실패**: 연속 3회 실패시 자동 롤백

### 수동 롤백 절차
1. **이슈 확인**: 문제 상황 파악
2. **롤백 결정**: 팀 리드 승인
3. **이전 버전 배포**: 마지막 정상 버전으로 복구
4. **검증**: 롤백 후 기능 정상 동작 확인
5. **원인 분석**: 문제 원인 파악 및 개선

### 롤백 스크립트
```bash
#!/bin/bash
# rollback.sh

ENVIRONMENT=$1
PREVIOUS_VERSION=$2

echo "Rolling back $ENVIRONMENT to $PREVIOUS_VERSION"

# S3에서 이전 버전 파일 복원
aws s3 sync s3://coolstay-cms-backup/$PREVIOUS_VERSION/ s3://coolstay-cms-$ENVIRONMENT/

# CloudFront 캐시 무효화
aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths "/*"

echo "Rollback completed"
```

## 🧪 배포 전 테스트

### 자동화 테스트
- **Unit Tests**: Jest 단위 테스트
- **Integration Tests**: API 연동 테스트
- **E2E Tests**: Cypress 종단간 테스트
- **Performance Tests**: Lighthouse CI

### 수동 테스트 체크리스트
- [ ] 로그인/로그아웃 정상 동작
- [ ] 주요 페이지 접근 및 기능 테스트
- [ ] 다양한 브라우저 호환성 확인
- [ ] 모바일 반응형 동작 확인
- [ ] 외부 API 연동 상태 확인

## 📋 배포 체크리스트

### 배포 전
- [ ] 코드 리뷰 완료
- [ ] 모든 테스트 통과
- [ ] 빌드 에러 없음
- [ ] 환경 설정 확인
- [ ] 배포 승인 획득

### 배포 중
- [ ] 배포 프로세스 모니터링
- [ ] 에러 로그 실시간 확인
- [ ] 성능 지표 관찰
- [ ] 사용자 피드백 모니터링

### 배포 후
- [ ] 헬스체크 통과
- [ ] 핵심 기능 검증
- [ ] 성능 지표 정상
- [ ] 에러율 정상 범위
- [ ] 배포 완료 알림 발송

---
**최종 업데이트**: 2024년 9월 24일
**문의사항**: devops@coolstay.co.kr