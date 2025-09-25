# 꿀스테이 CMS 아키텍처 가이드

## 📋 개요
꿀스테이 CMS의 전체 아키텍처 구조 및 설계 원칙을 안내합니다.

## 🏗️ 전체 시스템 아키텍처

### 클라이언트-서버 구조
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   React CMS     │◄──►│   Backend API   │◄──►│   Database      │
│   (Frontend)    │    │   (Java/Node)   │    │   (MySQL/Redis) │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌────▼────┐             ┌────▼────┐             ┌────▼────┐
    │ AWS S3  │             │ AWS ECS │             │ AWS RDS │
    │CloudFront│             │   ALB   │             │ElastiCache│
    └─────────┘             └─────────┘             └─────────┘
```

### 레이어드 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│         (Pages, Components, Routes, State Management)       │
├─────────────────────────────────────────────────────────────┤
│                     Business Layer                          │
│            (Services, Utils, Hooks, Context)                │
├─────────────────────────────────────────────────────────────┤
│                    Data Access Layer                        │
│               (API Calls, Local Storage, Cache)             │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                       │
│            (HTTP Client, Authentication, Logging)           │
└─────────────────────────────────────────────────────────────┘
```

## 📁 프로젝트 구조

### 폴더 구조
```
src/
├── components/                 # 공통 컴포넌트
│   ├── common/                # 범용 컴포넌트
│   │   ├── DefaultSearchForm/ # 검색 폼
│   │   ├── DefaultTable/      # 테이블
│   │   └── PopupComponent/    # 팝업
│   ├── @extended/             # 확장 컴포넌트
│   └── third-party/           # 서드파티 래퍼
├── sections/                  # 페이지별 섹션 (도메인별 분리)
│   ├── member/                # 회원 관리
│   │   ├── owner/            # 업주 관리
│   │   ├── member/           # 일반회원 관리
│   │   ├── non-member/       # 비회원 관리
│   │   └── admin/            # 관리자 관리
│   ├── cs-service/           # 고객 서비스
│   │   ├── inquiry/          # 문의 관리
│   │   ├── counseling/       # 상담 관리
│   │   └── history/          # 이력 관리
│   ├── coupon-mileage/       # 쿠폰/마일리지
│   ├── marketing/            # 마케팅 관리
│   └── statistics/           # 통계/리포트
├── store/                    # Redux Store
│   ├── reducers/            # 리듀서
│   ├── slices/              # RTK Query 슬라이스
│   └── middleware/          # 미들웨어
├── utils/                   # 유틸리티 함수
├── hooks/                   # 커스텀 훅
├── contexts/                # React Context
├── constants/               # 상수 정의
├── themes/                  # Material-UI 테마
├── assets/                  # 정적 자원
└── routes/                  # 라우팅 설정
```

### 섹션별 세부 구조
```
sections/member/owner/
├── OwnerList.jsx             # 메인 리스트 페이지
├── component/                # 페이지 전용 컴포넌트
├── popup/                    # 팝업 컴포넌트
│   ├── popupComponent/       # 공통 팝업
│   ├── ownerInfoPopup/       # 업주 정보 팝업
│   └── storeInfoPopup/       # 숙소 정보 팝업
└── data/                     # 폼/테이블 설정 데이터
    ├── ownerColumnsData.js   # 테이블 컬럼 정의
    ├── ownerManagementSearchForm.js  # 검색폼 정의
    └── formData.js           # 폼 필드 정의
```

## 🔄 상태 관리 아키텍처

### Redux Toolkit 구조
```javascript
// store/index.js
import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';

const store = configureStore({
  reducer: {
    auth: authSlice,
    menu: menuSlice,
    snackbar: snackbarSlice,
    api: apiSlice.reducer
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER]
      }
    }).concat(apiSlice.middleware)
});
```

### RTK Query API 슬라이스
```javascript
// store/slices/memberApi.js
export const memberApi = createApi({
  reducerPath: 'memberApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/member',
    prepareHeaders: (headers, { getState }) => {
      const token = getState().auth.token;
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      return headers;
    }
  }),
  tagTypes: ['Member', 'Owner'],
  endpoints: (builder) => ({
    getOwners: builder.query({
      query: (params) => ({
        url: '/owners',
        params
      }),
      providesTags: ['Owner']
    }),
    updateOwner: builder.mutation({
      query: ({ id, ...patch }) => ({
        url: `/owners/${id}`,
        method: 'PATCH',
        body: patch
      }),
      invalidatesTags: ['Owner']
    })
  })
});
```

### Context API 활용
```javascript
// contexts/JWTContext.js
export const JWTProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  useEffect(() => {
    const init = async () => {
      try {
        const token = window.localStorage.getItem('serviceToken');
        if (token && isValidToken(token)) {
          const decoded = jwt_decode(token);
          dispatch({ type: INITIALIZE, payload: { user: decoded } });
        } else {
          dispatch({ type: INITIALIZE, payload: { user: null } });
        }
      } catch (err) {
        dispatch({ type: INITIALIZE, payload: { user: null } });
      }
    };
    init();
  }, []);

  return (
    <JWTContext.Provider value={{ ...state, login, logout }}>
      {children}
    </JWTContext.Provider>
  );
};
```

## 🌐 라우팅 아키텍처

### Route 구조
```javascript
// routes/MainRoutes.js
const MainRoutes = {
  path: '/',
  element: <MainLayout />,
  children: [
    {
      path: '/member',
      children: [
        {
          path: 'owner',
          element: <OwnerManagement />
        },
        {
          path: 'member',
          element: <MemberManagement />
        }
      ]
    },
    {
      path: '/cs-service',
      children: [
        {
          path: 'inquiry',
          element: <InquiryManagement />
        }
      ]
    }
  ]
};
```

### 권한 기반 라우팅
```javascript
// components/ProtectedRoute.js
const ProtectedRoute = ({ children, roles = [] }) => {
  const { user } = useAuth();

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (roles.length > 0 && !roles.includes(user.role)) {
    return <Navigate to="/403" replace />;
  }

  return children;
};

// 사용법
<ProtectedRoute roles={['ADMIN', 'MANAGER']}>
  <AdminPanel />
</ProtectedRoute>
```

## 🔌 API 통신 아키텍처

### Axios 설정
```javascript
// utils/axios.js
const axiosServices = axios.create({
  baseURL: process.env.REACT_APP_API_URL,
  timeout: 10000
});

// Request Interceptor
axiosServices.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('serviceToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response Interceptor
axiosServices.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('serviceToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

### API 서비스 레이어
```javascript
// services/memberService.js
export const memberService = {
  getOwners: async (params) => {
    const response = await axiosServices.get('/member/owners', { params });
    return response.data;
  },

  getOwner: async (id) => {
    const response = await axiosServices.get(`/member/owners/${id}`);
    return response.data;
  },

  updateOwner: async (id, data) => {
    const response = await axiosServices.patch(`/member/owners/${id}`, data);
    return response.data;
  }
};
```

## 🎨 UI/UX 아키텍처

### Material-UI 테마 시스템
```javascript
// themes/index.js
export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0'
    },
    secondary: {
      main: '#dc004e'
    }
  },
  typography: {
    fontFamily: '"Pretendard", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.125rem',
      fontWeight: 600
    }
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8
        }
      }
    }
  }
});
```

### 반응형 디자인 시스템
```javascript
// themes/breakpoints.js
export const breakpoints = {
  values: {
    xs: 0,
    sm: 600,
    md: 900,
    lg: 1200,
    xl: 1536
  }
};

// 사용법
const useStyles = makeStyles((theme) => ({
  container: {
    padding: theme.spacing(2),
    [theme.breakpoints.up('md')]: {
      padding: theme.spacing(4)
    }
  }
}));
```

## 🔒 보안 아키텍처

### 인증/인가 플로우
```
1. 로그인 요청 → Backend 인증
2. JWT 토큰 발급 → Local Storage 저장
3. API 요청시 → Authorization Header 자동 추가
4. 토큰 만료시 → 자동 로그아웃 처리
5. 권한 확인 → Route/Component 레벨 접근 제어
```

### XSS 방어
```javascript
// utils/sanitizer.js
import DOMPurify from 'dompurify';

export const sanitizeHtml = (html) => {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u'],
    ALLOWED_ATTR: []
  });
};

// 사용법
<div dangerouslySetInnerHTML={{
  __html: sanitizeHtml(userContent)
}} />
```

## 📊 성능 최적화 아키텍처

### 코드 스플리팅
```javascript
// routes/LazyRoutes.js
const OwnerManagement = lazy(() => import('sections/member/owner/OwnerList'));
const MemberManagement = lazy(() => import('sections/member/member/MemberList'));

// 사용법
<Suspense fallback={<Loading />}>
  <OwnerManagement />
</Suspense>
```

### 메모이제이션 패턴
```javascript
// hooks/useTableData.js
export const useTableData = (data, searchParams) => {
  const filteredData = useMemo(() => {
    return data.filter(item =>
      item.name.toLowerCase().includes(searchParams.keyword.toLowerCase())
    );
  }, [data, searchParams.keyword]);

  const sortedData = useMemo(() => {
    return [...filteredData].sort((a, b) =>
      searchParams.sortField === 'name'
        ? a.name.localeCompare(b.name)
        : new Date(a.createdAt) - new Date(b.createdAt)
    );
  }, [filteredData, searchParams.sortField]);

  return { filteredData, sortedData };
};
```

### 가상화 리스트
```javascript
// components/VirtualizedTable.js
import { FixedSizeList as List } from 'react-window';

const VirtualizedTable = ({ items, itemHeight = 60 }) => {
  const Row = useCallback(({ index, style }) => (
    <div style={style}>
      <TableRow data={items[index]} />
    </div>
  ), [items]);

  return (
    <List
      height={400}
      itemCount={items.length}
      itemSize={itemHeight}
      width="100%"
    >
      {Row}
    </List>
  );
};
```

## 🧪 테스트 아키텍처

### 테스트 피라미드
```
                    E2E Tests
                  (Cypress)
                ▲
               ▲ ▲
              ▲   ▲
           Integration Tests
          (React Testing Library)
            ▲               ▲
           ▲                 ▲
          ▲                   ▲
       Unit Tests              ▲
      (Jest + RTL)           ▲
    ▲                       ▲
   ▲_______________________▲
```

### 테스트 구조
```javascript
// __tests__/components/DefaultTable.test.js
describe('DefaultTable', () => {
  const mockData = [
    { id: 1, name: 'Test User', email: 'test@example.com' }
  ];

  it('renders table with data', () => {
    render(<DefaultTable columns={mockColumns} rows={mockData} />);

    expect(screen.getByText('Test User')).toBeInTheDocument();
    expect(screen.getByText('test@example.com')).toBeInTheDocument();
  });

  it('handles row click', () => {
    const handleRowClick = jest.fn();
    render(
      <DefaultTable
        columns={mockColumns}
        rows={mockData}
        onRowClick={handleRowClick}
      />
    );

    fireEvent.click(screen.getByText('Test User'));
    expect(handleRowClick).toHaveBeenCalledWith(mockData[0]);
  });
});
```

## 🔄 데이터 플로우

### 단방향 데이터 플로우
```
Action → Reducer → Store → Component → UI
  ▲                                    │
  │                                    │
  └────── User Interaction ←───────────┘
```

### 비동기 데이터 처리
```javascript
// RTK Query를 이용한 데이터 페칭
const OwnerList = () => {
  const [searchParams, setSearchParams] = useState({});
  const {
    data: owners,
    error,
    isLoading,
    refetch
  } = useGetOwnersQuery(searchParams);

  const handleSearch = (params) => {
    setSearchParams(params);
  };

  if (isLoading) return <Loading />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <DefaultTable
      columns={ownerColumns}
      rows={owners?.data || []}
      onSearch={handleSearch}
    />
  );
};
```

---
**최종 업데이트**: 2024년 9월 24일
**문의사항**: architecture@coolstay.co.kr