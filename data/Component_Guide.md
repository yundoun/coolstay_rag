# 꿀스테이 컴포넌트 가이드

## 📋 개요
꿀스테이 CMS의 공통 컴포넌트 사용법 및 UI 패턴을 안내합니다.

## 🔍 DefaultSearchForm 컴포넌트

### 기본 사용법
CMS의 모든 검색 폼에서 사용하는 표준 컴포넌트입니다.

```jsx
import DefaultSearchForm from 'components/common/DefaultSearchForm';

<DefaultSearchForm
  data={searchFormData}
  onSearch={handleSearch}
  onReset={handleReset}
/>
```

### 데이터 구조
```javascript
// src/sections/member/owner/data/ownerManagementSearchForm.js
export const data = {
  width: '100%',
  height: '100%',
  row: 3,                    // 행 수
  columnsGroups: [           // 행별 컬럼 그룹
    [                        // 첫 번째 행
      {
        id: 'kindText',
        title: '유형',
        type: 'label',       // label, selectBox, radio, text, button
        typeValue: [],
        defaultValue: '',
        customStyle: {},
        callbackFunction: null
      }
    ]
  ]
}
```

### 필드 타입
- **label**: 라벨 텍스트
- **selectBox**: 드롭다운 선택
- **radio**: 라디오 버튼 그룹
- **text**: 텍스트 입력
- **button**: 검색/리셋 버튼
- **colspan**: 여러 컬럼 조합
- **rowspan**: 여러 행 조합

### 실제 사용 사례
```javascript
// 업주 관리 검색 폼 (src/sections/member/owner/data/ownerManagementSearchForm.js)
{
  id: 'type',
  title: '',
  type: 'radio',
  typeValue: [
    {
      id: 'PARTNER_OWNER',
      type: 'option',
      text: '파트너 업주',
      value: 'PARTNER_OWNER'
    }
  ],
  defaultValue: 'PARTNER_OWNER',
  customStyle: { width: '100%' }
}
```

## 📊 DefaultTable 컴포넌트

### 기본 사용법
```jsx
import DefaultTable from 'components/common/DefaultTable';

<DefaultTable
  columns={columnsData}
  rows={rowsData}
  onRowClick={handleRowClick}
  pagination={true}
  selectable={true}
/>
```

### 컬럼 데이터 구조
```javascript
// src/sections/member/owner/data/ownerColumnsData.js
export const columns = [
  {
    id: 'userId',
    type: 'text',
    label: '아이디',
    width: 120,
    sortable: true,
    clickable: true,
    align: 'left'
  },
  {
    id: 'status',
    type: 'chip',           // chip, button, text, number, date
    label: '상태',
    width: 100,
    chipColors: {
      'STABLE': 'success',
      'BLOCKED': 'error',
      'WAITING': 'warning'
    }
  }
];
```

### 컬럼 타입별 특징
- **text**: 기본 텍스트 표시
- **chip**: 상태별 색상 칩
- **button**: 클릭 가능한 버튼
- **number**: 숫자 포맷팅 (천단위 콤마)
- **date**: 날짜 포맷팅 (YYYY-MM-DD)
- **subColumn**: 계층적 헤더 구조

### 계층적 헤더 구조
```javascript
{
  id: 'bookHistory',
  type: 'subColumn',
  label: '사용이력',
  colspan: 9,
  subColumns: [
    { id: 'bookCnt', type: 'text', label: '예약' },
    { id: 'refundCnt', type: 'text', label: '취소' }
  ]
}
```

## 🖼️ PopupComponent

### 기본 구조
```jsx
import PopupComponent from 'components/common/PopupComponent';

<PopupComponent
  open={isOpen}
  onClose={handleClose}
  title="팝업 제목"
  size="large"              // small, medium, large, fullscreen
  showCloseButton={true}
>
  <PopupContent />
</PopupComponent>
```

### 팝업 크기 설정
- **small**: 400px x 300px
- **medium**: 600px x 500px
- **large**: 800px x 700px
- **fullscreen**: 화면 전체

### 실제 팝업 예시
```jsx
// src/sections/member/owner/popup/ownerInfoPopup/OwnerInfoPopup.jsx
const OwnerInfoPopup = ({ open, onClose, ownerData, mode }) => {
  return (
    <PopupComponent
      open={open}
      onClose={onClose}
      title={mode === 'edit' ? '업주 정보 수정' : '업주 정보 등록'}
      size="large"
    >
      <OwnerInfoForm
        data={ownerData}
        mode={mode}
        onSubmit={handleSubmit}
      />
    </PopupComponent>
  );
};
```

## 🎛️ 폼 관련 컴포넌트

### FormikProvider 사용
```jsx
import { Formik, Form } from 'formik';
import * as Yup from 'yup';

const validationSchema = Yup.object({
  userId: Yup.string().required('아이디는 필수입니다'),
  email: Yup.string().email('올바른 이메일 형식이 아닙니다')
});

<Formik
  initialValues={initialValues}
  validationSchema={validationSchema}
  onSubmit={handleSubmit}
>
  <Form>
    <FormField name="userId" label="아이디" />
    <FormField name="email" label="이메일" type="email" />
  </Form>
</Formik>
```

### FormField 컴포넌트
```jsx
import FormField from 'components/common/FormField';

<FormField
  name="storeName"
  label="숙소명"
  type="text"               // text, email, password, select, textarea
  placeholder="숙소명을 입력하세요"
  required={true}
  options={selectOptions}   // select 타입일 때
  rows={4}                  // textarea 타입일 때
/>
```

## 📁 파일 업로드 컴포넌트

### FileUpload 사용법
```jsx
import FileUpload from 'components/common/FileUpload';

<FileUpload
  accept=".jpg,.jpeg,.png"
  multiple={true}
  maxSize={5 * 1024 * 1024}  // 5MB
  onUpload={handleFileUpload}
  onError={handleError}
>
  <div>클릭하여 파일을 선택하거나 드래그하세요</div>
</FileUpload>
```

### 이미지 미리보기
```jsx
<ImagePreview
  src={imageUrl}
  alt="미리보기"
  width={200}
  height={150}
  onRemove={handleRemove}
  showRemoveButton={true}
/>
```

## 🔢 데이터 표시 컴포넌트

### NumberFormat 컴포넌트
```jsx
import NumberFormat from 'components/common/NumberFormat';

<NumberFormat
  value={1234567}
  format="currency"         // number, currency, percentage
  currency="KRW"
  locale="ko-KR"
/>
// 출력: ₩1,234,567
```

### DateFormat 컴포넌트
```jsx
import DateFormat from 'components/common/DateFormat';

<DateFormat
  value={new Date()}
  format="YYYY-MM-DD HH:mm"
  relative={false}
/>
```

## 📋 리스트 및 카드 컴포넌트

### CardList 컴포넌트
```jsx
<CardList
  data={cardData}
  columns={3}               // 그리드 컬럼 수
  gap={16}                 // 카드 간격
  renderCard={(item) => (
    <Card key={item.id}>
      <CardContent>
        <Typography variant="h6">{item.title}</Typography>
        <Typography variant="body2">{item.description}</Typography>
      </CardContent>
    </Card>
  )}
/>
```

### VirtualizedList 컴포넌트
```jsx
import VirtualizedList from 'components/common/VirtualizedList';

<VirtualizedList
  items={largeDataSet}
  itemHeight={80}
  containerHeight={400}
  renderItem={({ item, index }) => (
    <ListItem key={item.id}>
      <ListItemText primary={item.name} secondary={item.description} />
    </ListItem>
  )}
/>
```

## 🎨 스타일링 가이드

### Theme 사용
```jsx
import { useTheme } from '@mui/material/styles';

const MyComponent = () => {
  const theme = useTheme();

  return (
    <Box
      sx={{
        backgroundColor: theme.palette.primary.main,
        color: theme.palette.primary.contrastText,
        padding: theme.spacing(2),
        borderRadius: theme.shape.borderRadius
      }}
    >
      Content
    </Box>
  );
};
```

### 공통 스타일 패턴
```jsx
// 페이지 컨테이너
const pageStyles = {
  container: {
    padding: 3,
    backgroundColor: 'background.paper',
    borderRadius: 1,
    boxShadow: 1
  }
};

// 섹션 헤더
const headerStyles = {
  title: {
    fontSize: '1.5rem',
    fontWeight: 600,
    marginBottom: 2
  }
};
```

## 🔧 유틸리티 컴포넌트

### Loading 컴포넌트
```jsx
import Loading from 'components/common/Loading';

<Loading
  show={isLoading}
  message="데이터를 불러오는 중..."
  overlay={true}
/>
```

### ConfirmDialog 컴포넌트
```jsx
import ConfirmDialog from 'components/common/ConfirmDialog';

<ConfirmDialog
  open={confirmOpen}
  title="삭제 확인"
  content="정말로 삭제하시겠습니까?"
  onConfirm={handleConfirm}
  onCancel={handleCancel}
  confirmText="삭제"
  cancelText="취소"
/>
```

## 📱 반응형 컴포넌트

### ResponsiveContainer 사용
```jsx
import ResponsiveContainer from 'components/common/ResponsiveContainer';

<ResponsiveContainer
  mobile={<MobileView />}
  tablet={<TabletView />}
  desktop={<DesktopView />}
  breakpoints={{
    mobile: 'sm',
    tablet: 'md',
    desktop: 'lg'
  }}
/>
```

### 미디어 쿼리 Hook
```jsx
import { useMediaQuery } from '@mui/material';
import { useTheme } from '@mui/material/styles';

const MyComponent = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Box sx={{
      flexDirection: isMobile ? 'column' : 'row'
    }}>
      {/* 반응형 레이아웃 */}
    </Box>
  );
};
```

---
**최종 업데이트**: 2024년 9월 24일
**문의사항**: frontend@coolstay.co.kr