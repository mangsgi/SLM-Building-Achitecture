# SLM Building Architecture - Frontend

> 딥러닝 모델 아키텍처를 시각적으로 설계하고 구성할 수 있는 웹 애플리케이션의 프론트엔드를 구현한 디렉토리

- 사용자는 드래그 앤 드롭 인터페이스를 통해 다양한 레이어와 블록을 조합하여 자신만의 모델을 만들고, 파라미터를 수정하며, 전체 구조를 한눈에 파악할 수 있다.

## ✨ 주요 기능

- **시각적 모델 설계:** 노드 기반의 캔버스 위에서 컴포넌트를 드래그 앤 드롭하여 모델 아키텍처를 직관적으로 설계
- **다양한 컴포넌트 제공:** `TransformerBlock`, `TokenEmbedding`, `Attention` 등 SLM(Small Language Model) 구축에 필요한 다양한 레이어와 블록 제공
- **실시간 파라미터 수정:** 각 컴포넌트의 세부 파라미터를 사이드바에서 실시간으로 수정하고 즉시 반영
- **상태 관리:** Redux와 ReactFlow를 활용하여 캔버스의 상태와 노드 데이터를 효율적으로 관리

## 🛠️ 기술 스택

- **Framework/Library:** React, Redux Toolkit, ReactFlow
- **Language:** TypeScript
- **Build Tool:** Vite
- **CSS:** Tailwind CSS
- **Linting/Formatting:** ESLint, Prettier
- **Git Hooks:** Husky, lint-staged

## 🚀 시작하기

### 1. 저장소 복제 (Clone Repository)

```bash
git clone https://github.com/your-repository/SLM-Building-Achitecture.git
cd SLM-Building-Achitecture/frontend
```

### 2. 의존성 설치 (Install Dependencies)

```bash
npm install
```

### 3. 개발 서버 실행 (Run Development Server)

```bash
npm run dev
```

- 이제 브라우저에서 `http://localhost:5173` (또는 Vite가 지정한 다른 포트)으로 접속하여 애플리케이션을 확인

## 📜 사용 가능한 스크립트

- `npm run dev`: 개발 모드로 Vite 서버를 실행합니다.
- `npm run build`: 프로덕션용으로 프로젝트를 빌드합니다.
- `npm run lint`: ESLint를 사용하여 코드 스타일을 검사합니다.
- `npm run preview`: 프로덕션 빌드 결과물을 로컬에서 미리 봅니다.

## 📁 프로젝트 구조

```
frontend/
├── src/
│   ├── assets/           # 이미지, 폰트 등 정적 에셋
│   ├── ui-component/     # 공통 UI 컴포넌트 (버튼, 모달 등)
│   ├── constants/        # 상수 관리
│   ├── nodes/            # ReactFlow 노드 컴포넌트
│   │   └── components/   # 노드 내부에서 사용되는 공통 컴포넌트
│   ├── store/            # Redux 상태 관리 (리듀서, 액션 등)
│   ├── App.tsx           # 메인 애플리케이션 컴포넌트
│   ├── FlowCanvas.tsx    # ReactFlow의 메인 코드
│   ├── TestPage.tsx      # 모델 테스트 페이지
│   ├── DataSelection.tsx # 데이터셋 선택 페이지
│   └── main.tsx          # 애플리케이션 진입점
├── tailwind.config.mjs   # Tailwind CSS 설정
└── vite.config.ts        # Vite 설정
```
