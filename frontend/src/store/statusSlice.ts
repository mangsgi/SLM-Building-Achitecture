import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type TrainingStatus = 'IDLE' | 'TRAINING' | 'COMPLETED';

interface StatusState {
  trainingStatus: TrainingStatus;
  mlflowUrl: string | null;
  task_id: string | null;
  error: string | null; // 오류 메시지를 저장할 필드 추가
}

// localStorage에서 상태를 불러오거나 기본값으로 초기화
const getInitialState = (): StatusState => {
  const storedStatus = localStorage.getItem(
    'trainingStatus',
  ) as TrainingStatus | null;
  const storedMlflowUrl = localStorage.getItem('mlflowUrl');
  const storedTaskId = localStorage.getItem('task_id');

  // COMPLETED 상태는 페이지 새로고침 시 IDLE로 초기화
  if (storedStatus && storedStatus !== 'COMPLETED') {
    return {
      trainingStatus: storedStatus,
      mlflowUrl: storedStatus === 'TRAINING' ? storedMlflowUrl : null,
      task_id: storedStatus === 'TRAINING' ? storedTaskId : null,
      error: null, // error 필드 초기화
    };
  }
  return {
    trainingStatus: 'IDLE',
    mlflowUrl: null,
    task_id: null,
    error: null,
  };
};

const statusSlice = createSlice({
  name: 'status',
  initialState: getInitialState(),
  reducers: {
    startTraining: (
      state,
      action: PayloadAction<{ mlflowUrl: string; task_id: string }>,
    ) => {
      state.trainingStatus = 'TRAINING';
      state.mlflowUrl = action.payload.mlflowUrl;
      state.task_id = action.payload.task_id;
      state.error = null; // 학습 시작 시 에러 초기화
      localStorage.setItem('trainingStatus', 'TRAINING');
      localStorage.setItem('mlflowUrl', action.payload.mlflowUrl);
      localStorage.setItem('task_id', action.payload.task_id);
    },
    completeTraining: (state) => {
      state.trainingStatus = 'COMPLETED';
      state.mlflowUrl = null;
      state.task_id = null;
      state.error = null;
      localStorage.setItem('trainingStatus', 'COMPLETED');
      localStorage.removeItem('mlflowUrl');
      localStorage.removeItem('task_id');
    },
    // 학습 실패 액션 추가
    failTraining: (state, action: PayloadAction<{ message: string }>) => {
      state.trainingStatus = 'COMPLETED'; // 상태는 완료(Blue)로 변경
      state.error = action.payload.message; // 에러 메시지 저장
      state.mlflowUrl = null;
      state.task_id = null;
      localStorage.setItem('trainingStatus', 'COMPLETED');
      localStorage.removeItem('mlflowUrl');
      localStorage.removeItem('task_id');
    },
    resetStatus: (state) => {
      state.trainingStatus = 'IDLE';
      state.mlflowUrl = null;
      state.task_id = null;
      state.error = null; // 리셋 시 에러 초기화
      localStorage.removeItem('trainingStatus');
      localStorage.removeItem('mlflowUrl');
      localStorage.removeItem('task_id');
    },
  },
});

export const { startTraining, completeTraining, failTraining, resetStatus } =
  statusSlice.actions;
export default statusSlice.reducer;
