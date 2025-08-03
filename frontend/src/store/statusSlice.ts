import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type TrainingStatus = 'IDLE' | 'TRAINING' | 'COMPLETED';

interface StatusState {
  trainingStatus: TrainingStatus;
  mlflowUrl: string | null;
}

// localStorage에서 상태를 불러오거나 기본값으로 초기화
const getInitialState = (): StatusState => {
  const storedStatus = localStorage.getItem(
    'trainingStatus',
  ) as TrainingStatus | null;
  const storedMlflowUrl = localStorage.getItem('mlflowUrl');

  // COMPLETED 상태는 페이지 새로고침 시 IDLE로 초기화
  if (storedStatus && storedStatus !== 'COMPLETED') {
    return {
      trainingStatus: storedStatus,
      mlflowUrl: storedStatus === 'TRAINING' ? storedMlflowUrl : null,
    };
  }
  return { trainingStatus: 'IDLE', mlflowUrl: null };
};

const statusSlice = createSlice({
  name: 'status',
  initialState: getInitialState(),
  reducers: {
    startTraining: (state, action: PayloadAction<{ mlflowUrl: string }>) => {
      state.trainingStatus = 'TRAINING';
      state.mlflowUrl = action.payload.mlflowUrl;
      localStorage.setItem('trainingStatus', 'TRAINING');
      localStorage.setItem('mlflowUrl', action.payload.mlflowUrl);
    },
    completeTraining: (state) => {
      state.trainingStatus = 'COMPLETED';
      state.mlflowUrl = null;
      localStorage.setItem('trainingStatus', 'COMPLETED');
      localStorage.removeItem('mlflowUrl');
    },
    resetStatus: (state) => {
      state.trainingStatus = 'IDLE';
      state.mlflowUrl = null;
      localStorage.removeItem('trainingStatus');
      localStorage.removeItem('mlflowUrl');
    },
  },
});

export const { startTraining, completeTraining, resetStatus } =
  statusSlice.actions;
export default statusSlice.reducer;
