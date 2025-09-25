import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type TrainingStatus =
  | 'IDLE'
  | 'SUBMITTING'
  | 'TRAINING'
  | 'COMPLETED'
  | 'ERROR';

interface StatusState {
  trainingStatus: TrainingStatus;
  mlflowUrl: string | null;
  task_id: string | null;
  error: string | null;
}

const getInitialState = (): StatusState => {
  const storedStatus = localStorage.getItem(
    'trainingStatus',
  ) as TrainingStatus | null;
  const storedMlflowUrl = localStorage.getItem('mlflowUrl');
  const storedTaskId = localStorage.getItem('task_id');

  if (storedStatus === 'TRAINING') {
    return {
      trainingStatus: 'TRAINING',
      mlflowUrl: storedMlflowUrl,
      task_id: storedTaskId,
      error: null,
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
    setSubmitting: (state) => {
      state.trainingStatus = 'SUBMITTING';
      state.error = null;
    },
    startTraining: (
      state,
      action: PayloadAction<{ mlflowUrl: string; task_id: string }>,
    ) => {
      state.trainingStatus = 'TRAINING';
      state.mlflowUrl = action.payload.mlflowUrl;
      state.task_id = action.payload.task_id;
      state.error = null;
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
    failTraining: (state, action: PayloadAction<{ message: string }>) => {
      state.trainingStatus = 'ERROR';
      state.error = action.payload.message;
      state.mlflowUrl = null;
      state.task_id = null;
      localStorage.removeItem('trainingStatus');
      localStorage.removeItem('mlflowUrl');
      localStorage.removeItem('task_id');
    },
    resetStatus: (state) => {
      state.trainingStatus = 'IDLE';
      state.mlflowUrl = null;
      state.task_id = null;
      state.error = null;
      localStorage.removeItem('trainingStatus');
      localStorage.removeItem('mlflowUrl');
      localStorage.removeItem('task_id');
    },
  },
});

export const {
  setSubmitting,
  startTraining,
  completeTraining,
  failTraining,
  resetStatus,
} = statusSlice.actions;
export default statusSlice.reducer;
