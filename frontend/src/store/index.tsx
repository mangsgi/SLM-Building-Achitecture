import { configureStore } from '@reduxjs/toolkit';
import canvasReducer from './canvasReducer';
import statusReducer from './statusSlice';

export const store = configureStore({
  reducer: {
    canvas: canvasReducer,
    status: statusReducer,
  },
});

// RootState와 AppDispatch 타입을 스토어에서 직접 추론
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
