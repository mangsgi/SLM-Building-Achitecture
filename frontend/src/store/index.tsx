import { configureStore } from '@reduxjs/toolkit';
import canvasReducer from './canvasReducer';
import statusReducer from './statusSlice';
import sseMiddleware from './sseMiddleware';

export const store: any = configureStore({
  reducer: {
    canvas: canvasReducer,
    status: statusReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(sseMiddleware),
});

// RootState와 AppDispatch 타입을 스토어에서 직접 추론
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
