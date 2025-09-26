import { Middleware } from '@reduxjs/toolkit';
import {
  startTraining,
  completeTraining,
  failTraining,
  resetStatus,
} from './statusSlice';
import { RootState } from './index';

// EventSource 연결을 미들웨어 외부에서 관리하여 참조를 유지합니다.
let eventSource: EventSource | null = null;

const sseMiddleware: Middleware<object, RootState> =
  (store) => (next) => (action) => {
    // 다음 미들웨어/리듀서로 액션을 전달
    const result = next(action);

    // 'startTraining' 액션이 디스패치되었는지 확인
    if (startTraining.match(action)) {
      const { task_id } = action.payload;
      if (!task_id) return result;

      // 기존 연결이 있다면 중복 생성을 방지하기 위해 닫습니다.
      if (eventSource) {
        eventSource.close();
      }

      // SSE 연결 URL 생성
      const sseUrl = `http://localhost:8000/api/v1/events/${task_id}`;
      // console.log(`[SSE] Connecting to ${sseUrl}`);

      // EventSource 인스턴스 생성
      eventSource = new EventSource(sseUrl);

      // 메시지 수신 리스너
      eventSource.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          // console.log('[SSE] Received data:', parsedData);

          // 백엔드가 보낸 이벤트 타입 확인
          if (parsedData.event === 'finished') {
            // console.log(
            //   '[SSE] Training complete event received. Dispatching completeTraining.',
            // );
            // 학습 완료 액션 디스패치
            store.dispatch(completeTraining());
            // 연결 종료
            eventSource?.close();
            eventSource = null;
          } else if (parsedData.event === 'error') {
            // console.error(
            //   '[SSE] Training error event received:',
            //   parsedData.data?.message,
            // );
            // 학습 실패 액션 디스패치
            store.dispatch(
              failTraining({
                message: parsedData.data?.message || 'Unknown error',
              }),
            );
            // 연결 종료
            eventSource?.close();
            eventSource = null;
          }
        } catch (error) {
          console.error('[SSE] Error parsing event data:', error);
        }
      };

      // 에러 리스너
      eventSource.onerror = (error) => {
        console.error('[SSE] EventSource failed:', error);
        store.dispatch(failTraining({ message: 'SSE connection failed.' }));
        eventSource?.close();
        eventSource = null;
      };
    }

    // 학습 중단 액션 처리
    if (resetStatus.match(action)) {
      if (eventSource) {
        // console.log('[SSE] Resetting status, closing EventSource connection.');
        eventSource.close();
        eventSource = null;
      }
    }

    return result;
  };

export default sseMiddleware;
