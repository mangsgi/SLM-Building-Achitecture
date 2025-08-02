import { combineReducers } from 'redux';

import canvasReducer from './canvasReducer';

// 모든 reducer를 하나로 합치는 reducer
const reducer = combineReducers({
  canvas: canvasReducer,
});

export default reducer;
