import * as actionTypes from './actions';

export const initialState = {
  isDirty: false,
};

// canvas의 상태를 관리하는 reducer
const canvasReducer = (state = initialState, action: any) => {
  switch (action.type) {
    case actionTypes.SET_DIRTY:
      return {
        ...state,
        isDirty: true,
      };
    default:
      return state;
  }
};

export default canvasReducer;
