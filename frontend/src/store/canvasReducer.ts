import * as actionTypes from './actions';

export const initialState = {
  isDirty: false,
};

// canvasReducer
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
