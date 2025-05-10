import { createStore } from 'redux';
import reducer from './reducer';

// store 생성
const store = createStore(reducer);
const persister = 'Free';

export { store, persister };
