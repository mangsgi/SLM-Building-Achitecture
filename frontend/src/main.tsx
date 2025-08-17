import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import App from './App';
import DatasetSelection from './DatasetSelection';
import TestPage from './TestPage.tsx';
import './index.css';
import { Provider } from 'react-redux';
import { store } from './store/index.tsx';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Provider store={store}>
      <Router>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/canvas" element={<App />} />
          <Route path="/canvas/dataset" element={<DatasetSelection />} />
          <Route path="/test" element={<TestPage />} />
        </Routes>
      </Router>
    </Provider>
  </React.StrictMode>,
);
