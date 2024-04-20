import React from 'react';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/homePage/homePage';
import LiveVideoPage from './pages/liveVideoPage/liveVideoPage';

function App() {
  return (
    <div className='App'>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/live-video" element={<LiveVideoPage />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
