import React from 'react';
import Camera from './components/Camera';
import SentenceDisplay from './components/SentenceDisplay';

function App() {
  return (
    <div style={{ textAlign: 'center', padding: '20px' }}>
      <h1>Hand Gesture Translation App</h1>
      <SentenceDisplay />
      <Camera />
    </div>
  );
}

export default App;
