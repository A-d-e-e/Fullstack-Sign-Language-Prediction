import React, { useState, useEffect } from 'react';

function SentenceDisplay() {
  const [sentence, setSentence] = useState("");

  const fetchSentence = async () => {
    try {
      const response = await fetch('/sentence');
      const data = await response.json();
      setSentence(data.sentence);
    } catch (error) {
      console.error("Error fetching sentence:", error);
    }
  };

  useEffect(() => {
    const intervalId = setInterval(fetchSentence, 1000); 
    return () => clearInterval(intervalId);
  }, []);

  const handleReset = async () => {
    try {
      const response = await fetch('/reset');
      const data = await response.json();
      setSentence(data.sentence);
    } catch (error) {
      console.error("Error resetting sentence:", error);
    }
  };

  return (
    <div style={{ marginBottom: '20px' }}>
      <h2>Current Sentence:</h2>
      <p style={{ 
          fontSize: '24px', 
          border: '1px solid #ccc', 
          padding: '10px', 
          minHeight: '50px',
          width: '80%',
          margin: 'auto'
        }}>
        {sentence}
      </p>
      <button onClick={handleReset} style={{ padding: '10px 20px', fontSize: '16px' }}>
        Clear Sentence
      </button>
    </div>
  );
}

export default SentenceDisplay;
