import React, { useRef, useEffect } from 'react';

function Camera() {
  const videoRef = useRef(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => {
        console.error("Error accessing camera: ", err);
      });
  }, []);

  return (
    <div>
      <h2>Camera Feed</h2>
      <video 
        ref={videoRef} 
        autoPlay 
        style={{ width: '640px', height: '480px', border: '1px solid black' }}
      ></video>
    </div>
  );
}

export default Camera;
