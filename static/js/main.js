// static/js/main.js

// Import React and ReactDOM
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App'; // Adjust the path if your App component is elsewhere

// Render the React app into the #root div
ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);