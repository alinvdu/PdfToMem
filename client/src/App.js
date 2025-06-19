import './App.css';
import { useState } from 'react';
import { useMcp } from 'use-mcp/react';

function App() {
  const {
    state,
    tools,
    error,
    callTool,
    retry,
    authenticate,
  } = useMcp({
    url: 'http://localhost:8000/mcp',
    clientName: 'pdf_ingest',
    autoReconnect: true,
  });

  const [selectedTool, setSelectedTool] = useState(null);
  const [inputValues, setInputValues] = useState({});
  const [callResult, setCallResult] = useState(null);

  if (state === 'failed') {
    return (
      <div>
        <p>Connection failed: {error}</p>
        <button onClick={retry}>Retry</button>
        <button onClick={authenticate}>Authenticate Manually</button>
      </div>
    );
  }

  if (state !== 'ready') {
    return <div>Connecting to AI service...</div>;
  }

  const handleInputChange = (name, value) => {
    setInputValues(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = async (name, file) => {
    const base64 = await convertFileToBase64(file);
    handleInputChange(name, base64);
  };

  const convertFileToBase64 = file =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64 = reader.result.split(',')[1]; // Strip "data:...base64,"
        resolve(base64);
      };
      reader.onerror = reject;
    });

  const handleCallTool = async () => {
    try {
      const result = await callTool(selectedTool.name, inputValues);
      setCallResult(result);
    } catch (err) {
      setCallResult(`Error: ${err.message}`);
    }
  };

  const renderParameterInput = (name, schema) => {
    if (schema.format === 'binary') {
      return (
        <div key={name}>
          <label>{schema.title || name}:&nbsp;</label>
          <input type="file" onChange={e => handleFileChange(name, e.target.files[0])} />
        </div>
      );
    }

    if (schema.type === 'boolean') {
      return (
        <div key={name}>
          <label>
            <input
              type="checkbox"
              checked={!!inputValues[name]}
              onChange={e => handleInputChange(name, e.target.checked)}
            />
            {schema.title || name}
          </label>
        </div>
      );
    }

    // Fallback for string or number
    return (
      <div key={name}>
        <label>
          {schema.title || name}:&nbsp;
          <input
            type="text"
            value={inputValues[name] || ''}
            onChange={e => handleInputChange(name, e.target.value)}
          />
        </label>
      </div>
    );
  };

  return (
    <div>
      <h2>Available Tools: {tools.length}</h2>
      <ul>
        {tools.map(tool => (
          <li key={tool.name}>
            <button onClick={() => {
              setSelectedTool(tool);
              setInputValues({});
              setCallResult(null);
            }}>
              {tool.name}
            </button>
          </li>
        ))}
      </ul>

      {selectedTool && (
        <div style={{ marginTop: '1rem' }}>
          <h3>{selectedTool.name}</h3>
          <p>{selectedTool.description || 'No description provided.'}</p>

          <h4>Parameters:</h4>
          {selectedTool.inputSchema ? (
            Object.entries(selectedTool.inputSchema.properties).map(
              ([name, schema]) => renderParameterInput(name, schema)
            )
          ) : (
            <p>No parameters defined</p>
          )}

          <button onClick={handleCallTool} style={{ marginTop: '1rem' }}>
            Call Tool
          </button>

          {callResult && (
            <div style={{ marginTop: '1rem' }}>
              <h4>Result:</h4>
              <pre>{JSON.stringify(callResult, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
