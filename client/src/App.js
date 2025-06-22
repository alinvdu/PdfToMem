import './App.css';
import { useState, useEffect } from 'react';
import { useMcp } from 'use-mcp/react';
import ReactJson from 'react-json-view';
import { IoFileTrayFullOutline } from "react-icons/io5";
import { IoIosSend } from "react-icons/io";

function App() {
  const { state, tools, error, callTool, retry, authenticate } = useMcp({
    url: 'http://localhost:8000/mcp',
    clientName: 'pdf_ingest',
    autoReconnect: true,
  });

  // Modal visibility
  const [ingestionModalOpen, setIngestionModalOpen] = useState(false);
  const [storageModalOpen, setStorageModalOpen] = useState(false);
  const [chunksModalOpen, setChunksModalOpen] = useState(false);

  // Tool & ingestion
  const [selectedToolName, setSelectedToolName] = useState('');
  const [selectedTool, setSelectedTool] = useState(null);
  const [inputValues, setInputValues] = useState({});
  const [ingestionData, setIngestionData] = useState(null);
  const [dataView, setDataView] = useState('Memory Representation');
  const [isToolLoading, setIsToolLoading] = useState(false);
  const [isQueryLoading, setIsQueryLoading] = useState(false);

  // Storage
  const [isStorageLoading, setIsStorageLoading] = useState(false);
  const [storageDataView, setStorageDataView] = useState('Response');
  const [storageResponse, setStorageResponse] = useState(null);

  // Query
  const [readyToQuery, setReadyToQuery] = useState(false);
  const [queryText, setQueryText] = useState('');
  const [queryResponse, setQueryResponse] = useState(null);
  const [queryDataView, setQueryDataView] = useState('Response');

  const [fileNames, setFileNames] = useState({});

  // Reset when tools change
  useEffect(() => {
    setSelectedToolName('');
    setSelectedTool(null);
    setInputValues({});
    setIngestionData(null);
    setDataView('Memory Representation');
    setIsToolLoading(false);
    setReadyToQuery(false);
    setQueryText('');
    setQueryResponse(null);
    setQueryDataView('Response');
    setStorageDataView('Response');
  }, [tools]);

  if (state === 'failed') {
    return (
      <div className="full-screen-message">
        <p>Connection failed: {error}</p>
        <button className="btn" onClick={retry}>Retry</button>
        <button className="btn" onClick={authenticate}>Authenticate</button>
      </div>
    );
  }
  if (state !== 'ready') {
    return <div className="full-screen-message">Connecting to AI service...</div>;
  }

  const handleInputChange = (name, value) => setInputValues(prev => ({ ...prev, [name]: value }));
  const handleFileChange = async (name, file) => {
    setFileNames(prev => ({ ...prev, [name]: file.name }));
    const base64 = await new Promise((res, rej) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => res(reader.result.split(',')[1]);
      reader.onerror = () => rej();
    });
    handleInputChange(name, base64);
  };

  const startTool = async () => {
    if (!selectedTool) return;
    setIsToolLoading(true);
    try {
      const result = await callTool(selectedTool.name, inputValues);
      if (!result.isError && selectedTool.name === 'determine_pdf_ingestion_architecture_and_memory_representation') {
        const parsed = JSON.parse(result.content[0].text);
        parsed.ingestion_plan = JSON.parse(parsed.ingestion_plan);
        setIngestionData(parsed);
      }
    } catch (err) {
      console.error(err);
    }
    setIsToolLoading(false);
  };

  const startStorage = async () => {
    if (!ingestionData) return;
    setIsStorageLoading(true);
    console.log('calling with ingestion data', JSON.stringify(ingestionData))
    try {
      const res = await fetch('http://localhost:1000/storage_ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ingestionData),
      });
      const data = await res.json();
      setStorageResponse(data);
      setReadyToQuery(true);
    } catch (err) {
      console.error(err);
    }
    setIsStorageLoading(false);
    setIngestionModalOpen(false);
  };

  const handleQuerySubmit = async () => {
    setIsQueryLoading(true);
    try {
      const res = await fetch('http://localhost:1000/query_router', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText }),
      });
      const data = await res.json();
      setQueryResponse(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsQueryLoading(false);
    }
  };

  // Modal to display chunks
  const ChunksModal = () => (
    <div className="modal-overlay">
      <div className="modal">
        <button className="close-btn" onClick={() => setChunksModalOpen(false)}>&times;</button>
        <div style={{ fontSize: 21 }}>Chunks</div>
        <div style={{ maxHeight: '60vh', overflowY: 'auto', marginTop: 10 }}>
          {queryResponse.chunks.map((chunk, i) => (
            <div key={i} style={{ marginBottom: 10, background: "rgba(25, 25, 25, 0.25)", padding: 12, boxSizing: "border-box" }}>
              <pre style={{ whiteSpace: 'pre-wrap' }}>{chunk}</pre>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderParameterInput = (name, schema) => {
    if (schema.format === 'binary' || name.toLowerCase().includes('pdf')) {
      return (
        <div key={name} style={{ marginBottom: 10 }}>
          <div style={{marginBottom: 8}}>{schema.title || name}</div>
          <div style={{ display: "flex", alignItems: "center" }}>
            <input style={{ display: "none" }} type="file" id="file" onChange={e => handleFileChange(name, e.target.files[0])} />
            <label style={{ background: 'rgba(255, 255, 255, 0.25)', fontSize: 13, padding: 6, borderRadius: 3, boxSizing: "border-box" }} htmlFor="file">Select file</label>
            {fileNames[name] && (
              <div style={{ marginTop: 4, marginLeft: 5, fontSize: 12, color: 'white' }}>
                <strong>{fileNames[name]}</strong>
              </div>
            )}
          </div>
        </div>
      );
    }
    if (schema.type === 'boolean') {
      return (
        <div key={name} style={{ marginBottom: 10 }}>
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
    return (
      <div key={name}>
        <label>{schema.title || name}</label>
        <input
          type="text"
          value={inputValues[name] || ''}
          onChange={e => handleInputChange(name, e.target.value)}
        />
      </div>
    );
  };

  const renderDataView = () => {
    if (!ingestionData) return null;
    switch (dataView) {
      case 'Memory Representation':
        return (
          <div>
            {ingestionData.ingestion_plan.map((plan, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <strong>Strategy:</strong> {plan.strategy}<br />
                <strong>Description:</strong> {plan.query_node_description}
              </div>
            ))}
          </div>
        );
      case 'Tool Usage':
        return <div>{ingestionData.structure_envelope.steps_completed.join(', ')}</div>;
      case 'Extracted Text': {
        const sections = (ingestionData.structure_envelope.segment_output.sections || []).join('\n\n');
        const display = sections.length > 2000 ? sections.slice(0,2000)+'…' : sections;
        return <pre style={{ whiteSpace:'pre-wrap' }}>{display}</pre>;
      }
      case 'Extracted Tables':
        return <ReactJson src={ingestionData.structure_envelope.extractor.tables} name={false} collapsed={1} />;
      case 'Summaries':
        return (ingestionData.structure_envelope.semantic_output.summary || []).map((s,i)=>
          <p key={i}>{s}</p>
        );
      case 'Entities':
        return <ul>{(ingestionData.structure_envelope.semantic_output.entities||[]).map((e,i)=><li key={i}>{e}</li>)}</ul>;
      default:
        return null;
    }
  };

  const IngestionModal = () => {
    const renderContent = () => {
      if (ingestionData) {
        return (
          <>
            <div className="tabs">
              {['Memory Representation','Tool Usage','Extracted Text','Extracted Tables','Summaries','Entities'].map(key => (
                <button
                  key={key}
                  className={`tab-button ${dataView===key? 'active':''}`}
                  onClick={() => setDataView(key)}
                >{key}</button>
              ))}
            </div>
            <div className="tab-content">{renderDataView()}</div>
            {storageResponse && storageResponse.tools && storageResponse.tools.length && (
              <div>
                <div style={{ marginTop: 15 }}>Storage Configuration</div>
                {storageResponse.tools.map(tool => (
                  <div key={tool.name} style={{
                    marginTop: 8,
                    background: "#4e4e4e",
                    padding: 10,
                    boxSizing: "border-box"
                  }}>
                    <div>Configured Query Tool: {tool.name}</div>
                    <div>Description: {tool.description}</div>
                  </div>
                ))}
              </div>
            )}
            <div style={{ display: "flex" }}>
              {!storageResponse && (
                <button className="btn" style={{ alignSelf: "flex-start", marginTop: 20 }} onClick={startStorage} disabled={isStorageLoading}>
                  {isStorageLoading? <div className="spinner"/> : 'Create Storage'}
                </button>
              )}
              <button className="btn" style={{ alignSelf: "flex-start", marginTop: 20 }} onClick={() => {
                setIngestionData(null)
                setStorageResponse(null)
              }} disabled={isStorageLoading}>
                Clear Ingestion
              </button>
            </div>
          </>
        );
      }
      return (
        <>
          <div className="ingestion-inner">
            <select
              value={selectedToolName}
              className="tool-selector"
              onChange={e => {
                const name = e.target.value;
                setSelectedToolName(name);
                const t = tools.find(tt => tt.name === name) || null;
                setSelectedTool(t);
                setInputValues({});
                setIngestionData(null);
              }}
            >
              <option value="" disabled>Select tool...</option>
              {tools.map(t => <option key={t.name} value={t.name}>{t.name}</option>)}
            </select>
            {selectedTool && (
              <div style={{ paddingBottom: 20 }}>
                <div style={{ fontSize: 17, marginTop: 18, fontWeight: "bold" }}>Tool Description</div>
                <pre style={{ width: "100%", whiteSpace: "pre-wrap" }}>{selectedTool.description}</pre>
                <div style={{ fontSize: 17, marginTop: 18, fontWeight: "bold", marginBottom: 10 }}>Complete Inputs</div>
                {selectedTool.inputSchema
                  ? Object.entries(selectedTool.inputSchema.properties).map(([n,s]) => renderParameterInput(n,s))
                  : <p>No parameters.</p>
                }
                <button className="btn" onClick={startTool} disabled={isToolLoading}>
                  {isToolLoading ? <div className="spinner"/> : `Run ${selectedTool.name}`}
                </button>
              </div>
            )}
          </div>
        </>
      );
    };

    return (
      <div className="modal-overlay">
        <div className="modal">
          <button className="close-btn" onClick={() => setIngestionModalOpen(false)}>&times;</button>
          <div style={{ fontSize: 21 }}>Memory Viewer</div>
          <h4>Select tool to compose memory</h4>
          {renderContent()}
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      <div className="search-container">
        <h1 style={{ width: "100%", textAlign: "center", marginBottom: 55 }}>Welcome to PDF To Mem!</h1>
        <div className={'search-box-wrapper'}>
          <textarea
            className="search-textarea"
            placeholder={readyToQuery ? 'Memory ready, type query...' : 'Please handle Memory before querying...'}
            value={queryText}
            onChange={e => setQueryText(e.target.value)}
            disabled={!readyToQuery}
          />
          <div className="button-row">
            <div style={{ display: "flex" }}>
              <button className="btn" onClick={() => setIngestionModalOpen(true)}><IoFileTrayFullOutline style={{ fontSize: 18, marginRight: 8 }} /><span>Memory</span></button>
              {isStorageLoading && <div className="spinner" />}
            </div>
            <button className="btn-query" style={{ display: "flex", opacity: (!readyToQuery || !queryText || isQueryLoading) ? 0.35 : 1 }} onClick={handleQuerySubmit} disabled={(!readyToQuery || !queryText || isQueryLoading)}>
              {isQueryLoading
                ? <div className="spinner" />
                : <IoIosSend style={{ color: "white", marginRight: 5, fontSize: 17 }} />
              }
            </button>
          </div>
        </div>
        {/* Query Response Display */}
        {queryResponse && <div style={{ marginTop: 38, fontSize: 16, fontWeight: 'bold', color: 'white' }}>Response</div>}
          {queryResponse && (
            <div style={{ marginTop: 12, padding: '15px', display: "flex", flexDirection: "column", background: "rgba(25, 25, 25, 0.55)", borderRadius: 21, boxSizing: "border-box", width: 800 }}>
              <div style={{ color: 'white', marginTop: 5, justifyContent: "space-between", fontSize: 15 }}>{queryResponse.response}</div>
              <div style={{display: "flex", justifyContent: "space-between", alignItems: "center"}}>
                <div style={{ marginTop: 10, fontSize: 14, color: 'white', alignItems: "center" }}>
                  <strong>Used Memory Sources:</strong> {queryResponse.selected_queries.join(', ')}
                </div>
                <button className="btn" style={{ marginTop: 10, float: 'right', alignSelf: "flex-end" }} onClick={() => setChunksModalOpen(true)}>
                  Chunks
                </button>
              </div>
            </div>
          )}
      </div>
      {ingestionModalOpen && <IngestionModal />}
      {chunksModalOpen && <ChunksModal />}
    </div>
  );
}

export default App;
