import { useRef, useState } from 'react';

const ACCEPTED = '.pdf,.docx,.txt';

export default function UploadPanel({ onSubmit, isProcessing, error }) {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  function handleFiles(fileList) {
    const picked = fileList?.[0];
    if (picked) setFile(picked);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div>
      <div
        className={`dropzone ${dragActive ? 'dropzone--active' : ''}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <span className="dropzone__label">
          Drop a PDF, Word doc, or text file here, or click to browse.
        </span>
        {file && <div className="dropzone__filename">{file.name}</div>}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      <button
        className="btn"
        disabled={!file || isProcessing}
        onClick={() => onSubmit(file)}
      >
        {isProcessing ? 'Reading your notes…' : 'Analyze notes'}
      </button>

      {error && <div className="error-text">{error}</div>}
    </div>
  );
}