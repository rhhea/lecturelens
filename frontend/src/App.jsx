import { useState } from 'react';
import UploadPanel from './components/UploadPanel';
import SummaryPanel from './components/SummaryPanel';
import QuestionsPanel from './components/QuestionsPanel';
import ResourcesPanel from './components/ResourcesPanel';
import { processNotes } from './api';

export default function App() {
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(file) {
    setIsProcessing(true);
    setError(null);
    try {
      const data = await processNotes(file);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong while processing that file.');
    } finally {
      setIsProcessing(false);
    }
  }

  const sectionClass = `section ${result ? 'section--reveal' : ''}`;

  return (
    <div className="page">
      <header className="masthead">
        <h1 className="masthead__wordmark">LectureLens</h1>
        <p className="masthead__tagline">A close reading of your notes — summarized, questioned, and cross-referenced.</p>
        <hr className="masthead__rule" />
      </header>

      <main className="sections">
        <section className={sectionClass}>
          <span className="section-number">1</span>
          <h2 className="section-title">Upload your notes</h2>
          <p className="section-hint">PDF, Word, or plain text — lecture notes work best.</p>
          <UploadPanel onSubmit={handleSubmit} isProcessing={isProcessing} error={error} />
        </section>

        <section className={sectionClass}>
          <span className="section-number">2</span>
          <h2 className="section-title">Abstract</h2>
          <SummaryPanel summary={result?.summary} />
        </section>

        <section className={sectionClass}>
          <span className="section-number">3</span>
          <h2 className="section-title">Study questions</h2>
          <QuestionsPanel questions={result?.followup_questions} />
        </section>

        <section className={sectionClass}>
          <span className="section-number">4</span>
          <h2 className="section-title">Further reading</h2>
          <ResourcesPanel resources={result?.recommended_resources} />
        </section>
      </main>
    </div>
  );
}