export default function SummaryPanel({ summary }) {
  if (!summary) {
    return <p className="abstract abstract--empty">Your summary will appear here once notes are analyzed.</p>;
  }
  return <p className="abstract">{summary}</p>;
}