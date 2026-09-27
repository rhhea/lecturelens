export default function QuestionsPanel({ questions }) {
  if (!questions || questions.length === 0) {
    return <p className="abstract abstract--empty">Follow-up questions will appear here once notes are analyzed.</p>;
  }
  return (
    <ol className="questions">
      {questions.map((q, i) => (
        <li key={i}>{q}</li>
      ))}
    </ol>
  );
}