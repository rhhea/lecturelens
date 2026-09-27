export default function ResourcesPanel({ resources }) {
  if (!resources || resources.length === 0) {
    return <p className="abstract abstract--empty">Recommended tutorials and quizzes will appear here once notes are analyzed.</p>;
  }
  return (
    <ul className="resources">
      {resources.map((r, i) => (
        <li className="resource" key={i}>
          <div className="resource__title">
            <a href={r.url} target="_blank" rel="noreferrer">
              {r.title}
            </a>
          </div>
          <div className="resource__meta">{r.type}</div>
        </li>
      ))}
    </ul>
  );
}