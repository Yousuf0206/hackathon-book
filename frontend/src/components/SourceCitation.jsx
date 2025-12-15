import React from 'react';

const SourceCitation = ({ citation }) => {
  const { source_url, chapter, section, text_snippet } = citation;

  return (
    <div className="source-citation" role="listitem">
      {chapter && (
        <div className="citation-chapter" role="definition">
          <strong>Chapter:</strong> {chapter}
        </div>
      )}
      {section && (
        <div className="citation-section" role="definition">
          <strong>Section:</strong> {section}
        </div>
      )}
      {source_url && (
        <div className="citation-source" role="link">
          <a href={source_url} target="_blank" rel="noopener noreferrer" aria-label={`Source link for ${chapter || 'this citation'}`}>
            Source Link
          </a>
        </div>
      )}
      {text_snippet && (
        <div className="citation-text" role="note">
          <small><strong>Excerpt:</strong> "{text_snippet}"</small>
        </div>
      )}
    </div>
  );
};

export default SourceCitation;