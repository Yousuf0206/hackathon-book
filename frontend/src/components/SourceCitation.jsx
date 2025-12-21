import React from 'react';

const SourceCitation = ({ citation }) => {
  const { source_url, chapter, section, text_snippet } = citation;

  return (
    <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1" role="listitem">
      {chapter && (
        <div className="font-medium" role="definition">
          <span className="font-semibold">Chapter:</span> {chapter}
        </div>
      )}
      {section && (
        <div role="definition">
          <span className="font-semibold">Section:</span> {section}
        </div>
      )}
      {text_snippet && (
        <div className="italic mt-1" role="note">
          "{text_snippet}"
        </div>
      )}
      {source_url && (
        <div role="link">
          <a
            href={source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 dark:text-blue-400 hover:underline text-xs"
            aria-label={`Source link for ${chapter || 'this citation'}`}
          >
            Source Link →
          </a>
        </div>
      )}
    </div>
  );
};

export default SourceCitation;