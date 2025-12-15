import React, { useState, useEffect } from 'react';
import PersonalizationControls from '../../frontend/src/components/PersonalizationControls';
import UrduTranslationControls from '../../frontend/src/components/UrduTranslationControls';
import '../../frontend/src/components/personalization.css';
import '../../frontend/src/components/translation.css';
import './chapter-controls.css';

const ChapterControls = ({ children }) => {
  const [personalizedContent, setPersonalizedContent] = useState(null);
  const [translatedContent, setTranslatedContent] = useState(null);
  const [finalContent, setFinalContent] = useState(children);
  const [originalContent] = useState(children);
  const [activeTransformations, setActiveTransformations] = useState({
    personalized: false,
    translated: false
  });

  // Update final content when transformations change
  useEffect(() => {
    let contentToDisplay = originalContent;

    // Apply personalization if active
    if (activeTransformations.personalized && personalizedContent) {
      contentToDisplay = personalizedContent;
    }

    // Apply translation if active
    if (activeTransformations.translated && translatedContent) {
      contentToDisplay = translatedContent;
    }

    // If both are active, prioritize translation but mention personalization
    if (activeTransformations.personalized && activeTransformations.translated && translatedContent) {
      contentToDisplay = translatedContent;
    }

    setFinalContent(contentToDisplay);
  }, [personalizedContent, translatedContent, activeTransformations, originalContent]);

  const handlePersonalizationChange = (newContent) => {
    setPersonalizedContent(newContent);
    setActiveTransformations(prev => ({ ...prev, personalized: true }));
  };

  const handleTranslationChange = (newContent) => {
    setTranslatedContent(newContent);
    setActiveTransformations(prev => ({ ...prev, translated: true }));
  };

  const resetContent = () => {
    setPersonalizedContent(null);
    setTranslatedContent(null);
    setFinalContent(originalContent);
    setActiveTransformations({
      personalized: false,
      translated: false
    });
  };

  return (
    <div className="chapter-controls-wrapper">
      <div className="controls-header">
        <h3>Customize Your Learning Experience</h3>
        <button onClick={resetContent} className="reset-button">
          Reset to Original
        </button>
      </div>

      <div className="transformation-status">
        {activeTransformations.personalized && (
          <span className="status-badge personalized">Personalized</span>
        )}
        {activeTransformations.translated && (
          <span className="status-badge translated">Translated to Urdu</span>
        )}
      </div>

      <div className="personalization-section">
        <PersonalizationControls
          content={originalContent}
          onContentChange={handlePersonalizationChange}
        />
      </div>

      <div className="translation-section">
        <UrduTranslationControls
          content={activeTransformations.personalized && personalizedContent ? personalizedContent : originalContent}
          onContentChange={handleTranslationChange}
        />
      </div>

      <div className="chapter-content">
        {typeof finalContent === 'string' ? (
          <div
            className={activeTransformations.translated ? 'urdu-text' : ''}
            dangerouslySetInnerHTML={{ __html: finalContent }}
          />
        ) : (
          finalContent
        )}
      </div>
    </div>
  );
};

export default ChapterControls;