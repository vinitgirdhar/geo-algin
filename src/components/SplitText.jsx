import React from 'react';

/**
 * SplitText component to split text into characters, words, or lines for GSAP animation.
 * 
 * @param {Object} props
 * @param {string} props.text - The text to split (supports `<br>` for line breaks).
 * @param {'chars'|'words'|'lines'} props.type - The split type.
 * @param {string} [props.className] - Optional class name for the wrapper.
 */
export default function SplitText({ text, type, className }) {
  if (!text) return null;

  if (type === 'chars') {
    return (
      <span className={className}>
        {text.split('').map((char, index) => (
          <span 
            key={index} 
            className="char" 
            style={{ display: 'inline-block' }}
          >
            {char === ' ' ? '\u00A0' : char}
          </span>
        ))}
      </span>
    );
  }

  if (type === 'words') {
    const parts = text.split(/<br\s*\/?>/i);
    return (
      <span className={className}>
        {parts.map((part, pIdx) => (
          <React.Fragment key={pIdx}>
            {pIdx > 0 && <br />}
            {part.split(' ').map((word, wIdx) => (
              <span 
                key={wIdx} 
                className="word" 
                style={{ display: 'inline-block' }}
              >
                {word}{wIdx < part.split(' ').length - 1 ? '\u00A0' : ''}
              </span>
            ))}
          </React.Fragment>
        ))}
      </span>
    );
  }

  if (type === 'lines') {
    const lines = text.split(/<br\s*\/?>/i);
    return (
      <span className={className} style={{ display: 'block' }}>
        {lines.map((line, index) => (
          <span 
            key={index} 
            className="line-mask" 
            style={{ display: 'block', overflow: 'hidden', position: 'relative' }}
          >
            <span 
              className="line-span" 
              style={{ display: 'inline-block', position: 'relative' }}
            >
              {line}
            </span>
          </span>
        ))}
      </span>
    );
  }

  return <span className={className}>{text}</span>;
}
