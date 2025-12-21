import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatWidget from './ChatWidget';

// Mock fetch API
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () =>
      Promise.resolve({
        response: 'Test response from AI',
        citations: [
          {
            chunk_id: 'test-id',
            source_url: 'http://example.com',
            chapter: 'Test Chapter',
            section: 'Test Section',
            relevance_score: 0.9,
            text_snippet: 'This is a test snippet...'
          }
        ],
        session_id: 'test-session-id'
      })
  })
);

describe('ChatWidget Component', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  test('renders chat widget with welcome message', () => {
    render(<ChatWidget />);

    expect(screen.getByText('Book Assistant')).toBeInTheDocument();
    expect(screen.getByText(/Hello! I'm your book assistant/i)).toBeInTheDocument();
  });

  test('allows user to type and send a message', async () => {
    render(<ChatWidget />);

    const input = screen.getByPlaceholderText(/Ask a question about the book/i);
    const sendButton = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Test question' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledTimes(1);
    });
  });

  test('handles text selection functionality', () => {
    render(<ChatWidget />);

    // Mock window.getSelection
    Object.defineProperty(window, 'getSelection', {
      value: () => ({
        toString: () => 'Selected test text'
      }),
      writable: true
    });

    const selectionButton = screen.getByText('Use Selected Text');
    fireEvent.click(selectionButton);

    expect(screen.getByText(/Selected: Selected test text/i)).toBeInTheDocument();
  });

  test('displays AI response with citations', async () => {
    render(<ChatWidget />);

    const input = screen.getByPlaceholderText(/Ask a question about the book/i);
    const sendButton = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Test question' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(screen.getByText('Test response from AI')).toBeInTheDocument();
    });

    expect(screen.getByText('Sources:')).toBeInTheDocument();
    expect(screen.getByText('Chapter: Test Chapter')).toBeInTheDocument();
  });
});