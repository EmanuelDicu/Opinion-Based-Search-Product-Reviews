import React, { useState, useRef, useEffect, FormEvent } from 'react';
import './App.scss';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || '/api';

type MessageType = 'user' | 'assistant';

interface SearchResult {
  product_id: string;
  product_name: string;
  model?: string;  // Phone model name (same as product_name) for display
  aspect: string;
  opinion: string;
  sentiment: 'positive' | 'negative' | 'neutral' | string;
  sentiment_display?: 'positive' | 'mixed' | 'negative';  // Display label (neutral shown as mixed)
  review_text: string;
}

interface ChatMessage {
  type: MessageType;
  content: string;
  results?: SearchResult[];
  timestamp: Date;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessageText = input.trim();
    setInput('');
    setLoading(true);

    const newUserMessage: ChatMessage = {
      type: 'user',
      content: userMessageText,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newUserMessage]);

    try {
      const response = await axios.post(`${API_URL}/search`, {
        query: userMessageText,
      });

      const assistantMessage: ChatMessage = {
        type: 'assistant',
        content: response.data.summary,
        results: response.data.results as SearchResult[],
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: unknown) {
      console.error('Error:', error);
      const err = error && typeof error === 'object' && 'response' in error
        ? (error as { response?: { status?: number; data?: { error?: string } } }).response
        : null;
      const message = err?.status === 503 && typeof err?.data?.error === 'string'
        ? err.data.error
        : 'Sorry, I encountered an error processing your query. Please try again.';
      const errorMessage: ChatMessage = {
        type: 'assistant',
        content: message,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        <header className="chat-header">
          <h1>Opinion-Based Search</h1>
          <p>Smartphone features and reviews</p>
        </header>

        <main className="chat-messages">
          {messages.map((message, index) => (
            <div key={index} className={`message-row message-row--${message.type}`}>
              <div className={`message-bubble message-bubble--${message.type}`}>
                <div className="message-text">{message.content}</div>
                {message.results && message.results.length > 0 && (
                  <div className="results-container">
                    <h4 className="results-title">Related Reviews ({message.results.length})</h4>
                    <div className="results-grid">
                      {message.results.slice(0, 20).map((result, idx) => {
                        const model = result.model ?? result.product_name;
                        const sentiment = result.sentiment_display ?? (result.sentiment === 'neutral' ? 'mixed' : result.sentiment);
                        return (
                          <div key={idx} className="result-item">
                            <div className="result-product-sentiment">
                              <span className="product-name">{model}</span>
                              <span className={`sentiment-badge ${sentiment}`}>
                                {sentiment}
                              </span>
                            </div>
                            <div className="result-review">
                              {result.review_text}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message-row message-row--assistant">
              <div className="message-bubble message-bubble--assistant message-bubble--loading">
                <div className="loading-dots">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </main>

        <footer className="chat-input-area">
          <form className="chat-input-form" onSubmit={handleSend}>
            <input
              type="text"
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder='Ask about phones, e.g. "What phone is good for gaming?"'
              disabled={loading}
            />
            <button type="submit" className="send-button" disabled={loading || !input.trim()}>
              Send
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
};

export default App;