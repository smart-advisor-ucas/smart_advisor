import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://127.0.0.1:8000';

const SUGGESTIONS = [
  'ايش شروط المنح الدراسية؟',
  'ما هي مساقات السنة الأولى؟',
  'من هم أعضاء هيئة التدريس؟',
  'ايش فرص العمل بعد التخرج؟',
];

function getTime() {
  return new Date().toLocaleTimeString('ar', { hour: '2-digit', minute: '2-digit' });
}

function getSessionId() {
  let id = localStorage.getItem('advisor_session_id');
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem('advisor_session_id', id);
  }
  return id;
}

function cleanMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/#{1,6} /g, '')
    .replace(/`(.*?)`/g, '$1')
    .replace(/^[-] /gm, '')
    .replace(/(\d+\. )/g, '\n$1')
    .replace(/\| /g, '\n')
    .trim();
}

const sessionId = getSessionId();

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput]         = useState('');
  const [loading, setLoading]     = useState(false);
  const [recording, setRecording] = useState(false);
  const [error, setError]         = useState('');
  const [darkMode, setDarkMode]   = useState(false);
  const bottomRef                 = useRef(null);
  const inputRef                  = useRef(null);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const res = await axios.get(`${API_URL}/chat/history/${sessionId}`);
        if (res.data.history && res.data.history.length > 0) {
          const mapped = res.data.history.map(m => ({
            role: m.role === 'assistant' ? 'bot' : 'user',
            text: m.content,
            time: getTime()
          }));
          setMessages(mapped);
        } else {
          startNew();
        }
      } catch {
        startNew();
      }
    };
    loadHistory();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const startNew = async () => {
    try {
      const res = await axios.post(`${API_URL}/chat/reset`, {
        session_id: sessionId,
        message: ''
      });
      setMessages([{
        role: 'bot',
        text: res.data.reply,
        time: getTime()
      }]);
    } catch {
      setError('تعذّر الاتصال بالخادم. تأكدي من تشغيل الـ Backend.');
    }
  };

  const sendMessage = async (text) => {
    const q = text.trim();
    if (!q || loading) return;
    setError('');
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: q, time: getTime() }]);
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/chat`, {
        session_id: sessionId,
        message: q
      });
      setMessages(prev => [...prev, {
        role: 'bot',
        text: res.data.reply,
        time: getTime()
      }]);
    } catch {
      setError('تعذّر الاتصال بالخادم. تأكدي من تشغيل الـ Backend.');
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const toggleRecording = () => {
    setRecording(prev => !prev);
    // TODO: سجى — ربط Web Speech API هنا
  };

  return (
    <div className={`app ${darkMode ? 'dark' : ''}`}>

      <header className="header">
        <div className="header-top">
          <img
            src="https://upload.wikimedia.org/wikipedia/commons/5/58/%D8%B4%D8%B9%D8%A7%D8%B1_%D8%A7%D9%84%D9%83%D9%84%D9%8A%D8%A9_%D8%A7%D9%84%D8%AC%D8%A7%D9%85%D8%B9%D9%8A%D8%A9_.jpg"
            alt="UCAS"
            className="header-logo-img"
          />
          <h1 className="header-title">Data Science and Artificial Intelligence Smart Advisor</h1>
        </div>
        <p className="header-subtitle">المستشار الأكاديمي الذكي</p>
        <div className="header-actions">
          <button className="theme-toggle" onClick={() => setDarkMode(prev => !prev)}>
            {darkMode ? '☀️ فاتح' : '🌙 داكن'}
          </button>
          <button className="new-chat-btn" onClick={startNew}>
            ✦ محادثة جديدة
          </button>
        </div>
      </header>

      <main className="messages-area">

        {messages.length === 1 && (
          <div className="suggestions">
            {SUGGESTIONS.map((s, i) => (
              <button key={i} className="suggestion-chip" onClick={() => sendMessage(s)}>
                {s}
              </button>
            ))}
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message-row ${msg.role}`}>
            <div className={`avatar ${msg.role}`}>
              {msg.role === 'bot' ? '🤖' : 'أنا'}
            </div>
            <div>
              <div className={`bubble ${msg.role}`}>
                {cleanMarkdown(msg.text).split('\n').map((line, j) => (
                  <p key={j} style={{ margin: j === 0 ? 0 : '4px 0 0' }}>{line}</p>
                ))}
              </div>
              <div className="msg-time">{msg.time}</div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="message-row bot">
            <div className="avatar bot">🤖</div>
            <div className="bubble bot">
              <div className="typing-indicator">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <footer className="input-area">
        {error && <div className="error-bubble">{error}</div>}

        {recording && (
          <div className="voice-bar">
            <span>🎙️</span>
            <span>جاري الاستماع...</span>
            <div className="wave">
              <span /><span /><span /><span /><span />
            </div>
            <button className="voice-stop" onClick={toggleRecording}>إيقاف</button>
          </div>
        )}

        <div className="input-row">
          <button
            className={`mic-btn ${recording ? 'recording' : ''}`}
            onClick={toggleRecording}
            title="تسجيل صوتي"
          >
            🎙️
          </button>
          <input
            ref={inputRef}
            className="text-input"
            placeholder="اكتب سؤالك هنا..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || loading}
          >
            ➤
          </button>
        </div>

        <div className="input-hint">
          اضغط Enter للإرسال · المعلومات مستخرجة من وثائق UCAS الرسمية
        </div>
      </footer>

    </div>
  );
}