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

function formatDuration(sec) {
  if (!isFinite(sec) || sec < 0) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
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

const REACTIONS = ['👍', '❤️', '😂', '😮', '😢', '🙏'];

// ── فقاعة صوتية بأسلوب واتساب: زر تشغيل دائري + خط تقدم قابل للسحب + مدة + قائمة (تنزيل/تفاعل) ──
function VoiceBubble({ audioUrl, role, reaction, onReact }) {
  const audioRef = useRef(null);
  const trackRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0); // 0..1
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [reactOpen, setReactOpen] = useState(false);

  useEffect(() => {
    const audio = new Audio(audioUrl);
    audioRef.current = audio;

    const onLoaded = () => setDuration(audio.duration || 0);
    const onTime = () => {
      if (dragging) return;
      setCurrentTime(audio.currentTime);
      if (audio.duration) setProgress(audio.currentTime / audio.duration);
    };
    const onEnd = () => {
      setPlaying(false);
      setProgress(0);
      setCurrentTime(0);
    };

    audio.addEventListener('loadedmetadata', onLoaded);
    audio.addEventListener('timeupdate', onTime);
    audio.addEventListener('ended', onEnd);

    return () => {
      audio.pause();
      audio.removeEventListener('loadedmetadata', onLoaded);
      audio.removeEventListener('timeupdate', onTime);
      audio.removeEventListener('ended', onEnd);
    };
  }, [audioUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) {
      audio.pause();
      setPlaying(false);
    } else {
      audio.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    }
  };

  // يحسب النسبة (0..1) حسب موقع الضغط/السحب داخل الشريط
  const ratioFromEvent = (e) => {
    const track = trackRef.current;
    if (!track) return 0;
    const rect = track.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    // الواجهة RTL، فالبداية يمين والنهاية يسار
    const ratio = (rect.right - clientX) / rect.width;
    return Math.min(1, Math.max(0, ratio));
  };

  const seekTo = (ratio) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const newTime = ratio * duration;
    audio.currentTime = newTime;
    setCurrentTime(newTime);
    setProgress(ratio);
  };

  const handlePointerDown = (e) => {
    setDragging(true);
    seekTo(ratioFromEvent(e));
  };

  const handlePointerMove = (e) => {
    if (!dragging) return;
    seekTo(ratioFromEvent(e));
  };

  const handlePointerUp = () => {
    setDragging(false);
  };

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = audioUrl;
    a.download = 'voice-message.wav';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setMenuOpen(false);
  };

  const pickReaction = (emoji) => {
    onReact(emoji);
    setReactOpen(false);
    setMenuOpen(false);
  };

  const shownTime = playing || progress > 0 ? currentTime : duration;

  return (
    <div className="voice-msg-wrap">
      <div className={`voice-msg ${role}`}>
        <button className="voice-msg-play" onClick={toggle}>
          {playing ? '⏸️' : '▶️'}
        </button>
        <div
          className="voice-msg-track"
          ref={trackRef}
          onMouseDown={handlePointerDown}
          onMouseMove={handlePointerMove}
          onMouseUp={handlePointerUp}
          onMouseLeave={handlePointerUp}
          onTouchStart={handlePointerDown}
          onTouchMove={handlePointerMove}
          onTouchEnd={handlePointerUp}
        >
          <div className="voice-msg-progress" style={{ width: `${progress * 100}%` }} />
          <div className="voice-msg-knob" style={{ right: `calc(${progress * 100}% - 6px)` }} />
        </div>
        <span className="voice-msg-time">{formatDuration(shownTime)}</span>

        <button
          className="voice-msg-menu-btn"
          onClick={() => setMenuOpen(prev => !prev)}
          title="خيارات"
        >
          ⋮
        </button>

        {menuOpen && (
          <div className="voice-msg-menu">
            {!reactOpen ? (
              <>
                <button className="voice-msg-menu-item" onClick={() => setReactOpen(true)}>
                  😊 تفاعل
                </button>
                <button className="voice-msg-menu-item" onClick={handleDownload}>
                  ⬇️ تنزيل
                </button>
              </>
            ) : (
              <div className="voice-msg-reactions">
                {REACTIONS.map(emoji => (
                  <button
                    key={emoji}
                    className="voice-msg-reaction-option"
                    onClick={() => pickReaction(emoji)}
                  >
                    {emoji}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {reaction && <div className={`voice-msg-badge ${role}`}>{reaction}</div>}
    </div>
  );
}

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
            kind: 'text',
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
      const reply = res.data.reply;
      setMessages([{ role: 'bot', kind: 'text', text: reply, time: getTime() }]);
    } catch {
      setError('تعذّر الاتصال بالخادم. تأكدي من تشغيل الـ Backend.');
    }
  };

  // نص عادي → جواب نصي عادي
  const sendMessage = async (text) => {
    const q = text.trim();
    if (!q || loading) return;
    setError('');
    setInput('');
    setMessages(prev => [...prev, { role: 'user', kind: 'text', text: q, time: getTime() }]);
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/chat`, {
        session_id: sessionId,
        message: q
      });
      setMessages(prev => [...prev, { role: 'bot', kind: 'text', text: res.data.reply, time: getTime() }]);
    } catch {
      setError('تعذّر الاتصال بالخادم. تأكدي من تشغيل الـ Backend.');
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  // سؤال صوتي → فقاعة صوتية للسؤال + فقاعة صوتية للجواب (زي واتساب)
  const sendVoiceMessage = async (transcribedText, userAudioUrl) => {
    const q = transcribedText.trim();
    if (!q) return;
    setError('');

    setMessages(prev => [...prev, {
      role: 'user', kind: 'voice', audioUrl: userAudioUrl, time: getTime()
    }]);

    setLoading(true);
    try {
      const chatRes = await axios.post(`${API_URL}/chat`, {
        session_id: sessionId,
        message: q
      });
      const reply = chatRes.data.reply;

      const ttsRes = await axios.post(
        `${API_URL}/voice/synthesize`,
        { text: reply },
        { responseType: 'blob' }
      );
      const botAudioUrl = URL.createObjectURL(ttsRes.data);

      setMessages(prev => [...prev, {
        role: 'bot', kind: 'voice', audioUrl: botAudioUrl, time: getTime()
      }]);
    } catch {
      setError('تعذّر الاتصال بالخادم. تأكدي من تشغيل الـ Backend.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const toggleRecording = () => {
    if (recording) {
      if (window._currentRecorder?.state === 'recording') {
        window._currentRecorder.stop();
      }
      setRecording(false);
      return;
    }

    setRecording(true);

    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = e => chunks.push(e.data);

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const userAudioUrl = URL.createObjectURL(blob);

        const formData = new FormData();
        formData.append('audio', blob, 'audio.webm');

        try {
          const res = await axios.post(`${API_URL}/voice/transcribe`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          if (res.data.text) {
            sendVoiceMessage(res.data.text, userAudioUrl);
          } else if (res.data.error) {
            setError(res.data.error);
          }
        } catch {
          setError('تعذّر تحويل الصوت إلى نص');
        }

        setRecording(false);
        stream.getTracks().forEach(t => t.stop());
        window._currentRecorder = null;
      };

      recorder.start();
      window._currentRecorder = recorder;

    }).catch(() => {
      setError('تعذّر الوصول للميكروفون');
      setRecording(false);
    });
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
              {msg.kind === 'voice' ? (
                <VoiceBubble
                  audioUrl={msg.audioUrl}
                  role={msg.role}
                  reaction={msg.reaction}
                  onReact={(emoji) => {
                    setMessages(prev => prev.map((m, idx) =>
                      idx === i ? { ...m, reaction: m.reaction === emoji ? null : emoji } : m
                    ));
                  }}
                />
              ) : (
                <div className={`bubble ${msg.role}`}>
                  {cleanMarkdown(msg.text).split('\n').map((line, j) => (
                    <p key={j} style={{ margin: j === 0 ? 0 : '4px 0 0' }}>{line}</p>
                  ))}
                </div>
              )}
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
            <span>جاري الاستماع... تكلم الآن</span>
            <div className="wave">
              <span /><span /><span /><span /><span />
            </div>
            <button className="voice-stop" onClick={() => {
              if (window._currentRecorder?.state === 'recording') {
                window._currentRecorder.stop();
              }
            }}>إيقاف</button>
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
          اضغط Enter للإرسال أو 🎙️ للتحدث · المعلومات مستخرجة من وثائق UCAS الرسمية
        </div>
      </footer>

    </div>
  );
}