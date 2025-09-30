import React, { useEffect, useState } from 'react';
import Header from './ui-component/Header';

interface Message {
  text: string;
  sender: 'user' | 'bot';
}

interface Model {
  fullName: string;
  displayName: string;
}

const TestPage: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // 백엔드 API를 통해 모델 목록을 가져옴
  useEffect(() => {
    fetch(`http://localhost:8000/api/v1/completed-models`)
      .then((res) => res.json())
      .then((data) => {
        const fetchedModels = data.models.map((model: any) => ({
          fullName: model.model_name,
          displayName: model.model_name.replace(/\.[^/.]+$/, ''),
        }));
        setModels(fetchedModels);
        // console.log('Fetched models:', fetchedModels);
      })
      .catch((err) => console.error('Failed to fetch models:', err));
  }, []);

  // 메시지 전송 함수
  const handleSendMessage = () => {
    if (input.trim() === '' || !selectedModel || isLoading) return;

    const newMessages: Message[] = [
      ...messages,
      { text: input, sender: 'user' },
    ];
    setMessages(newMessages);
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    fetch(`http://localhost:8000/api/v1/generate-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_name: selectedModel,
        input_text: currentInput,
      }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorData = await res
            .json()
            .catch(() => ({ detail: 'Invalid JSON response' }));
          throw new Error(errorData.detail || 'Network response was not ok');
        }
        return res.json();
      })
      .then((data) => {
        if (data.output_text) {
          const botMessage: Message = {
            text: data.output_text,
            sender: 'bot',
          };
          setMessages((prevMessages) => [...prevMessages, botMessage]);
        }
      })
      .catch((err) => {
        console.error('Failed to send message:', err);
        const errorMessage: Message = {
          text: err.message || 'Error: Could not get a response from the bot.',
          sender: 'bot',
        };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      })
      .finally(() => {
        setIsLoading(false);
      });
  };

  const selectedModelDisplayName =
    models.find((m) => m.fullName === selectedModel)?.displayName || null;

  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex flex-grow bg-gray-50">
        {/* Model List Sidebar */}
        <aside className="w-1/6 bg-white p-4 border-r">
          <h2 className="text-xl font-bold mb-4 text-gray-800">Models</h2>
          <ul className="space-y-2">
            {models.map((model) => (
              <li key={model.fullName}>
                <button
                  onClick={() => setSelectedModel(model.fullName)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    selectedModel === model.fullName
                      ? 'bg-blue-500 text-white font-semibold'
                      : 'hover:bg-gray-100 text-gray-700'
                  }`}
                >
                  {model.displayName}
                </button>
              </li>
            ))}
          </ul>
        </aside>

        {/* Chat Area */}
        <main className="flex-1 flex flex-col p-4">
          <div className="flex-1 mb-4 overflow-y-auto p-4 bg-white rounded-lg shadow-inner">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`mb-4 p-3 rounded-lg max-w-lg ${
                  msg.sender === 'user' ? 'bg-blue-100 ml-auto' : 'bg-gray-200'
                }`}
              >
                <p className="text-gray-800">{msg.text}</p>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-center items-center">
                <p className="text-gray-500">Bot is typing...</p>
              </div>
            )}
            {!selectedModel && messages.length === 0 && (
              <div className="flex items-center justify-center h-full">
                <p className="text-gray-500">
                  Select a model to start chatting.
                </p>
              </div>
            )}
          </div>
          <div className="flex items-center">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder={
                isLoading
                  ? 'Bot is thinking...'
                  : selectedModel
                    ? `Message ${selectedModelDisplayName}...`
                    : 'Select a model first'
              }
              className="flex-1 p-3 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={!selectedModel || isLoading}
            />
            <button
              onClick={handleSendMessage}
              className="bg-blue-500 text-white p-3 rounded-r-lg hover:bg-blue-600 disabled:bg-gray-400"
              disabled={!selectedModel || isLoading}
            >
              {isLoading ? '...' : 'Send'}
            </button>
          </div>
        </main>
      </div>
    </div>
  );
};

export default TestPage;
