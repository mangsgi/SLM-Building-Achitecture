import React, { useEffect, useState } from 'react';
import Header from './ui-component/Header';

interface Message {
  text: string;
  sender: 'user' | 'bot';
}

const TestPage: React.FC = () => {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  useEffect(() => {
    // 백엔드 API를 통해 모델 목록을 가져옵니다.
    // 현재는 백엔드 구현이 없으므로 임시로 하드코딩합니다.
    // fetch('/api/models')
    //   .then(res => res.json())
    //   .then(data => setModels(data.files))
    //   .catch(err => console.error("Failed to fetch models:", err));

    // 임시 모델 목록
    setModels(['model1.pth', 'model2.pth', 'model3.pth']);
  }, []);

  const handleSendMessage = () => {
    if (input.trim() === '') return;

    const newMessages: Message[] = [
      ...messages,
      { text: input, sender: 'user' },
    ];
    setMessages(newMessages);
    setInput('');

    // TODO: 백엔드에 메시지를 보내고 봇의 응답을 받는 로직 추가
    // 예: getBotResponse(input, selectedModel).then(botResponse => {
    //   setMessages([...newMessages, { text: botResponse, sender: 'bot' }]);
    // });
  };

  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex flex-grow bg-gray-50">
        {/* Model List Sidebar */}
        <aside className="w-1/4 bg-white p-4 border-r">
          <h2 className="text-xl font-bold mb-4 text-gray-800">Models</h2>
          <ul className="space-y-2">
            {models.map((model) => (
              <li key={model}>
                <button
                  onClick={() => setSelectedModel(model)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    selectedModel === model
                      ? 'bg-blue-500 text-white font-semibold'
                      : 'hover:bg-gray-100 text-gray-700'
                  }`}
                >
                  {model}
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
                selectedModel
                  ? `Message ${selectedModel}...`
                  : 'Select a model first'
              }
              className="flex-1 p-3 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={!selectedModel}
            />
            <button
              onClick={handleSendMessage}
              className="bg-blue-500 text-white p-3 rounded-r-lg hover:bg-blue-600 disabled:bg-gray-400"
              disabled={!selectedModel}
            >
              Send
            </button>
          </div>
        </main>
      </div>
    </div>
  );
};

export default TestPage;
