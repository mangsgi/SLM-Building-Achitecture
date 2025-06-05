import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// 임시 데이터셋 목록
const datasets = [
  { id: 1, name: 'Dataset 1', description: 'First sample dataset' },
  { id: 2, name: 'Dataset 2', description: 'Second sample dataset' },
  { id: 3, name: 'Dataset 3', description: 'Third sample dataset' },
  { id: 4, name: 'Dataset 4', description: 'Fourth sample dataset' },
];

function DatasetSelection() {
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null);
  const navigate = useNavigate();

  const handleSubmit = async () => {
    if (!selectedDataset) return;

    try {
      // TODO: 여기에 실제 모델과 선택된 데이터셋을 백엔드로 전송하는 로직 추가
      const response = await fetch('/api/model/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          datasetId: selectedDataset,
          // model은 전역 상태나 context에서 가져와야 합니다
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to submit model and dataset');
      }

      // 성공 시 처리
      console.log('Model and dataset submitted successfully');
    } catch (error) {
      console.error('Error submitting model and dataset:', error);
    }
  };

  return (
    <div className="flex flex-col w-full h-screen">
      <header className="bg-white p-4 shadow">
        <h1 className="text-2xl font-semibold text-left">
          Building Your Own SLM
        </h1>
      </header>

      <div className="flex-1 p-8">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Select Dataset</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className={`p-6 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedDataset === dataset.id
                    ? 'border-gray-600 bg-gray-50'
                    : 'border-gray-200 hover:border-gray-500'
                }`}
                onClick={() => setSelectedDataset(dataset.id)}
              >
                <h3 className="text-xl font-semibold mb-2">{dataset.name}</h3>
                <p className="text-gray-600">{dataset.description}</p>
              </div>
            ))}
          </div>

          <div className="mt-8 flex justify-end gap-4">
            <button
              onClick={() => navigate('/canvas')}
              className="px-6 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
            >
              Back
            </button>
            <button
              onClick={handleSubmit}
              disabled={!selectedDataset}
              className={`px-6 py-2 rounded-md ${
                selectedDataset
                  ? 'bg-black text-white hover:bg-gray-600'
                  : 'bg-gray-300 text-gray-600 cursor-not-allowed'
              }`}
            >
              Submit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DatasetSelection;
