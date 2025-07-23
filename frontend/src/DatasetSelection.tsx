import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { FiInfo } from 'react-icons/fi';
import Modal from './ui-component/Modal';
import { ModelNode } from './App';

// 임시 데이터셋 목록
const datasets = [
  { id: 1, name: 'Dataset 1', description: 'First sample dataset' },
  { id: 2, name: 'Dataset 2', description: 'Second sample dataset' },
  { id: 3, name: 'Dataset 3', description: 'Third sample dataset' },
  { id: 4, name: 'Dataset 4', description: 'Fourth sample dataset' },
];

type Dataset = (typeof datasets)[0];

function DatasetSelection() {
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(
    null,
  );
  const [savePath, setSavePath] = useState('models'); // Default save path
  const [modalInfo, setModalInfo] = useState<{
    isOpen: boolean;
    title: string;
    description: string;
  } | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const { model, config } = location.state as {
    model: ModelNode[];
    config: Record<string, any>;
  };

  const handleShowInfo = (e: React.MouseEvent, dataset: Dataset) => {
    e.stopPropagation();
    setModalInfo({
      isOpen: true,
      title: dataset.name,
      description: dataset.description,
    });
  };

  const handleCloseModal = () => {
    setModalInfo(null);
  };

  const handleSubmit = async () => {
    if (!selectedDatasetId || !savePath) return;

    const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);
    if (!selectedDataset) return;

    try {
      const response = await fetch('/api/model/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          config: config,
          model: model,
          dataset: selectedDataset.name,
          savePath: savePath,
        }),
      });

      console.log(
        JSON.stringify({
          config,
          model,
          dataset: selectedDataset.name,
          savePath,
        }),
      );

      if (!response.ok) {
        throw new Error('Failed to submit model and dataset');
      }

      // 성공 시 처리
      console.log('Model and dataset submitted successfully');
      // 예를 들어, 학습 완료 페이지로 이동하거나 알림을 표시할 수 있습니다.
      navigate('/training-complete');
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
          <h2 className="text-3xl font-bold mb-8">1. Select Dataset</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className={`p-6 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedDatasetId === dataset.id
                    ? 'border-gray-600 bg-gray-50'
                    : 'border-gray-200 hover:border-gray-500'
                }`}
                onClick={() => setSelectedDatasetId(dataset.id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xl font-semibold">{dataset.name}</h3>
                  <button
                    onClick={(e) => handleShowInfo(e, dataset)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    <FiInfo size={20} />
                  </button>
                </div>
                <p className="text-gray-600">{dataset.description}</p>
              </div>
            ))}
          </div>

          <div className="mt-12">
            <h2 className="text-3xl font-bold mb-8">2. Set Save Directory</h2>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={savePath}
                onChange={(e) => setSavePath(e.target.value)}
                placeholder="e.g., models"
                className="flex-grow w-full p-3 border rounded-md"
              />
            </div>
            <p className="text-sm text-gray-500 mt-2">
              Please enter the desired save directory. Default is
              &apos;models&apos; inside the project root.
            </p>
          </div>

          <div className="mt-12 flex justify-end gap-4">
            <button
              onClick={() => navigate('/canvas')}
              className="px-6 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
            >
              Back
            </button>
            <button
              onClick={handleSubmit}
              disabled={!selectedDatasetId || !savePath}
              className={`px-6 py-2 rounded-md ${
                selectedDatasetId && savePath
                  ? 'bg-black text-white hover:bg-gray-600'
                  : 'bg-gray-300 text-gray-600 cursor-not-allowed'
              }`}
            >
              Submit
            </button>
          </div>
        </div>
      </div>
      {modalInfo && (
        <Modal isOpen={modalInfo.isOpen} onClose={handleCloseModal}>
          <h3 className="text-lg font-semibold mb-2">{modalInfo.title}</h3>
          <p className="text-sm">{modalInfo.description}</p>
        </Modal>
      )}
    </div>
  );
}

export default DatasetSelection;
