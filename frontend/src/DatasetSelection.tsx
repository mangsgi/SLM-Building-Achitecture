import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { FiInfo } from 'react-icons/fi';
import { useSelector, useDispatch } from 'react-redux';
import Header from './ui-component/Header';
import Spinner from './ui-component/Spinner';
import Modal from './ui-component/Modal';
import { ModelNode } from './App';
import { RootState, AppDispatch } from './store';
import {
  setSubmitting, // 임포트 추가
  startTraining,
  resetStatus,
  failTraining,
} from './store/statusSlice';
import { datasetInformation } from './constants/datasetInformation';

// 임시 데이터셋 목록
const datasets = [
  {
    id: 1,
    name: 'Tiny shakespeare',
    description: datasetInformation.tiny_shakespeare.description,
    short_description: 'Tiny shakespeare dataset',
    path: 'tiny_shakespeare',
    config: 'default',
  },
  {
    id: 2,
    name: 'OpenWebText-100k',
    description: datasetInformation.openwebtext_100k.description,
    short_description: 'OpenWebText 10만 샘플 슬라이스',
    path: 'mychen76/openwebtext-100k',
    config: 'default',
  },
  {
    id: 3,
    name: 'TinyStories',
    description: datasetInformation.tinystories.description,
    short_description: '단어 분포가 단순한 합성 동화 텍스트',
    path: 'roneneldan/TinyStories',
    config: 'default',
  },
  {
    id: 4,
    name: 'C4',
    description: datasetInformation.c4.description,
    short_description: 'T5 계열이 썼던 대형 웹코퍼스의 원조',
    path: 'allenai/c4',
    config: 'en',
  },
];

type Dataset = (typeof datasets)[0];

function DatasetSelection() {
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(
    null,
  );
  const [modelName, setModelName] = useState('my-slm-model');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalInfo, setModalInfo] = useState<{
    isOpen: boolean;
    title: string;
    description: string;
  } | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch: AppDispatch = useDispatch();
  const { trainingStatus, error } = useSelector(
    (state: RootState) => state.status,
  );

  const { model, config } = location.state as {
    model: ModelNode[];
    config: Record<string, any>;
  };

  const handleShowInfo = (e: React.MouseEvent, dataset: Dataset) => {
    e.stopPropagation();
    setIsModalOpen(true);
    setModalInfo({
      isOpen: true,
      title: 'Dataset Information',
      description: dataset.description,
    });
  };

  const handleCloseModal = () => {
    setModalInfo(null);
  };

  // 학습 제출 함수
  const handleSubmit = async () => {
    if (!selectedDatasetId || !modelName) return;

    const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);
    if (!selectedDataset) return;

    dispatch(setSubmitting()); // setSubmitting 활성화

    try {
      const response = await fetch(
        'http://localhost:8000/api/v1/train-complete-model',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            config: config,
            model: model,
            modelName: modelName,
            dataset: selectedDataset.path,
            dataset_config: selectedDataset.config,
          }),
        },
      );

      // console.log(
      //   JSON.stringify({
      //     config: config,
      //     model: model,
      //     modelName: modelName,
      //     dataset: selectedDataset.path,
      //     dataset_config: selectedDataset.config,
      //   }),
      // );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || 'Failed to submit model and dataset',
        );
      }

      const result = await response.json();
      const mlflowUrl = result.mlflow_url;
      if (mlflowUrl) {
        dispatch(startTraining({ mlflowUrl, task_id: result.task_id }));
        navigate('/canvas');
      } else {
        throw new Error(result.message || 'MLFlow URL not found in response');
      }
    } catch (error) {
      dispatch(failTraining({ message: (error as Error).message }));
    }
  };

  // 학습 취소 함수
  const handleCancel = async () => {
    try {
      const response = await fetch(
        'http://localhost:8000/api/v1/stop-training',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            task_id: localStorage.getItem('task_id') ?? null,
            force_kill: false,
          }),
        },
      );

      if (!response.ok) {
        throw new Error('Failed to cancel training');
      }

      dispatch(resetStatus());
    } catch (error) {
      // console.error('Error cancelling training:', error);
      alert((error as Error).message);
    }
  };

  const isActionInProgress =
    trainingStatus === 'SUBMITTING' || trainingStatus === 'TRAINING';

  return (
    <div className="flex flex-col w-full h-screen">
      {trainingStatus === 'SUBMITTING' && <Spinner />}
      <Header />

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
                <p className="text-gray-600">{dataset.short_description}</p>
              </div>
            ))}
          </div>

          <div className="mt-12">
            <h2 className="text-3xl font-bold mb-8">2. Set Model Name</h2>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g., my-slm-model"
                className="flex-grow w-full p-3 border rounded-md"
              />
            </div>
            <p className="text-sm text-gray-500 mt-2">
              Please enter the desired model name(Default is
              &apos;my-slm-model&apos;). Dataset will be saved in
              &apos;models&apos; directory.
            </p>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
              <div className="flex">
                <div className="py-1">
                  <FiInfo className="h-5 w-5 text-red-500 mr-3" />
                </div>
                <div>
                  <p className="font-bold">An error occurred:</p>
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            </div>
          )}

          <div className="mt-12 flex justify-end gap-4">
            <button
              onClick={() => navigate('/canvas')}
              className="px-6 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
            >
              Back
            </button>
            {trainingStatus === 'TRAINING' && (
              <button
                onClick={handleCancel}
                className="px-6 py-2 border border-red-500 text-red-500 rounded-md hover:bg-red-50"
              >
                Cancel Training
              </button>
            )}
            <button
              onClick={handleSubmit}
              disabled={!selectedDatasetId || !modelName || isActionInProgress}
              className={`px-6 py-2 rounded-md transition-colors ${
                !isActionInProgress && selectedDatasetId && modelName
                  ? 'bg-black text-white hover:bg-gray-800'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              {trainingStatus === 'SUBMITTING'
                ? 'Submitting...'
                : trainingStatus === 'TRAINING'
                  ? 'Training in Progress'
                  : 'Submit'}
            </button>
          </div>
        </div>
      </div>
      {isModalOpen && modalInfo && (
        <Modal
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          title={modalInfo.title}
          markdown={modalInfo.description}
        />
      )}
    </div>
  );
}

export default DatasetSelection;
