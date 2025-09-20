import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { FiInfo } from 'react-icons/fi';
import { useSelector, useDispatch } from 'react-redux';
import Header from './ui-component/Header';
import Modal from './ui-component/Modal';
import { ModelNode } from './App';
import { RootState, AppDispatch } from './store';
import { startTraining, resetStatus, failTraining } from './store/statusSlice';

// 임시 데이터셋 목록
const datasets = [
  {
    id: 1,
    name: 'Tiny shakespeare',
    description: 'Tiny shakespeare dataset',
    path: 'tiny_shakespeare',
  },
  {
    id: 2,
    name: 'Dataset 2',
    description: 'Second sample dataset',
    path: 'dataset2',
  },
  {
    id: 3,
    name: 'Dataset 3',
    description: 'Third sample dataset',
    path: 'dataset3',
  },
  {
    id: 4,
    name: 'Dataset 4',
    description: 'Fourth sample dataset',
    path: 'dataset4',
  },
];

type Dataset = (typeof datasets)[0];

function DatasetSelection() {
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(
    null,
  );
  const [modelName, setModelName] = useState('my-slm-model');
  const [modalInfo, setModalInfo] = useState<{
    isOpen: boolean;
    title: string;
    description: string;
  } | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch: AppDispatch = useDispatch();
  const trainingStatus = useSelector(
    (state: RootState) => state.status.trainingStatus,
  );

  const { model, config } = location.state as {
    model: ModelNode[];
    config: Record<string, any>;
  };

  // 상태 초기화
  useEffect(() => {
    dispatch(resetStatus());
  }, [dispatch]);

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

  // 학습 제출 함수
  const handleSubmit = async () => {
    if (!selectedDatasetId || !modelName || trainingStatus === 'TRAINING')
      return;

    const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);
    if (!selectedDataset) return;

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
            dataset: selectedDataset.path,
            modelName: modelName,
            dataset_config:
              selectedDataset.path === 'tiny_shakespeare'
                ? 'default'
                : 'default',
          }),
        },
      );

      console.log(
        JSON.stringify({
          config: config,
          model: model,
          dataset: selectedDataset.path,
          modelName: modelName,
          dataset_config:
            selectedDataset.path === 'tiny_shakespeare' ? 'default' : 'default',
        }),
      );

      if (!response.ok) {
        const errorData = await response.json();
        const errorMessage =
          errorData.message || 'Failed to submit model and dataset';
        dispatch(failTraining({ message: errorMessage }));
        navigate('/canvas');
        return;
      }

      const result = await response.json();
      const mlflowUrl = result.mlflow_url; // 백엔드 응답에서 mlflow_url 추출
      console.log(result);

      // 성공 시 처리
      if (mlflowUrl) {
        dispatch(startTraining({ mlflowUrl, task_id: result.task_id }));
        console.log('Model and dataset submitted successfully');
        navigate('/canvas');
      } else {
        // 성공했지만 mlflow_url이 없는 경우
        const errorMessage =
          result.message || 'MLFlow URL not found in response';
        dispatch(failTraining({ message: errorMessage }));
        navigate('/canvas');
      }
    } catch (error) {
      console.error('Error submitting model and dataset:', error);
      dispatch(failTraining({ message: (error as Error).message }));
      navigate('/canvas');
    }
  };

  // 학습 취소 함수
  const handleCancel = async () => {
    try {
      // dispatch(resetStatus());
      // 백엔드에 학습 중단 요청
      const response = await fetch(
        'http://localhost:8000/api/v1/stop-training',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            task_id: localStorage.getItem('task_id') ?? null,
            force_kill: true,
          }),
        },
      );
      console.log(
        JSON.stringify({
          task_id: localStorage.getItem('task_id') ?? null,
          force_kill: false,
        }),
      );

      if (!response.ok) {
        throw new Error('Failed to cancel training');
      }

      // 성공 시 상태 초기화
      dispatch(resetStatus());
      console.log('Training cancelled successfully');
    } catch (error) {
      console.error('Error cancelling training:', error);
    }
  };

  return (
    <div className="flex flex-col w-full h-screen">
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
                <p className="text-gray-600">{dataset.description}</p>
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
              disabled={
                !selectedDatasetId ||
                !modelName ||
                trainingStatus === 'TRAINING'
              }
              className={`px-6 py-2 rounded-md ${
                selectedDatasetId && modelName && trainingStatus !== 'TRAINING'
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
