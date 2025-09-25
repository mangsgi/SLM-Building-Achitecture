import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { TrainingStatus } from '../store/statusSlice';

interface HeaderProps {
  children?: React.ReactNode;
}

const StatusIndicator: React.FC<{
  status: TrainingStatus;
  mlflowUrl: string | null;
  error: string | null; // 1. error prop 받기
}> = ({ status, mlflowUrl, error }) => {
  const statusConfig = {
    IDLE: {
      color: 'bg-green-500',
      shortMessage: 'Ready',
      longMessage: 'You can submit a new model for training.',
    },
    SUBMITTING: {
      color: 'bg-yellow-500',
      shortMessage: 'Submitting...',
      longMessage: 'Submitting the model structure to the backend.',
    },
    TRAINING: {
      color: 'bg-red-500',
      shortMessage: 'Training...',
      longMessage: 'A model is currently training. Check the progress below.',
    },
    COMPLETED: {
      color: 'bg-blue-500',
      shortMessage: 'Completed',
      longMessage: 'Training complete! You can now start a new one.',
    },
    ERROR: {
      color: 'bg-gray-500',
      shortMessage: 'Error',
      longMessage: error ?? 'An unknown error occurred.',
    },
  };

  const { color, shortMessage, longMessage } = statusConfig[status];

  return (
    <div className="relative flex items-center gap-2 group">
      <div className={`w-3 h-3 rounded-full ${color}`}></div>
      <span className="text-sm text-gray-600">{shortMessage}</span>
      {/* 4. 툴팁 위치 아래로 변경 및 z-index 추가 */}
      <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-max bg-gray-800 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-50">
        <p>{longMessage}</p>
        {status === 'TRAINING' && mlflowUrl && (
          <a
            href={mlflowUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:underline hover:text-white"
          >
            View in MLFlow
          </a>
        )}
      </div>
    </div>
  );
};

function Header({ children }: HeaderProps) {
  const navigate = useNavigate();
  // 5. Redux store에서 error 상태 가져오기
  const { trainingStatus, mlflowUrl, error } = useSelector(
    (state: RootState) => state.status,
  );

  return (
    <header className="bg-white p-4 shadow flex justify-between items-center">
      <div className="flex items-center gap-4">
        <h1
          className="text-2xl font-semibold text-left cursor-pointer"
          onClick={() => navigate('/canvas')}
        >
          Building Your Own SLM
        </h1>
        {/* 6. error prop 전달 */}
        <StatusIndicator
          status={trainingStatus}
          mlflowUrl={mlflowUrl}
          error={error}
        />
      </div>
      <div className="flex items-center gap-4">{children}</div>
    </header>
  );
}

export default Header;
