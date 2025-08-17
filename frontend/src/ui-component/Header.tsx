import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../store';
import { resetStatus, TrainingStatus } from '../store/statusSlice';

interface HeaderProps {
  children?: React.ReactNode;
}

const StatusIndicator: React.FC<{
  status: TrainingStatus;
  mlflowUrl: string | null;
}> = ({ status, mlflowUrl }) => {
  const statusConfig = {
    IDLE: {
      color: 'bg-green-500',
      shortMessage: 'Ready For Training',
      longMessage: 'You can submit a new model for training.',
    },
    TRAINING: {
      color: 'bg-red-500',
      shortMessage: 'Training...',
      longMessage: 'A model is currently training. Check the progress below.',
    },
    COMPLETED: {
      color: 'bg-blue-500',
      shortMessage: 'Completed Training',
      longMessage: 'Training complete! You can now start a new one.',
    },
  };

  const { color, shortMessage, longMessage } = statusConfig[status];

  return (
    <div className="relative flex items-center gap-2 group">
      <div className={`w-3 h-3 rounded-full ${color}`}></div>
      <span className="text-sm text-gray-600">{shortMessage}</span>
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max bg-gray-800 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <p>{longMessage}</p>
        {status === 'TRAINING' && mlflowUrl && (
          <a
            href={mlflowUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:underline"
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
  const location = useLocation();
  const dispatch: AppDispatch = useDispatch();
  const { trainingStatus, mlflowUrl } = useSelector(
    (state: RootState) => state.status,
  );

  useEffect(() => {
    if (trainingStatus === 'COMPLETED') {
      dispatch(resetStatus());
    }
  }, [location.pathname, dispatch, trainingStatus]);

  return (
    <header className="bg-white p-4 shadow flex justify-between items-center">
      <div className="flex items-center gap-4">
        <h1
          className="text-2xl font-semibold text-left cursor-pointer"
          onClick={() => navigate('/canvas')}
        >
          Building Your Own SLM
        </h1>
        <StatusIndicator status={trainingStatus} mlflowUrl={mlflowUrl} />
      </div>
      <div className="flex items-center gap-4">{children}</div>
    </header>
  );
}

export default Header;
