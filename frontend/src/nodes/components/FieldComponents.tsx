import React, { MouseEventHandler } from 'react';
import { FiInfo } from 'react-icons/fi';

// Node의 Title 컴포넌트
interface NodeTitleProps {
  children: string;
  onClick?: MouseEventHandler<HTMLDivElement>;
}

export const NodeTitle: React.FC<NodeTitleProps> = ({ children, onClick }) => {
  return (
    <div onClick={onClick}>
      <h3 className="font-bold text-center">{children}</h3>
    </div>
  );
};

// 읽기: Node의 각 Data별 렌더링
export const ReadField: React.FC<{
  label: string;
  value: string;
  info?: {
    title: string;
    description: string;
  };
  onInfoClick?: (info: { title: string; description: string }) => void;
}> = ({ label, value, info, onInfoClick }) => {
  return (
    <div className="mb-2 pt-1">
      <div className="flex items-center gap-2">
        <label className="text-base">{label}</label>
        {info && onInfoClick && (
          <button
            className="text-gray-400 hover:text-gray-600"
            onClick={() => onInfoClick(info)}
          >
            <FiInfo size={16} />
          </button>
        )}
      </div>
      <div className="border rounded p-1 text-sm w-full h-[30px]">
        {value || '-'}
      </div>
    </div>
  );
};

// 쓰기: (Input 태그) Node의 각 Data별 렌더링
export const EditField: React.FC<{
  label: string;
  id: string;
  name: string;
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
  info?: {
    title: string;
    description: string;
  };
  onInfoClick?: (info: { title: string; description: string }) => void;
}> = ({ label, id, name, value, placeholder, onChange, info, onInfoClick }) => {
  return (
    <div className="mb-2 pt-1">
      <div className="flex items-center gap-2">
        <label htmlFor={id} className="text-base font-medium">
          {label}
        </label>
        {info && onInfoClick && (
          <button
            className="text-gray-400 hover:text-gray-600"
            onClick={() => onInfoClick(info)}
          >
            <FiInfo size={16} />
          </button>
        )}
      </div>
      <input
        id={id}
        name={name}
        type="number"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="border rounded p-1 text-sm w-full h-[30px]"
      />
    </div>
  );
};

// 쓰기: (Select 태그) Node의 각 Data별 렌더링
export const EditSelectField: React.FC<{
  label: string;
  id: string;
  name: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  info?: {
    title: string;
    description: string;
  };
  onInfoClick?: (info: { title: string; description: string }) => void;
}> = ({ label, id, name, value, onChange, options, info, onInfoClick }) => {
  return (
    <div className="mb-2 pt-1">
      <div className="flex items-center gap-2">
        <label htmlFor={id} className="text-base font-medium">
          {label}
        </label>
        {info && onInfoClick && (
          <button
            className="text-gray-400 hover:text-gray-600"
            onClick={() => onInfoClick(info)}
          >
            <FiInfo size={16} />
          </button>
        )}
      </div>
      <select
        id={id}
        name={name}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border rounded p-1 text-sm w-full h-[30px]"
      >
        {options.map((opt) => (
          <option key={opt} value={opt} className="text-sm">
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
};
