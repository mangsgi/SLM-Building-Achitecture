import React from 'react';

// Node의 Title 컴포넌트
export const NodeTitle: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return <h3 className="font-bold mb-2">{children}</h3>;
};

// 읽기: Node의 각 Data별 렌더링
export const ReadField: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => {
  return (
    <div className="mb-2 pt-1">
      <label className="text-base">{label}</label>
      <div className="border rounded p-1 text-sm w-full">{value || '-'}</div>
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
}> = ({ label, id, name, value, placeholder, onChange }) => {
  return (
    <div className="mb-2 pt-1">
      <label htmlFor={id} className="text-base font-medium">
        {label}
      </label>
      <input
        id={id}
        name={name}
        type="number"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="border rounded p-1 text-sm w-full"
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
}> = ({ label, id, name, value, onChange, options }) => {
  return (
    <div className="mb-2">
      <label htmlFor={id} className="text-sm">
        {label}
      </label>
      <select
        id={id}
        name={name}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border rounded p-1 text-sm w-full"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
};

// Save와 Edit 버튼
export const ActionButton: React.FC<{
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void;
  children: React.ReactNode;
  className?: string;
}> = ({ onClick, children, className = '' }) => {
  return (
    <button
      onClick={onClick}
      className={`bg-white p-2 hover:bg-green-100 rounded focus:outline-none focus:ring-0 ${className}`}
    >
      {children}
    </button>
  );
};
