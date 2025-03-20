import React from 'react';

import { Option } from './NodeData';

export const NodeTitle: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return <h3 className="font-bold">{children}</h3>;
};

export const ReadField: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => {
  return (
    <p className="text-sm">
      {label} {value || 'Not set'}
    </p>
  );
};

export const EditField: React.FC<{
  label: string;
  id: string;
  name: string;
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
}> = ({ label, id, name, value, placeholder, onChange }) => {
  return (
    <div className="mb-2">
      <label htmlFor={id} className="text-sm">
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

export const ActionButton: React.FC<{
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void;
  children: React.ReactNode;
  className?: string;
}> = ({ onClick, children, className = '' }) => {
  return (
    <button
      onClick={onClick}
      className={`mt-2 px-2 py-1 rounded text-sm ${className}`}
    >
      {children}
    </button>
  );
};

export const EditSelectField: React.FC<{
  label: string;
  id: string;
  name: string;
  value: string;
  onChange: (value: string) => void;
  options: Option[];
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
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
};
