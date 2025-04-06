import React, { useState } from 'react';
import DocumentWithGearIcon from './ui-component/DocumentWithGearIcon';

interface ConfigProps {
  onToggle: () => void;
}

const defaultConfig = {
  vocab_size: 50257,
  context_length: 1024,
  emb_dim: 768,
  n_heads: 12,
  n_layers: 12,
  drop_rate: 0.1,
  qkv_bias: false,
};

const Config: React.FC<ConfigProps> = ({ onToggle }) => {
  const [config, setConfig] = useState(defaultConfig);

  const handleChange = (key: keyof typeof config, value: string | boolean) => {
    let parsedValue: string | boolean | number = value;

    // boolean 처리
    if (value === 'true') parsedValue = true;
    else if (value === 'false') parsedValue = false;
    // 숫자 처리
    else if (!isNaN(Number(value)) && key !== 'qkv_bias')
      parsedValue = Number(value);

    setConfig((prev) => ({
      ...prev,
      [key]: parsedValue,
    }));
  };

  return (
    <aside className="absolute right-0 w-1/5 h-1/2 z-10 bg-white p-4 border-l shadow overflow-auto">
      {/* Config Header 영역 */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">GPT_CONFIG</h2>
        <button
          onClick={onToggle}
          className="p-2 bg-blue-100 rounded focus:outline-none shadow"
          aria-label="Toggle Config"
        >
          <DocumentWithGearIcon />
        </button>
      </div>

      {/* 설정 입력 영역 */}
      <div className="mt-4 space-y-4">
        {Object.entries(config).map(([key, value]) => (
          <div key={key} className="flex flex-col">
            <label className="text-sm font-medium mb-1 capitalize">{key}</label>
            {typeof value === 'boolean' ? (
              <select
                value={value.toString()}
                onChange={(e) =>
                  handleChange(key as keyof typeof config, e.target.value)
                }
                className="border p-2 rounded"
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <input
                type="text"
                value={value}
                onChange={(e) =>
                  handleChange(key as keyof typeof config, e.target.value)
                }
                className="border p-2 rounded"
              />
            )}
          </div>
        ))}
      </div>
    </aside>
  );
};

export default Config;
