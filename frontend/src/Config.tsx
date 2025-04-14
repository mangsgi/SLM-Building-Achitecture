import React from 'react';
import ConfigButton from './ui-component/ConfigButton';

interface ConfigProps {
  onToggle: () => void;
  config: typeof defaultConfig;
  setConfig: React.Dispatch<React.SetStateAction<typeof defaultConfig>>;
}

export const defaultConfig = {
  vocab_size: 50257,
  context_length: 1024,
  emb_dim: 768,
  n_heads: 12,
  n_layers: 12,
  drop_rate: 0.1,
  qkv_bias: false,
  batch_size: 8,
};

const Config: React.FC<ConfigProps> = ({ onToggle, config, setConfig }) => {
  // Config 값이 바뀔 때 이벤트 핸들러
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

  // 입력 범위가 1 이하 소수인 경우 렌더링 및 입력 처리
  const renderFractionInput = (
    key: keyof typeof config,
    value: number,
    onChange: (key: keyof typeof config, value: number) => void,
  ) => {
    const fractional = value.toString().split('.')[1] || '';
    return (
      <div className="flex items-center border rounded px-2 py-1">
        <span className="text-gray-500 select-none">0.</span>
        <input
          type="text"
          inputMode="numeric"
          pattern="[0-9]*"
          placeholder="123"
          value={fractional}
          onChange={(e) => {
            const digitsOnly = e.target.value.replace(/\D/g, '').slice(0, 3);
            const newValue = digitsOnly ? parseFloat(`0.${digitsOnly}`) : 0;
            onChange(key, newValue);
          }}
          className="w-full p-1 outline-none text-sm"
        />
      </div>
    );
  };

  const fractionalKeys: (keyof typeof config)[] = ['drop_rate'];

  return (
    <aside className="absolute right-0 w-1/6 h-1/2 z-10 bg-white p-4 shadow overflow-auto">
      {/* Config Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Model Configuration</h2>
        <div onClick={onToggle} className="z-10" aria-label="Toggle Config">
          <ConfigButton />
        </div>
      </div>

      {/* 설정 입력 영역 */}
      <div className="mt-4 space-y-4">
        {Object.entries(config).map(([key, value]) => {
          const typedKey = key as keyof typeof config;

          return (
            <div key={key} className="flex flex-col">
              <label className="text-sm font-medium mb-1 capitalize">
                {key}
              </label>

              {fractionalKeys.includes(typedKey) &&
              typeof value === 'number' ? (
                // 소수점 입력일 때때
                renderFractionInput(typedKey, value, (k, v) =>
                  setConfig((prev) => ({ ...prev, [k]: v })),
                )
              ) : typeof value === 'boolean' ? (
                // boolean 입력일 때
                <select
                  value={value.toString()}
                  onChange={(e) => handleChange(typedKey, e.target.value)}
                  className="border p-2 rounded"
                >
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              ) : (
                // 기본 숫자(정수) 입력일 때
                <input
                  type="text"
                  value={value}
                  onChange={(e) => handleChange(typedKey, e.target.value)}
                  className="border p-2 rounded"
                />
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
};

export default Config;
