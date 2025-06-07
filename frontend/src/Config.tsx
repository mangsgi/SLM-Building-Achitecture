import React, { useState } from 'react';
import ConfigButton from './ui-component/ConfigButton';
import InfoModal from './ui-component/InfoModal';

interface ConfigProps {
  onToggle: () => void;
  config: typeof defaultConfig;
  setConfig: React.Dispatch<React.SetStateAction<typeof defaultConfig>>;
}

const configMap: Record<keyof typeof defaultConfig, string> = {
  epochs: 'Epochs',
  batch_size: 'Batch Size',
  dtype: 'Data Type',
  vocab_size: 'Vocabulary Size',
  context_length: 'Context Length',
  emb_dim: 'Embedding Dimension',
  n_heads: 'Number of Heads',
  n_blocks: 'Number of Blocks',
  drop_rate: 'Dropout Rate',
  qkv_bias: 'QKV Bias',
};

const configDescriptions: Record<keyof typeof defaultConfig, string> = {
  epochs: '모델 학습을 반복할 횟수입니다.',
  batch_size: '한 번에 처리할 데이터의 크기입니다.',
  dtype: '모델의 데이터 타입을 지정합니다. (bf16, fp16, fp32)',
  vocab_size: '어휘 사전의 크기입니다.',
  context_length: '입력 시퀀스의 최대 길이입니다.',
  emb_dim: '임베딩 벡터의 차원입니다.',
  n_heads: '어텐션 헤드의 개수입니다.',
  n_blocks: '트랜스포머 블록의 개수입니다.',
  drop_rate: '드롭아웃 비율입니다. (0~1 사이의 값)',
  qkv_bias: 'Query, Key, Value 행렬에 편향을 추가할지 여부입니다.',
};

export const defaultConfig = {
  epochs: 10,
  batch_size: 8,
  dtype: 'bf16',
  vocab_size: 50257,
  context_length: 1024,
  emb_dim: 768,
  n_heads: 12,
  n_blocks: 12,
  drop_rate: 0.1,
  qkv_bias: false,
};

const Config: React.FC<ConfigProps> = ({ onToggle, config, setConfig }) => {
  const [showModal, setShowModal] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState<
    keyof typeof defaultConfig | null
  >(null);
  const dtypeOptions = ['bf16', 'fp16', 'fp32'];

  const handleChange = (key: keyof typeof config, value: string | boolean) => {
    let parsedValue: string | boolean | number = value;

    if (value === 'true') parsedValue = true;
    else if (value === 'false') parsedValue = false;
    else if (!isNaN(Number(value)) && key !== 'qkv_bias')
      parsedValue = Number(value);

    setConfig((prev) => ({
      ...prev,
      [key]: parsedValue,
    }));
  };

  // ✅ 입력 범위가 1 이하 소수인 경우 렌더링 및 입력 처리 for Dropout Rate
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

  // ✅ 소수점 입력 처리를 위한 키 목록
  const fractionalKeys: (keyof typeof config)[] = ['drop_rate'];

  return (
    <aside className="absolute right-0 w-[250px] h-1/2 z-10 bg-white p-4 shadow overflow-auto">
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
              <div className="flex items-center gap-2 mb-1">
                <label className="text-sm font-medium capitalize">
                  {configMap[typedKey]}
                </label>
                <button
                  onClick={() => {
                    setSelectedConfig(typedKey);
                    setShowModal(true);
                  }}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <i className="fas fa-info-circle"></i>
                </button>
              </div>

              {typedKey === 'dtype' ? (
                <select
                  value={value.toString()}
                  onChange={(e) => handleChange(typedKey, e.target.value)}
                  className="border p-2 rounded"
                >
                  {dtypeOptions.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              ) : fractionalKeys.includes(typedKey) &&
                typeof value === 'number' ? (
                renderFractionInput(typedKey, value, (k, v) =>
                  setConfig((prev) => ({ ...prev, [k]: v })),
                )
              ) : typeof value === 'boolean' ? (
                <select
                  value={value.toString()}
                  onChange={(e) => handleChange(typedKey, e.target.value)}
                  className="border p-2 rounded"
                >
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              ) : typeof value === 'number' ? (
                <input
                  type="text"
                  value={value}
                  onChange={(e) => handleChange(typedKey, e.target.value)}
                  className="border p-2 rounded"
                />
              ) : null}
            </div>
          );
        })}
      </div>

      {showModal && selectedConfig && (
        <InfoModal
          title={configMap[selectedConfig]}
          description={configDescriptions[selectedConfig]}
          onClose={() => setShowModal(false)}
        />
      )}
    </aside>
  );
};

export default Config;
