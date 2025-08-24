import React, { useState } from 'react';
import ConfigButton from './ui-component/ConfigButton';
import { FiInfo } from 'react-icons/fi';
import Modal from './ui-component/Modal';

type ModelType = 'GPT-2' | 'Llama2' | 'Llama3';

interface ConfigProps {
  onToggle: () => void;
  config: Record<string, any>;
  setConfig: React.Dispatch<React.SetStateAction<Record<string, any>>>;
}

// GPT-2 124M
const gpt2Config = {
  epochs: 1,
  batch_size: 1,
  vocab_size: 50257,
  context_length: 128,
  emb_dim: 768,
  n_heads: 12,
  n_blocks: 12,
  drop_rate: 0.1,
  qkv_bias: false,
  dtype: 'bf16',
};

// Llama2 7B
const llama2Config = {
  epochs: 1,
  batch_size: 1,
  vocab_size: 32000,
  context_length: 128,
  emb_dim: 4096,
  n_heads: 32,
  n_layers: 32,
  hidden_dim: 11008,
  dtype: 'bf16',
};

// Llama3 8B
const llama3Config = {
  epochs: 1,
  batch_size: 1,
  vocab_size: 128256,
  context_length: 128,
  emb_dim: 4096,
  n_heads: 32,
  n_layers: 32,
  hidden_dim: 14336,
  n_kv_groups: 8,
  rope_base: 5000000,
  rope_freq: {
    factor: 8.0,
    low_freq_factor: 1.0,
    high_freq_factor: 4.0,
    original_context_length: 8192,
  },
  dtype: 'bf16',
};

const modelConfigs: Record<ModelType, Record<string, any>> = {
  'GPT-2': gpt2Config,
  Llama2: llama2Config,
  Llama3: llama3Config,
};

const configMap: Record<string, string> = {
  model: 'Model',
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
  n_layers: 'Number of Layers',
  hidden_dim: 'Hidden Dimension',
  n_kv_groups: 'Number of KV Groups',
  rope_base: 'RoPE Base',
  rope_freq: 'RoPE Frequency',
};

const configDescriptions: Record<string, string> = {
  model: '모델 유형을 선택합니다.',
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
  n_layers: 'Llama2 모델의 레이어 개수입니다.',
  hidden_dim: 'Llama2 모델의 히든 차원입니다.',
  n_kv_groups: 'Llama3 모델의 KV 그룹 개수입니다.',
  rope_base: 'RoPE 기본 값입니다.',
  rope_freq: 'RoPE 주파수 스케일링 값입니다.',
};

export const defaultConfig = {
  ...gpt2Config,
  model: 'gpt-2',
};

const Config: React.FC<ConfigProps> = ({ onToggle, config, setConfig }) => {
  const [selectedModel, setSelectedModel] = useState<ModelType>('GPT-2');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalInfo, setModalInfo] = useState<{
    title: string;
    description: string;
  } | null>(null);
  const dtypeOptions = ['bf16', 'fp16', 'fp32'];

  const handleModelChange = (model: ModelType) => {
    setSelectedModel(model);
    const modelId = model === 'GPT-2' ? 'gpt-2' : model.toLowerCase();
    setConfig({
      ...modelConfigs[model],
      model: modelId,
    });
  };

  const handleShowInfo = (title: string, description: string) => {
    setModalInfo({ title, description });
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setModalInfo(null);
  };

  const handleChange = (key: string, value: string | boolean) => {
    if (value === 'true' || value === 'false') {
      setConfig((prev) => ({ ...prev, [key]: value === 'true' }));
      return;
    }

    if (key === 'dtype') {
      setConfig((prev) => ({ ...prev, [key]: value as string }));
      return;
    }

    const numValue = Number(value);
    if (!isNaN(numValue)) {
      setConfig((prev) => ({ ...prev, [key]: numValue }));
    }
  };

  const handleNestedChange = (
    parentKey: string,
    childKey: string,
    value: string,
  ) => {
    const numValue = Number(value);
    if (!isNaN(numValue) && value.trim() !== '') {
      setConfig((prev: Record<string, any>) => ({
        ...prev,
        [parentKey]: {
          ...prev[parentKey],
          [childKey]: numValue,
        },
      }));
    }
  };

  const renderFractionInput = (
    key: string,
    value: number,
    onChange: (key: string, value: number) => void,
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

  const fractionalKeys: string[] = ['drop_rate'];

  return (
    <aside className="absolute right-0 w-[250px] h-1/2 z-10 bg-white p-4 shadow overflow-auto">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Model Configuration</h2>
        <div onClick={onToggle} className="z-10" aria-label="Toggle Config">
          <ConfigButton />
        </div>
      </div>

      <div className="mt-4">
        <div className="flex items-center gap-2 mb-1">
          <label className="text-sm font-medium">Model Type</label>
          <button
            onClick={() =>
              handleShowInfo(configMap['model'], configDescriptions['model'])
            }
            className="text-gray-500 hover:text-gray-700"
          >
            <FiInfo size={16} />
          </button>
        </div>
        <select
          value={selectedModel}
          onChange={(e) => handleModelChange(e.target.value as ModelType)}
          className="w-full border p-2 rounded mt-1"
        >
          <option value="GPT-2">GPT-2</option>
          <option value="Llama2">Llama2</option>
          <option value="Llama3">Llama3</option>
        </select>
      </div>

      <div className="mt-4 space-y-4">
        {Object.entries(config).map(([key, value]) => {
          return (
            <div key={key} className="flex flex-col">
              <div className="flex items-center gap-2 mb-1">
                <label className="text-sm font-medium capitalize">
                  {configMap[key]}
                </label>
                <button
                  onClick={() =>
                    handleShowInfo(configMap[key], configDescriptions[key])
                  }
                  className="text-gray-500 hover:text-gray-700"
                >
                  <FiInfo size={16} />
                </button>
              </div>

              {key === 'dtype' ? (
                <select
                  value={value.toString()}
                  onChange={(e) => handleChange(key, e.target.value)}
                  className="border p-2 rounded"
                >
                  {dtypeOptions.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              ) : fractionalKeys.includes(key) && typeof value === 'number' ? (
                renderFractionInput(key, value, (k, v) =>
                  setConfig((prev) => ({ ...prev, [k]: v })),
                )
              ) : typeof value === 'boolean' ? (
                <select
                  value={value.toString()}
                  onChange={(e) => handleChange(key, e.target.value)}
                  className="border p-2 rounded"
                >
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              ) : typeof value === 'number' ? (
                <input
                  type="text"
                  value={value}
                  onChange={(e) => handleChange(key, e.target.value)}
                  className="border p-2 rounded"
                />
              ) : typeof value === 'object' &&
                value !== null &&
                !Array.isArray(value) ? (
                <div className="pl-4 mt-2 border-l-2 border-gray-200 space-y-2">
                  {Object.entries(value).map(([childKey, childValue]) => (
                    <div key={childKey}>
                      <label className="text-sm font-medium capitalize text-gray-500">
                        {childKey.replace(/_/g, ' ')}
                      </label>
                      <input
                        type="text"
                        value={childValue as any}
                        onChange={(e) =>
                          handleNestedChange(key, childKey, e.target.value)
                        }
                        className="border p-2 rounded w-full mt-1"
                      />
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      {isModalOpen && modalInfo && (
        <Modal isOpen={isModalOpen} onClose={handleCloseModal}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">{modalInfo.title}</h3>
            <button
              onClick={handleCloseModal}
              className="text-gray-500 hover:text-gray-700"
            >
              <i className="fas fa-times"></i>
            </button>
          </div>
          <p className="text-gray-600">{modalInfo.description}</p>
        </Modal>
      )}
    </aside>
  );
};

export default Config;
