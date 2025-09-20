import React, { useState } from 'react';
import ConfigButton from './ui-component/ConfigButton';
import { FiInfo } from 'react-icons/fi';
import Modal from './ui-component/Modal';

// --- 타입 정의 시작 ---
type ModelType = 'GPT-2' | 'Llama2' | 'Llama3';

// 공통 설정값 인터페이스
interface BaseConfig {
  epochs: number;
  batch_size: number;
  vocab_size: number;
  context_length: number;
  emb_dim: number;
  n_heads: number;
  dtype: string;
}

// 모델별 고유 설정값 인터페이스
export interface GPT2Config extends BaseConfig {
  model: 'gpt-2';
  n_blocks: number;
  drop_rate: number;
  qkv_bias: boolean;
}

export interface Llama2Config extends BaseConfig {
  model: 'llama2';
  n_blocks: number;
  hidden_dim: number;
}

export interface Llama3Config extends BaseConfig {
  model: 'llama3';
  n_blocks: number;
  hidden_dim: number;
  n_kv_groups: number;
  rope_base: number;
  rope_freq: {
    factor: number;
    low_freq_factor: number;
    high_freq_factor: number;
    original_context_length: number;
  };
}

// 구별된 유니온 타입
export type ModelConfig = GPT2Config | Llama2Config | Llama3Config;
// --- 타입 정의 끝 ---

interface ConfigProps {
  onToggle: () => void;
  config: ModelConfig;
  setConfig: React.Dispatch<React.SetStateAction<ModelConfig>>;
}

// --- 설정값 객체들 ---
const gpt2Config: Omit<GPT2Config, 'model'> = {
  epochs: 1,
  batch_size: 1,
  vocab_size: 50257,
  context_length: 128,
  emb_dim: 768,
  n_heads: 12,
  n_blocks: 12,
  drop_rate: 0.1,
  qkv_bias: true,
  dtype: 'bf16',
};

const llama2Config: Omit<Llama2Config, 'model'> = {
  epochs: 1,
  batch_size: 1,
  vocab_size: 32000,
  context_length: 128,
  emb_dim: 4096,
  n_heads: 32,
  n_blocks: 32,
  hidden_dim: 11008,
  dtype: 'bf16',
};

const llama3Config: Omit<Llama3Config, 'model'> = {
  epochs: 1,
  batch_size: 1,
  vocab_size: 128256,
  context_length: 128,
  emb_dim: 4096,
  n_heads: 32,
  n_blocks: 32,
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

const modelConfigs: Record<ModelType, Omit<ModelConfig, 'model'>> = {
  'GPT-2': gpt2Config,
  Llama2: llama2Config,
  Llama3: llama3Config,
};

// --- 기타 설정 맵 ---
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
  hidden_dim: 'Llama2 모델의 히든 차원입니다.',
  n_kv_groups: 'Llama3 모델의 KV 그룹 개수입니다.',
  rope_base: 'RoPE 기본 값입니다.',
  rope_freq: 'RoPE 주파수 스케일링 값입니다.',
};

// --- 기본 설정값 ---
export const defaultConfig: Llama2Config = {
  ...llama2Config,
  model: 'llama2',
};

const Config: React.FC<ConfigProps> = ({ onToggle, config, setConfig }) => {
  const getModelTypeFromId = (modelId: ModelConfig['model']): ModelType => {
    if (modelId === 'gpt-2') return 'GPT-2';
    if (modelId === 'llama2') return 'Llama2';
    if (modelId === 'llama3') return 'Llama3';
    return 'GPT-2'; // Fallback
  };

  const [selectedModel, setSelectedModel] = useState<ModelType>(
    getModelTypeFromId(config.model),
  );
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalInfo, setModalInfo] = useState<{
    title: string;
    description: string;
  } | null>(null);
  const dtypeOptions = ['bf16', 'fp16', 'fp32'];

  // 모델 변경 이벤트 핸들러
  const handleModelChange = (model: ModelType) => {
    setSelectedModel(model);
    const modelId =
      model === 'GPT-2'
        ? 'gpt-2'
        : (model.toLowerCase() as ModelConfig['model']);
    const newConfigData = modelConfigs[model];

    const newConfig: ModelConfig = {
      ...newConfigData,
      model: modelId,
    } as ModelConfig;

    setConfig(newConfig);
  };

  // 정보 모달을 열기 위한 이벤트 핸들러
  const handleShowInfo = (title: string, description: string) => {
    setModalInfo({ title, description });
    setIsModalOpen(true);
  };

  // 정보 모달을 닫기 위한 이벤트 핸들러
  const handleCloseModal = () => {
    setIsModalOpen(false);
    setModalInfo(null);
  };

  // 설정 변경 이벤트 핸들러
  const handleChange = (key: string, value: string | boolean) => {
    if (value === 'true' || value === 'false') {
      setConfig(
        (prev) => ({ ...prev, [key]: value === 'true' }) as ModelConfig,
      );
      return;
    }

    if (key === 'dtype') {
      setConfig((prev) => ({ ...prev, [key]: value as string }) as ModelConfig);
      return;
    }

    const numValue = Number(value);
    if (!isNaN(numValue)) {
      setConfig((prev) => ({ ...prev, [key]: numValue }) as ModelConfig);
    }
  };

  // 중첩된 설정 변경 이벤트 핸들러
  const handleNestedChange = (
    parentKey: string,
    childKey: string,
    value: string,
  ) => {
    const numValue = Number(value);
    if (!isNaN(numValue) && value.trim() !== '') {
      setConfig(
        (prev) =>
          ({
            ...prev,
            [parentKey]: {
              ...(prev as any)[parentKey],
              [childKey]: numValue,
            },
          }) as ModelConfig,
      );
    }
  };

  // 분수 입력 렌더링 함수
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

  // 분수 키 목록
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
        {Object.entries(config)
          .filter(([key]) => key !== 'model')
          .map(([key, value]) => {
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
                ) : fractionalKeys.includes(key) &&
                  typeof value === 'number' ? (
                  renderFractionInput(key, value, (k, v) =>
                    setConfig((prev) => ({ ...prev, [k]: v }) as ModelConfig),
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
                  key === 'vocab_size' ? (
                    <input
                      type="text"
                      value={value}
                      readOnly
                      className="border p-2 rounded bg-gray-100 cursor-not-allowed"
                    />
                  ) : (
                    <input
                      type="text"
                      value={value}
                      onChange={(e) => handleChange(key, e.target.value)}
                      className="border p-2 rounded"
                    />
                  )
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
        <Modal
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          title={modalInfo.title}
          markdown={modalInfo.description}
        />
      )}
    </aside>
  );
};

export default Config;
