// --- 타입 정의 시작 ---
export type ModelType = 'GPT-2' | 'Llama2' | 'Llama3';

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

export const modelConfigs: Record<ModelType, Omit<ModelConfig, 'model'>> = {
  'GPT-2': gpt2Config,
  Llama2: llama2Config,
  Llama3: llama3Config,
};
